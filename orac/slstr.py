"""Filename parsing, granule discovery and IO for ORAC SLSTR L2 output.

The data tree is::

    <root>/YYYY/MM/DD/<absolute_orbit>/
        C3S-312bL1-L2-CLOUD-CLD-SLSTR_ORAC_Sentinel3a_<YYYYMMDDhhmm>_<orbit>_R9999.primary.nc
        C3S-312bL1-L2-CLOUD-CLD-SLSTR_ORAC_Sentinel3a_<YYYYMMDDhhmm>_<orbit>_R9999.secondary.nc
        C3S-312bL1-L2-CLOUD-CLD-SLSTR_ORAC_Sentinel3a_<YYYYMMDDhhmm>_<orbit>_R9999.bugsrad.nc

Unlike the geostationary SEVIRI product (a single fixed 3712² disk sampled every
15 min), SLSTR is a **polar-orbiter swath**: each ``*.primary.nc`` is an
independent granule of shape ``(along_track≈1200, across_track≈1500)`` with 2-D
``lat``/``lon``/``time``. A granule spans ~3 min of orbit; the filename timestamp
is the granule start. A single retrieval stream (``R9999``) — there is no
R10/R11 split as there was for SEVIRI.

The ``done/`` sibling tree holds only zero-byte completion markers; the real
NetCDFs are under ``l2b/``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr

_FILENAME_RE = re.compile(
    r"^C3S-312bL1-L2-CLOUD-CLD-SLSTR_ORAC_(?P<platform>[A-Za-z0-9]+)"
    r"_(?P<ts>\d{12})_(?P<orbit>\d+)_(?P<retrieval>R\d+)\.(?P<kind>primary|secondary|bugsrad)\.nc$"
)


@dataclass(frozen=True)
class SlstrGranule:
    """A single ORAC SLSTR L2 granule (primary + optional secondary)."""

    primary: Path
    start_time: datetime          # granule start (from filename), UTC
    orbit: int                    # absolute orbit number
    platform: str                 # e.g. "Sentinel3a"

    @property
    def secondary(self) -> Path:
        return self.primary.with_name(self.primary.name.replace(".primary.nc", ".secondary.nc"))

    @property
    def stem(self) -> str:
        """``<YYYYMMDDhhmm>_<orbit>`` — stable per-granule id."""
        return f"{self.start_time:%Y%m%d%H%M}_{self.orbit}"

    @property
    def pixel_key(self) -> int:
        """Integer key encoding the granule start minute, for globally-unique
        ``sev_pixel_id`` construction downstream (``key * 1_800_000 + along*ncols + across``)."""
        return int(self.start_time.strftime("%Y%m%d%H%M"))


def parse_slstr_filename(path: str | Path) -> SlstrGranule:
    """Parse an ORAC SLSTR primary filename into a :class:`SlstrGranule`."""
    path = Path(path)
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Not an ORAC SLSTR filename: {path.name}")
    if m.group("kind") != "primary":
        raise ValueError(f"Expected a .primary.nc file, got: {path.name}")
    ts = datetime.strptime(m.group("ts"), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    return SlstrGranule(
        primary=path,
        start_time=ts,
        orbit=int(m.group("orbit")),
        platform=m.group("platform"),
    )


def discover_granules(
    root: str | Path,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[SlstrGranule]:
    """Walk ``root/YYYY/MM/DD/<orbit>`` and return one granule per ``*.primary.nc``.

    ``start`` / ``end`` bound the granule **start** time (inclusive / exclusive,
    UTC). Because a granule spans ~3 min, callers matching against a time window
    should widen ``[start, end)`` by the granule duration plus their tolerance.
    """
    root = Path(root)
    if start is not None and start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end is not None and end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    out: list[SlstrGranule] = []
    for p in root.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*/*.primary.nc"):
        try:
            g = parse_slstr_filename(p)
        except ValueError:
            continue
        if start is not None and g.start_time < start:
            continue
        if end is not None and g.start_time >= end:
            continue
        out.append(g)
    return sorted(out, key=lambda g: (g.start_time, g.orbit))


def open_granule(
    granule: SlstrGranule,
    variables: Iterable[str] | None = None,
    include_secondary: bool = True,
) -> xr.Dataset:
    """Open one granule as a merged primary[+secondary] dataset.

    ``mask_and_scale=True`` decodes packed shorts; ``decode_times=False`` leaves
    the Julian-day ``time`` variable raw (decode lazily with
    :func:`orac.io.julian_to_datetime`).
    """
    vars_list = list(variables) if variables is not None else None

    def _open(path: Path) -> xr.Dataset:
        ds = xr.open_dataset(path, mask_and_scale=True, decode_times=False)
        if vars_list is not None:
            keep = [v for v in vars_list if v in ds.variables]
            ds = ds[keep] if keep else ds[[]]
        return ds

    ds = _open(granule.primary)
    if include_secondary and granule.secondary.exists():
        ds2 = _open(granule.secondary)
        dupes = [v for v in ds2.data_vars if v in ds.variables]
        if dupes:
            ds2 = ds2.drop_vars(dupes)
        ds = xr.merge([ds, ds2], compat="override")
    ds.attrs["start_time"] = granule.start_time.isoformat()
    ds.attrs["orbit"] = granule.orbit
    ds.attrs["source_path"] = str(granule.primary)
    return ds


def granule_centroid(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (unit-vector centroid, angular radius in radians) of a swath.

    Used to cheaply reject granules that cannot overlap an EarthCARE frame before
    the expensive nearest-pixel query. Robust across the dateline / poles because
    it works in 3-D Cartesian space.
    """
    xyz = lla_to_unit_xyz(lat.ravel(), lon.ravel())
    good = np.isfinite(xyz).all(axis=1)
    xyz = xyz[good]
    if xyz.shape[0] == 0:
        return np.array([np.nan, np.nan, np.nan]), np.nan
    c = xyz.mean(axis=0)
    n = np.linalg.norm(c)
    if n == 0:
        return c, np.pi
    c = c / n
    cos_ang = np.clip(xyz @ c, -1.0, 1.0)
    radius = float(np.arccos(cos_ang.min()))
    return c, radius


def lla_to_unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Geographic (deg) → unit vectors on the sphere, shape ``(n, 3)``.

    Dateline- and pole-safe; the Euclidean distance between two unit vectors is a
    monotone function of great-circle distance, so a Euclidean KD-tree returns the
    true nearest neighbour everywhere.
    """
    phi = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lam = np.deg2rad(np.asarray(lon, dtype=np.float64))
    cphi = np.cos(phi)
    return np.column_stack([cphi * np.cos(lam), cphi * np.sin(lam), np.sin(phi)])
