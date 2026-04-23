"""Lazy xarray readers for ORAC SEVIRI L2 files.

Design rules:

* ``mask_and_scale=True`` — xarray applies ``scale_factor`` and ``add_offset``
  automatically, so ``stemp`` (``add_offset=100``) is correctly decoded.
* ``decode_times=False`` — the ``time`` variable uses a non-standard Julian-day
  reference (``"days since -4712-01-01 12:00:00"``). Decoding a 3712² array on
  every open is wasteful, so callers decode lazily via :func:`julian_to_datetime`.
* Primary + secondary are merged on ``(along_track, across_track)``. Variable
  names do not collide between the two.
* ``open_paired`` stacks R10 and R11 on a new ``retrieval`` dim, with R10 first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import xarray as xr

from .metadata import Retrieval, SlotRecord

# Julian-day offset for the Unix epoch (1970-01-01 00:00 UTC).
_JULIAN_UNIX_EPOCH = 2440587.5


def _open(path: Path, variables: Sequence[str] | None) -> xr.Dataset:
    ds = xr.open_dataset(
        path,
        mask_and_scale=True,
        decode_times=False,
        chunks={"along_track": 928, "across_track": 928},
    )
    if variables is not None:
        keep = [v for v in variables if v in ds.variables]
        ds = ds[keep] if keep else ds[[]]
    return ds


def open_slot(
    slot: SlotRecord,
    retrieval: Retrieval,
    variables: Iterable[str] | None = None,
    include_secondary: bool = True,
) -> xr.Dataset:
    """Open one ``(slot, retrieval)`` pair as a merged primary[+secondary] dataset.

    Raises :class:`FileNotFoundError` if the requested files are missing.
    """
    primary = slot.get(retrieval, "primary")
    if primary is None:
        raise FileNotFoundError(f"{retrieval} primary missing for slot {slot.scan_time:%Y-%m-%d %H:%M}")
    vars_list = list(variables) if variables is not None else None
    ds = _open(primary.path, vars_list)
    if include_secondary:
        secondary = slot.get(retrieval, "secondary")
        if secondary is not None:
            ds2 = _open(secondary.path, vars_list)
            # Drop any duplicate variables that appear in both files.
            dupes = [v for v in ds2.data_vars if v in ds.variables]
            if dupes:
                ds2 = ds2.drop_vars(dupes)
            ds = xr.merge([ds, ds2], compat="override")
    ds.attrs["retrieval"] = retrieval
    ds.attrs["scan_time"] = slot.scan_time.isoformat()
    ds.attrs["source_path"] = str(primary.path)
    return ds


def open_paired(
    slot: SlotRecord,
    variables: Iterable[str] | None = None,
    include_secondary: bool = True,
) -> xr.Dataset:
    """Open R10 and R11 for one slot, concatenated along a new ``retrieval`` dim.

    Only returns retrievals that are present for this slot. Raises
    :class:`FileNotFoundError` if neither R10 nor R11 has a primary file.
    """
    present: list[tuple[str, xr.Dataset]] = []
    for r in ("R10", "R11"):
        if slot.get(r, "primary") is not None:  # type: ignore[arg-type]
            present.append((r, open_slot(slot, r, variables, include_secondary)))  # type: ignore[arg-type]
    if not present:
        raise FileNotFoundError(f"No retrieval present for slot {slot.scan_time:%Y-%m-%d %H:%M}")
    labels, datasets = zip(*present)
    combined = xr.concat(datasets, dim="retrieval")
    combined = combined.assign_coords(retrieval=list(labels))
    combined.attrs["scan_time"] = slot.scan_time.isoformat()
    return combined


def julian_to_datetime(julian_days: np.ndarray | xr.DataArray) -> np.ndarray:
    """Convert the per-pixel ``time`` variable (Julian days) to ``datetime64[ns]``.

    Values below ~2 million are treated as fills and mapped to NaT.
    """
    arr = np.asarray(julian_days, dtype=np.float64)
    out = np.full(arr.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = np.isfinite(arr) & (arr > 2_000_000.0)
    if valid.any():
        seconds = (arr[valid] - _JULIAN_UNIX_EPOCH) * 86400.0
        out[valid] = (np.datetime64("1970-01-01T00:00:00", "ns") +
                      (seconds * 1e9).astype("timedelta64[ns]"))
    return out


def read_prior_file(slot: SlotRecord, retrieval: Retrieval = "R11") -> str | None:
    """Return the ``Prior_File`` global attribute for one retrieval (or ``None``)."""
    of = slot.get(retrieval, "primary")
    if of is None:
        return None
    with xr.open_dataset(of.path, decode_cf=False) as ds:
        val = ds.attrs.get("Prior_File", "")
    return val or None
