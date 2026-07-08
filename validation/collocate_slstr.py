"""Sample-level collocation: EarthCARE nadir track → ORAC SLSTR swath pixel.

The SEVIRI analogue (``collocate.py``) exploits a fixed geostationary grid that
is *always* available, so it just picks the nearest 15-min slot and matches every
on-disk profile. SLSTR is a polar-orbiter swath: EarthCARE and Sentinel-3A only
coincide near orbit-track crossings, so here the **temporal window is the binding
constraint** and most profiles do not match at all.

Per EarthCARE frame this module:

1. selects SLSTR granules whose start time is within the frame's time span
   ± ``max_time_diff`` (+ the ~3-min granule duration);
2. prunes granules that cannot overlap the frame using a cheap corner-sampled
   centroid/radius catalogue (3-D, dateline/pole-safe);
3. for each surviving granule, builds a unit-vector KD-tree on its valid pixels
   and finds the nearest pixel for every eligible profile;
4. keeps, per profile, the temporally-valid spatially-nearest pixel and samples
   the requested ORAC variables there.

The output DataFrame uses the **same column names as ``match_track_to_seviri``**
(``sev_*``, ``distance_km``, ``time_diff_s``, ``valid_match`` …) so the existing
``statistics.py`` / ``figures.py`` / ``track_figures.py`` consume it unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from orac.io import julian_to_datetime
from orac.slstr import SlstrGranule, discover_granules, lla_to_unit_xyz, open_granule

_EARTH_R_KM = 6371.0
# A granule spans ~3 min; pad the temporal discovery window by this much so a
# granule that *starts* just before the window can still cover a profile.
_GRANULE_DUR_S = 240.0
# Pixels sit ~1 km apart at nadir. A profile genuinely inside the swath has a
# sub-km nearest pixel; anything beyond this is off-swath (kd-tree still returns
# a far edge pixel). Doubles as the spatial "on-swath" gate.
_DEFAULT_MAX_PIXEL_DIST_KM = 3.0


def _chord_to_km(chord: np.ndarray) -> np.ndarray:
    """Euclidean distance between unit vectors → great-circle km."""
    central = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
    return _EARTH_R_KM * central


@dataclass
class GranuleCatalog:
    """Corner-sampled spatial index over a set of SLSTR granules.

    Holds one unit-vector centroid and angular radius (radians) per granule so
    candidate granules can be rejected for an EarthCARE frame without reading
    their full geolocation.
    """

    granules: list[SlstrGranule]
    centroids: np.ndarray   # (m, 3) unit vectors
    radii: np.ndarray       # (m,) radians
    start_times: np.ndarray  # (m,) datetime64[ns]


def build_granule_catalog(
    granules: Sequence[SlstrGranule], stride: int = 64
) -> GranuleCatalog:
    """Build a :class:`GranuleCatalog` by reading a strided lat/lon subsample.

    ``stride`` controls the coarse grid read from each granule (default reads
    ~19×24 points from the 1200×1500 swath — enough for a centroid and a padded
    radius). Unreadable granules are dropped.
    """
    keep: list[SlstrGranule] = []
    cents: list[np.ndarray] = []
    radii: list[float] = []
    starts: list[np.datetime64] = []
    for g in granules:
        try:
            with h5py.File(g.primary, "r") as f:
                lat = np.asarray(f["lat"][::stride, ::stride], dtype=np.float64)
                lon = np.asarray(f["lon"][::stride, ::stride], dtype=np.float64)
        except Exception:  # noqa: BLE001 — a corrupt granule just drops out
            continue
        xyz = lla_to_unit_xyz(lat.ravel(), lon.ravel())
        good = np.isfinite(xyz).all(axis=1)
        xyz = xyz[good]
        if xyz.shape[0] == 0:
            continue
        c = xyz.mean(axis=0)
        n = np.linalg.norm(c)
        if n == 0:
            continue
        c = c / n
        cos_ang = np.clip(xyz @ c, -1.0, 1.0)
        # Pad the radius: the coarse subsample misses the true swath corners.
        radius = float(np.arccos(cos_ang.min())) + np.deg2rad(1.0)
        keep.append(g)
        cents.append(c)
        radii.append(radius)
        starts.append(np.datetime64(g.start_time.replace(tzinfo=None), "ns"))
    if not keep:
        return GranuleCatalog([], np.empty((0, 3)), np.empty((0,)),
                              np.empty((0,), dtype="datetime64[ns]"))
    return GranuleCatalog(keep, np.array(cents), np.array(radii),
                          np.array(starts, dtype="datetime64[ns]"))


def match_track_to_slstr(
    ec_lat: np.ndarray,
    ec_lon: np.ndarray,
    ec_time: np.ndarray,
    slstr_root: str | Path,
    orac_vars: Iterable[str] = (),
    max_time_diff_seconds: float = 1800.0,
    max_pixel_dist_km: float = _DEFAULT_MAX_PIXEL_DIST_KM,
    catalog: GranuleCatalog | None = None,
) -> pd.DataFrame:
    """Match each EarthCARE profile to its nearest in-time SLSTR pixel.

    Parameters
    ----------
    ec_lat, ec_lon
        ``(n_profile,)`` geographic coordinates (deg).
    ec_time
        ``(n_profile,)`` ``datetime64[ns]`` UTC.
    slstr_root
        Root of the ORAC SLSTR ``l2b`` archive (``<root>/YYYY/MM/DD/<orbit>/…``).
    orac_vars
        ORAC variable names to sample at each matched pixel (raw names; the
        caller renames to the ``*_orac`` convention).
    max_time_diff_seconds
        Profiles whose matched pixel is further than this in time get
        ``valid_match=False``. Default 30 min (SLSTR crossings are sparse; the
        binding parameter — tune via the Δt sweep).
    max_pixel_dist_km
        Nearest-pixel distance beyond which a profile is treated as off-swath.
    catalog
        Optional prebuilt :class:`GranuleCatalog` (share one across all frames of
        a run). If ``None`` it is built from the granules in the time window.

    Returns
    -------
    pandas.DataFrame
        One row per EarthCARE profile, SEVIRI-schema-compatible columns plus one
        column per ``orac_vars`` entry.
    """
    ec_lat = np.asarray(ec_lat, dtype=np.float64)
    ec_lon = np.asarray(ec_lon, dtype=np.float64)
    ec_time = np.asarray(ec_time, dtype="datetime64[ns]")
    n = ec_lat.shape[0]
    orac_vars = tuple(orac_vars)

    valid_t = ~np.isnat(ec_time)
    if not valid_t.any():
        return _empty_result(ec_lat, ec_lon, ec_time, orac_vars)

    t_lo = ec_time[valid_t].min()
    t_hi = ec_time[valid_t].max()
    pad = timedelta(seconds=max_time_diff_seconds + _GRANULE_DUR_S)
    win_lo = pd.Timestamp(t_lo).floor("us").to_pydatetime().replace(tzinfo=timezone.utc) - pad
    win_hi = pd.Timestamp(t_hi).floor("us").to_pydatetime().replace(tzinfo=timezone.utc) + pad

    if catalog is None:
        granules = discover_granules(slstr_root, win_lo, win_hi)
        catalog = build_granule_catalog(granules)

    # Time-window candidates from the catalogue.
    max_dt = np.timedelta64(int(max_time_diff_seconds + _GRANULE_DUR_S), "s")
    if catalog.start_times.size:
        cand_mask = (catalog.start_times >= np.datetime64(win_lo.replace(tzinfo=None), "ns")) & \
                    (catalog.start_times <= np.datetime64(win_hi.replace(tzinfo=None), "ns"))
        cand_idx = np.flatnonzero(cand_mask)
    else:
        cand_idx = np.empty(0, dtype=int)

    # EarthCARE frame centroid for coarse spatial rejection.
    ec_xyz = lla_to_unit_xyz(ec_lat, ec_lon)
    ec_good = np.isfinite(ec_xyz).all(axis=1)
    ec_c = ec_xyz[ec_good].mean(axis=0)
    ec_c = ec_c / np.linalg.norm(ec_c)
    ec_radius = float(np.arccos(np.clip((ec_xyz[ec_good] @ ec_c).min(), -1.0, 1.0)))

    # Per-profile running best. Prefer time-valid, then spatially nearest.
    best_dist = np.full(n, np.inf)
    best_invalid = np.ones(n, dtype=bool)   # current match violates the time gate
    best_along = np.full(n, -1, dtype=np.int64)
    best_across = np.full(n, -1, dtype=np.int64)
    best_slat = np.full(n, np.nan)
    best_slon = np.full(n, np.nan)
    best_ptime = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    best_tdiff = np.full(n, np.nan)
    best_pixkey = np.full(n, -1, dtype=np.int64)
    best_ncols = np.full(n, -1, dtype=np.int64)
    matched_any = np.zeros(n, dtype=bool)
    var_best: dict[str, np.ndarray] = {v: np.full(n, np.nan) for v in orac_vars}

    for ci in cand_idx:
        g = catalog.granules[ci]
        # Coarse spatial reject.
        sep = float(np.arccos(np.clip(ec_c @ catalog.centroids[ci], -1.0, 1.0)))
        if sep > ec_radius + catalog.radii[ci] + np.deg2rad(1.0):
            continue

        # Eligible profiles: within the time window of this granule.
        g_t0 = catalog.start_times[ci]
        elig = valid_t & (np.abs(ec_time - g_t0) <= (max_dt + np.timedelta64(int(_GRANULE_DUR_S), "s")))
        if not elig.any():
            continue

        ds = open_granule(g, variables=("lat", "lon", "time", *orac_vars))
        sl_lat = np.asarray(ds["lat"].values, dtype=np.float64)
        sl_lon = np.asarray(ds["lon"].values, dtype=np.float64)
        sl_time = julian_to_datetime(ds["time"].values)
        nrow, ncol = sl_lat.shape

        finite = np.isfinite(sl_lat) & np.isfinite(sl_lon)
        flat_idx = np.flatnonzero(finite.ravel())
        if flat_idx.size == 0:
            ds.close()
            continue
        pix_xyz = lla_to_unit_xyz(sl_lat.ravel()[flat_idx], sl_lon.ravel()[flat_idx])
        tree = cKDTree(pix_xyz)

        eidx = np.flatnonzero(elig)
        q_xyz = ec_xyz[eidx]
        chord, k = tree.query(q_xyz, k=1)
        dist_km = _chord_to_km(chord)
        pixel_flat = flat_idx[k]
        at_i, ac_i = np.unravel_index(pixel_flat, (nrow, ncol))

        p_time = sl_time.ravel()[pixel_flat]
        tdiff = np.abs((ec_time[eidx] - p_time) / np.timedelta64(1, "s")).astype(np.float64)

        on_swath = dist_km <= max_pixel_dist_km
        cand_invalid = tdiff > max_time_diff_seconds
        # Better if: on-swath AND (becomes valid where we were invalid,
        # or same validity but closer).
        improves = on_swath & (
            (cand_invalid.astype(int) < best_invalid[eidx].astype(int))
            | ((cand_invalid == best_invalid[eidx]) & (dist_km < best_dist[eidx]))
        )
        upd = eidx[improves]
        if upd.size == 0:
            ds.close()
            continue

        best_dist[upd] = dist_km[improves]
        best_invalid[upd] = cand_invalid[improves]
        best_along[upd] = at_i[improves]
        best_across[upd] = ac_i[improves]
        best_slat[upd] = sl_lat[at_i[improves], ac_i[improves]]
        best_slon[upd] = sl_lon[at_i[improves], ac_i[improves]]
        best_ptime[upd] = p_time[improves]
        best_tdiff[upd] = tdiff[improves]
        best_pixkey[upd] = g.pixel_key
        best_ncols[upd] = ncol
        matched_any[upd] = True

        for v in orac_vars:
            arr = np.asarray(ds[v].squeeze(drop=True).values)
            var_best[v][upd] = arr[at_i[improves], ac_i[improves]]
        ds.close()

    pixel_id = np.where(
        matched_any,
        best_pixkey * 1_800_000 + best_along * np.where(best_ncols > 0, best_ncols, 1) + best_across,
        -1,
    ).astype(np.int64)

    out = pd.DataFrame({
        "ec_lat": ec_lat,
        "ec_lon": ec_lon,
        "ec_time": ec_time,
        "on_disk": matched_any,
        "sev_along_track": best_along.astype(np.int32),
        "sev_across_track": best_across.astype(np.int32),
        "sev_lat": best_slat,
        "sev_lon": best_slon,
        "sev_pixel_id": pixel_id,
        "sev_scan_time": best_ptime,
        "distance_km": np.where(matched_any, best_dist, np.nan),
        "time_diff_s": best_tdiff,
    })
    for v in orac_vars:
        out[v] = var_best[v]
    out["valid_match"] = matched_any & (best_tdiff <= max_time_diff_seconds)
    return out


def _empty_result(ec_lat, ec_lon, ec_time, orac_vars) -> pd.DataFrame:
    n = len(ec_lat)
    out = pd.DataFrame({
        "ec_lat": ec_lat, "ec_lon": ec_lon, "ec_time": ec_time,
        "on_disk": np.zeros(n, dtype=bool),
        "sev_along_track": np.full(n, -1, dtype=np.int32),
        "sev_across_track": np.full(n, -1, dtype=np.int32),
        "sev_lat": np.full(n, np.nan), "sev_lon": np.full(n, np.nan),
        "sev_pixel_id": np.full(n, -1, dtype=np.int64),
        "sev_scan_time": np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]"),
        "distance_km": np.full(n, np.nan), "time_diff_s": np.full(n, np.nan),
    })
    for v in orac_vars:
        out[v] = np.nan
    out["valid_match"] = np.zeros(n, dtype=bool)
    return out
