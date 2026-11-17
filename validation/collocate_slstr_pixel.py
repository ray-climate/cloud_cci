"""SLSTR-pixel-based collocation: one row per ORAC SLSTR pixel under the EarthCARE track.

``collocate_slstr.py`` is EarthCARE-driven: one row per profile, each keeping a
single "best" SLSTR pixel. That double-counts SLSTR pixels crossed by more than
one profile (~10% of them) and, when two Sentinel-3 overpasses fall inside the
time window, keeps the spatially nearest rather than both. For validating the
*SLSTR* retrieval the unit has to be the SLSTR pixel, so this module:

1. finds **every** (profile, granule) pair inside the time window whose nearest
   pixel is within ``max_pixel_dist_km`` — the same unit-vector KD-tree search,
   but nothing is discarded as "second best";
2. groups the pairs by (granule, along_track, across_track) and reduces the
   EarthCARE values of the profiles inside each pixel to one row.

The time window is deliberately wide (default ±60 min); the signed time
difference is a column so the threshold is chosen at analysis time.
"""
from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from orac.io import julian_to_datetime
from orac.slstr import lla_to_unit_xyz, open_granule

from .collocate_slstr import _GRANULE_DUR_S, GranuleCatalog, _chord_to_km

# Pixels are on a ~1 km grid, so a profile inside the swath is < 0.71 km from its
# nearest pixel centre; allow for grid distortion, reject off-swath profiles.
DEFAULT_MAX_PIXEL_DIST_KM = 1.5

# ORAC primary-file variables sampled at each matched pixel.
ORAC_PIXEL_VARS = (
    "cot", "cot_uncertainty", "cer", "cer_uncertainty", "cwp", "cwp_uncertainty",
    "cth", "cth_uncertainty", "cth_corrected", "cth_corrected_uncertainty",
    "ctt", "ctt_uncertainty", "ctt_corrected", "ctp", "ctp_uncertainty", "ctp_corrected",
    "phase", "phase_pavolonis", "ann_phase", "ann_phase_uncertainty",
    "cldtype", "cldmask", "cldmask_uncertainty", "cccot_pre", "cc_total",
    "qcflag", "costja", "costjm", "niter",
    "illum", "lsflag", "lusflag", "dem", "stemp",
    "solar_zenith_view_no1", "satellite_zenith_view_no1", "rel_azimuth_view_no1",
)
# Variables that also get a 3x3-pixel mean / std (scene homogeneity).
ORAC_NEIGHBOURHOOD_VARS = ("cot", "cth")
_ORAC_INT_VARS = {
    "phase", "phase_pavolonis", "ann_phase", "cldtype", "cldmask", "qcflag",
    "niter", "illum", "lsflag", "lusflag",
}


def _neighbourhood_stats(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray):
    """Mean, std and valid count of the 3x3 box around each (row, col)."""
    nrow, ncol = arr.shape
    stack = np.full((9, rows.size), np.nan)
    k = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = rows + dr, cols + dc
            ok = (r >= 0) & (r < nrow) & (c >= 0) & (c < ncol)
            stack[k, ok] = arr[r[ok], c[ok]]
            k += 1
    count = np.isfinite(stack).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN boxes
        return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0), count


def find_pixel_pairs(
    ec_lat: np.ndarray,
    ec_lon: np.ndarray,
    ec_time: np.ndarray,
    catalog: GranuleCatalog,
    orac_vars: Iterable[str] = ORAC_PIXEL_VARS,
    max_time_diff_seconds: float = 3600.0,
    max_pixel_dist_km: float = DEFAULT_MAX_PIXEL_DIST_KM,
) -> pd.DataFrame:
    """All (EarthCARE profile, SLSTR pixel) pairs inside the time window.

    One row per pair: a profile seen by two overpasses appears twice, and two
    profiles in one pixel give two rows sharing ``slstr_granule`` /
    ``slstr_along_track`` / ``slstr_across_track``. ``dt_s`` is EarthCARE minus
    SLSTR pixel time. ORAC variables are sampled at the pixel as ``orac_<name>``.
    """
    ec_lat = np.asarray(ec_lat, dtype=np.float64)
    ec_lon = np.asarray(ec_lon, dtype=np.float64)
    ec_time = np.asarray(ec_time, dtype="datetime64[ns]")
    orac_vars = tuple(orac_vars)

    ec_xyz = lla_to_unit_xyz(ec_lat, ec_lon)
    ec_ok = np.isfinite(ec_xyz).all(axis=1) & ~np.isnat(ec_time)
    if not ec_ok.any() or not catalog.granules:
        return pd.DataFrame()
    ec_c = ec_xyz[ec_ok].mean(axis=0)
    ec_c = ec_c / np.linalg.norm(ec_c)
    ec_radius = float(np.arccos(np.clip((ec_xyz[ec_ok] @ ec_c).min(), -1.0, 1.0)))

    # A pixel can be up to one granule duration later than the granule start.
    reach = np.timedelta64(int(max_time_diff_seconds + _GRANULE_DUR_S), "s")
    t_lo, t_hi = ec_time[ec_ok].min(), ec_time[ec_ok].max()
    cand = np.flatnonzero(
        (catalog.start_times >= t_lo - reach) & (catalog.start_times <= t_hi + reach))

    parts: list[pd.DataFrame] = []
    for ci in cand:
        g = catalog.granules[ci]
        sep = float(np.arccos(np.clip(ec_c @ catalog.centroids[ci], -1.0, 1.0)))
        if sep > ec_radius + catalog.radii[ci] + np.deg2rad(1.0):
            continue
        elig = np.flatnonzero(ec_ok & (np.abs(ec_time - catalog.start_times[ci]) <= reach))
        if elig.size == 0:
            continue

        ds = open_granule(g, variables=("lat", "lon", "time", *orac_vars),
                          include_secondary=False)
        try:
            sl_lat = np.asarray(ds["lat"].values, dtype=np.float64)
            sl_lon = np.asarray(ds["lon"].values, dtype=np.float64)
            nrow, ncol = sl_lat.shape
            flat_idx = np.flatnonzero((np.isfinite(sl_lat) & np.isfinite(sl_lon)).ravel())
            if flat_idx.size == 0:
                continue
            tree = cKDTree(lla_to_unit_xyz(sl_lat.ravel()[flat_idx], sl_lon.ravel()[flat_idx]))
            chord, k = tree.query(ec_xyz[elig], k=1)
            dist_km = _chord_to_km(chord)
            at_i, ac_i = np.unravel_index(flat_idx[k], (nrow, ncol))
            p_time = julian_to_datetime(ds["time"].values)[at_i, ac_i]
            dt_s = (ec_time[elig] - p_time) / np.timedelta64(1, "s")
            keep = (dist_km <= max_pixel_dist_km) & (np.abs(dt_s) <= max_time_diff_seconds)
            if not keep.any():
                continue
            at_i, ac_i = at_i[keep], ac_i[keep]

            part = pd.DataFrame({
                "ec_index": elig[keep],
                "slstr_granule": g.primary.name,
                "slstr_orbit": g.orbit,
                "slstr_along_track": at_i.astype(np.int32),
                "slstr_across_track": ac_i.astype(np.int32),
                "slstr_time": p_time[keep],
                "slstr_lat": sl_lat[at_i, ac_i],
                "slstr_lon": sl_lon[at_i, ac_i],
                "distance_km": dist_km[keep],
                "dt_s": dt_s[keep],
            })
            for v in orac_vars:
                if v not in ds.variables:
                    part[f"orac_{v}"] = np.nan
                    continue
                arr = np.asarray(ds[v].squeeze(drop=True).values)
                part[f"orac_{v}"] = arr[at_i, ac_i]
                if v in ORAC_NEIGHBOURHOOD_VARS:
                    mean, std, cnt = _neighbourhood_stats(arr.astype(np.float64), at_i, ac_i)
                    part[f"orac_{v}_3x3_mean"] = mean
                    part[f"orac_{v}_3x3_std"] = std
                    part[f"orac_{v}_3x3_n"] = cnt
            parts.append(part)
        finally:
            ds.close()

    if not parts:
        return pd.DataFrame()
    pairs = pd.concat(parts, ignore_index=True)
    # Consecutive granules of one orbit abut: a profile just past the last row
    # of one is still within ``max_pixel_dist_km`` of its edge pixel as well as
    # inside the next. Keep the nearest pixel per (profile, orbit) — different
    # orbits are genuinely separate overpasses and all stay.
    pairs = pairs.sort_values("distance_km", kind="stable")
    pairs = pairs.drop_duplicates(["ec_index", "slstr_orbit"], keep="first")
    return pairs.sort_values(["slstr_time", "ec_index"], kind="stable").reset_index(drop=True)


def aggregate_pairs_to_pixels(
    pairs: pd.DataFrame,
    ec_values: pd.DataFrame,
    mean_cols: Sequence[str],
    range_cols: Sequence[str] = (),
    nearest_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Collapse profile-pixel pairs to one row per SLSTR pixel.

    ``ec_values`` is indexed by profile (row ``i`` ↔ ``ec_index == i``).

    * ``mean_cols``    — NaN-ignoring mean over the profiles inside the pixel
    * ``range_cols``   — additionally ``<col>_range`` = max − min (sub-pixel spread)
    * ``nearest_cols`` — value from the profile closest to the pixel centre
      (flags and identifiers, which must not be averaged)

    Pixel-side columns (keys, ORAC values) and ``distance_km`` / ``dt_s`` are
    taken from the nearest profile; ``n_profiles`` counts the profiles used.
    """
    if pairs.empty:
        return pairs
    keys = ["slstr_granule", "slstr_along_track", "slstr_across_track"]
    ec_cols = list(dict.fromkeys([*mean_cols, *nearest_cols]))
    df = pairs.join(ec_values[ec_cols], on="ec_index")
    df = df.sort_values([*keys, "distance_km"], kind="stable")
    grouped = df.groupby(keys, sort=False)

    # Nearest profile's row (not groupby.first(), which skips NaN per column).
    out = df.drop_duplicates(keys, keep="first").set_index(keys)
    out["n_profiles"] = grouped.size()
    out["ec_index_first"] = grouped["ec_index"].min()
    out["ec_index_last"] = grouped["ec_index"].max()
    means = grouped[list(mean_cols)].mean()
    for c in mean_cols:
        out[c] = means[c]
    for c in range_cols:
        out[f"{c}_range"] = grouped[c].max() - grouped[c].min()
    return out.reset_index().rename(columns={"ec_index": "ec_index_nearest"})


def format_pixel_table(df: pd.DataFrame) -> pd.DataFrame:
    """Cast ORAC integer flags to nullable ints so fills are empty in the CSV."""
    for v in _ORAC_INT_VARS:
        c = f"orac_{v}"
        if c in df.columns:
            df[c] = df[c].round().astype("Int32")
    return df
