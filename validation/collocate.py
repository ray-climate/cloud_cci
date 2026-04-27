"""Sample-level collocation: ATLID profile track → SEVIRI pixel.

Emits one row per ATLID profile. Each row carries the index of its nearest
SEVIRI pixel in the slot whose scan_time is closest in time, the haversine
distance to that pixel, and the time difference. Off-disk profiles get
on_disk=False and NaN match info — they are kept in the table so the count
matches the input track length.

Aggregation by SEVIRI pixel and stratification by distance/time happen in
``statistics.py``; this module is purely the matcher.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from orac.io import open_slot
from orac.metadata import Retrieval, SlotRecord, discover_slots


def _on_disk(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Crude SEVIRI footprint mask. The fine cut is whether the nearest pixel
    has finite lat/lon — done implicitly via the kd-tree on valid pixels.
    """
    return (np.abs(lat) <= 81.5) & (np.abs(lon) <= 81.5) & np.isfinite(lat) & np.isfinite(lon)


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = phi2 - phi1
    dlam = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _query_slot(
    slot: SlotRecord,
    retrieval: Retrieval,
    atlid_lat: np.ndarray,
    atlid_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bulk nearest-pixel query against one SEVIRI slot.

    Returns (along_track_idx, across_track_idx, sev_lat, sev_lon, distance_km),
    each of length ``len(atlid_lat)``.
    """
    ds = open_slot(slot, retrieval, variables=("lat", "lon"), include_secondary=False)
    sl_lat = np.asarray(ds["lat"].values, dtype=np.float64)
    sl_lon = np.asarray(ds["lon"].values, dtype=np.float64)
    nrow_at, nrow_ac = sl_lat.shape

    valid = np.isfinite(sl_lat) & np.isfinite(sl_lon)
    flat_idx = np.flatnonzero(valid.ravel())
    if flat_idx.size == 0:
        # Slot has no valid pixels (corrupt or all-fill). Return sentinel arrays
        # so the caller can mark these matches as off-disk / unmatched.
        n = atlid_lat.shape[0]
        return (
            np.full(n, -1, dtype=np.int64),
            np.full(n, -1, dtype=np.int64),
            np.full(n, np.nan),
            np.full(n, np.nan),
            np.full(n, np.nan),
        )
    lat_flat = sl_lat.ravel()[flat_idx]
    lon_flat = sl_lon.ravel()[flat_idx]

    mean_lat = float(np.nanmean(atlid_lat))
    scale = float(np.cos(np.deg2rad(mean_lat)))
    tree = cKDTree(np.column_stack([lat_flat, lon_flat * scale]))

    pts = np.column_stack([atlid_lat, atlid_lon * scale])
    _, k_idx = tree.query(pts, k=1)
    pixel_flat = flat_idx[k_idx]
    at_i, ac_i = np.unravel_index(pixel_flat, (nrow_at, nrow_ac))

    sev_lat = sl_lat[at_i, ac_i]
    sev_lon = sl_lon[at_i, ac_i]
    dist = _haversine_km(atlid_lat, atlid_lon, sev_lat, sev_lon)
    return at_i, ac_i, sev_lat, sev_lon, dist


def match_track_to_seviri(
    atlid_lat: np.ndarray,
    atlid_lon: np.ndarray,
    atlid_time: np.ndarray,
    seviri_root: str | Path,
    retrieval: Retrieval = "R11",
    max_time_diff_seconds: float = 450.0,
) -> pd.DataFrame:
    """Match each ATLID profile to its nearest SEVIRI pixel.

    Parameters
    ----------
    atlid_lat, atlid_lon
        ``(n_profile,)`` float64 arrays of geographic coordinates.
    atlid_time
        ``(n_profile,)`` ``datetime64[ns]`` UTC.
    seviri_root
        Root of the ORAC SEVIRI L2 archive (`<root>/YYYY/MM/DD/HHMM/...`).
    retrieval
        ORAC retrieval label, "R10" or "R11".
    max_time_diff_seconds
        ATLID profiles whose nearest SEVIRI slot is further than this in
        time get ``valid_match=False`` (default 7.5 min = half a SEVIRI
        cadence, matches CLAAS-3 / Holz 2008).

    Returns
    -------
    pandas.DataFrame
        One row per ATLID profile. Columns:

        ec_lat, ec_lon, ec_time
            ATLID profile geometry.
        on_disk
            Sample falls inside the SEVIRI footprint.
        sev_along_track, sev_across_track
            Pixel indices on the 3712 × 3712 grid (–1 if no match).
        sev_lat, sev_lon
            Centre coordinates of the matched pixel.
        sev_pixel_id
            Flat (along × n_across + across) global pixel id — the groupby key
            for downstream pixel-aggregate statistics.
        sev_scan_time
            UTC scan_time of the matched slot.
        distance_km
            Great-circle distance ATLID → SEVIRI pixel centre.
        time_diff_s
            |Δt| between ATLID profile and SEVIRI scan_time.
        valid_match
            on_disk AND time_diff_s ≤ max_time_diff_seconds.
    """
    n = len(atlid_lat)
    on_disk = _on_disk(atlid_lat, atlid_lon)

    # Slot enumeration covering the frame ± half a cadence. Floor to
    # microseconds before .to_pydatetime() — Python datetime is µs precision
    # while ATLID time is ns, otherwise pandas warns on the conversion.
    margin = timedelta(seconds=max_time_diff_seconds + 60)
    t_min = pd.Timestamp(atlid_time.min()).floor("us").tz_localize(timezone.utc)
    t_max = pd.Timestamp(atlid_time.max()).floor("us").tz_localize(timezone.utc)
    slots: list[SlotRecord] = discover_slots(
        seviri_root,
        (t_min - margin).to_pydatetime(),
        (t_max + margin).to_pydatetime(),
        retrievals=(retrieval,),
    )
    if not slots:
        raise RuntimeError(
            f"No SEVIRI {retrieval} slots in {seviri_root} for {t_min}..{t_max}"
        )
    slot_times = np.array(
        [np.datetime64(s.scan_time.replace(tzinfo=None), "ns") for s in slots],
        dtype="datetime64[ns]",
    )

    # Per-profile nearest slot in time (vectorised over both axes; for tens of
    # slots × thousands of profiles this is ~MB-scale and trivial).
    diffs = np.abs(atlid_time[:, None] - slot_times[None, :])
    nearest_slot_idx = diffs.argmin(axis=1)
    time_diff_s = (
        diffs[np.arange(n), nearest_slot_idx] / np.timedelta64(1, "s")
    ).astype(np.float64)

    out = pd.DataFrame(
        {
            "ec_lat": atlid_lat,
            "ec_lon": atlid_lon,
            "ec_time": atlid_time,
            "on_disk": on_disk,
            "sev_along_track": np.full(n, -1, dtype=np.int32),
            "sev_across_track": np.full(n, -1, dtype=np.int32),
            "sev_lat": np.full(n, np.nan),
            "sev_lon": np.full(n, np.nan),
            "sev_pixel_id": np.full(n, -1, dtype=np.int64),
            "sev_scan_time": np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]"),
            "distance_km": np.full(n, np.nan),
            "time_diff_s": time_diff_s,
        }
    )

    # For each used slot: bulk kd-tree query on the on-disk subset assigned to it.
    used_slots = np.unique(nearest_slot_idx[on_disk])
    n_across = None
    for si in used_slots:
        mask = on_disk & (nearest_slot_idx == si)
        if not mask.any():
            continue
        slot = slots[si]
        at_i, ac_i, sev_lat, sev_lon, dist = _query_slot(
            slot, retrieval, atlid_lat[mask], atlid_lon[mask]
        )
        if n_across is None:
            ds = open_slot(slot, retrieval, variables=("lat",), include_secondary=False)
            n_across = ds["lat"].shape[1]
        out.loc[mask, "sev_along_track"] = at_i.astype(np.int32)
        out.loc[mask, "sev_across_track"] = ac_i.astype(np.int32)
        out.loc[mask, "sev_lat"] = sev_lat
        out.loc[mask, "sev_lon"] = sev_lon
        out.loc[mask, "sev_pixel_id"] = (at_i.astype(np.int64) * n_across + ac_i).astype(np.int64)
        out.loc[mask, "sev_scan_time"] = np.datetime64(
            slot.scan_time.replace(tzinfo=None), "ns"
        )
        out.loc[mask, "distance_km"] = dist

    out["valid_match"] = on_disk & (time_diff_s <= max_time_diff_seconds)
    return out


def open_seviri_at_matches(
    matches: pd.DataFrame,
    seviri_root: str | Path,
    retrieval: Retrieval,
    variables: Iterable[str],
) -> pd.DataFrame:
    """Augment a matches DataFrame with ORAC SEVIRI variables sampled at each
    matched (along_track, across_track) pixel.

    Reads each scan_time slot once. Skips rows with ``valid_match=False``.
    Returns a copy of ``matches`` with one new column per requested variable.
    """
    df = matches.copy()
    for v in variables:
        df[v] = np.nan

    valid = df["valid_match"].values
    if not valid.any():
        return df

    for st, group in df[valid].groupby("sev_scan_time"):
        # Find the slot record matching this scan_time.
        slot_dt = pd.Timestamp(st).to_pydatetime().replace(tzinfo=timezone.utc)
        slots = discover_slots(
            seviri_root,
            slot_dt - timedelta(minutes=1),
            slot_dt + timedelta(minutes=1),
            retrievals=(retrieval,),
        )
        if not slots:
            continue
        slot = slots[0]
        ds = open_slot(slot, retrieval, variables=tuple(variables))
        at = group["sev_along_track"].astype(int).values
        ac = group["sev_across_track"].astype(int).values
        for v in variables:
            if v not in ds.variables:
                continue
            # Some ORAC vars (e.g. cldmask) carry a singleton ``views`` dim; drop
            # any size-1 leading dims so the trailing axes are (along_track, across_track).
            da = ds[v].squeeze(drop=True)
            arr = np.asarray(da.values)
            df.loc[group.index, v] = arr[at, ac]
    return df
