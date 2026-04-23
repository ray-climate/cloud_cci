"""Per-slot and monthly summary statistics for ORAC SEVIRI retrievals."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from .flags import qc_pass_mask
from .io import open_slot
from .metadata import Retrieval, SlotRecord, discover_slots

# Variables loaded for the default summary. Kept small so walks run in
# reasonable memory on a laptop.
DEFAULT_SUMMARY_VARS: tuple[str, ...] = (
    "cot", "cer", "ctp", "cth", "cwp", "cc_total",
    "niter", "costja", "costjm",
    "qcflag", "cldmask", "phase",
    "degrees_of_freedom_signal",
)


def per_slot_stats(
    ds: xr.Dataset,
    variables: Iterable[str] = ("cot", "cer", "ctp", "cth", "cwp"),
    qc_rules: str | None = "default",
) -> dict[str, float | int]:
    """Compute a flat dict of summary stats for a single ``(slot, retrieval)``.

    If ``qc_rules`` is set, statistics of the listed variables are computed on
    QC-passed pixels only. Pixel counts and cloudy fraction always use the full
    (on-disk) field.
    """
    # On-disk mask — off-disk pixels have NaN lat/lon after mask_and_scale.
    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    on_disk = np.isfinite(lat) & np.isfinite(lon)
    n_ondisk = int(on_disk.sum())

    stats: dict[str, float | int] = {"pixels_on_disk": n_ondisk}

    if "cldmask" in ds.variables:
        cm = np.asarray(ds["cldmask"].squeeze(drop=True).values)
        cloudy = on_disk & (cm == 1)
        n_cloudy = int(cloudy.sum())
        stats["pixels_cloudy"] = n_cloudy
        stats["cloudy_fraction"] = (n_cloudy / n_ondisk) if n_ondisk else np.nan

    if qc_rules is not None and "qcflag" in ds.variables:
        qc_arr = np.asarray(qc_pass_mask(ds["qcflag"], rules=qc_rules).values)
        qc_mask = qc_arr & on_disk
        stats["pixels_qc_pass"] = int(qc_mask.sum())
        stats["qc_pass_rate"] = (stats["pixels_qc_pass"] / n_ondisk) if n_ondisk else np.nan
    else:
        qc_mask = on_disk

    for var in variables:
        if var not in ds.variables:
            continue
        arr = np.asarray(ds[var].squeeze(drop=True).values)
        vals = arr[qc_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            stats[f"{var}_mean"] = float(np.mean(vals))
            stats[f"{var}_median"] = float(np.median(vals))
        else:
            stats[f"{var}_mean"] = np.nan
            stats[f"{var}_median"] = np.nan

    # Retrieval diagnostics on all converged pixels (no extra QC filtering).
    for var in ("niter", "costja", "costjm", "degrees_of_freedom_signal"):
        if var in ds.variables:
            arr = np.asarray(ds[var].squeeze(drop=True).values)
            vals = arr[on_disk]
            vals = vals[np.isfinite(vals)]
            stats[f"{var}_median"] = float(np.median(vals)) if vals.size else np.nan

    return stats


def monthly_summary(
    root: str | Path,
    year: int,
    month: int,
    retrievals: Iterable[Retrieval] = ("R10", "R11"),
    variables: Iterable[str] = DEFAULT_SUMMARY_VARS,
    qc_rules: str | None = "default",
    progress: bool = True,
) -> pd.DataFrame:
    """Walk a month of data and return a tidy stats table (one row per slot/retrieval)."""
    from calendar import monthrange

    start = datetime(year, month, 1)
    end_day = monthrange(year, month)[1]
    end = datetime(year, month + 1, 1) if month < 12 else datetime(year + 1, 1, 1)
    slots = discover_slots(root, start=start, end=end, retrievals=retrievals)

    stat_vars = [v for v in variables if v not in {"qcflag", "cldmask", "phase", "niter",
                                                    "costja", "costjm",
                                                    "degrees_of_freedom_signal"}]
    load_vars = list(set(variables) | {"lat", "lon"})

    rows: list[dict] = []
    it = enumerate(slots)
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            it = enumerate(tqdm(slots, desc=f"{year}-{month:02d}"))
        except ImportError:
            pass

    for _, slot in it:
        for r in retrievals:
            if not slot.has(r):
                continue
            try:
                ds = open_slot(slot, r, variables=load_vars)
                s = per_slot_stats(ds, variables=stat_vars, qc_rules=qc_rules)
                ds.close()
            except Exception as exc:  # pragma: no cover - logged, not raised
                s = {"error": str(exc)}
            s.update({"scan_time": slot.scan_time, "retrieval": r,
                      "slot_folder": slot.slot_folder})
            rows.append(s)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["scan_time", "retrieval"]).reset_index(drop=True)
    return df


def missing_slot_report(
    root: str | Path,
    year: int,
    month: int,
    retrievals: Iterable[Retrieval] = ("R10", "R11"),
) -> pd.DataFrame:
    """Return one row per expected 15-min slot with which files are present."""
    from calendar import monthrange

    days = monthrange(year, month)[1]
    all_slots = pd.date_range(datetime(year, month, 1), periods=96 * days, freq="15min", tz="UTC")
    expected = pd.DataFrame({"expected_slot": all_slots})
    expected["expected_scan_time"] = expected["expected_slot"] + pd.Timedelta(minutes=12)

    start = datetime(year, month, 1)
    end = datetime(year, month + 1, 1) if month < 12 else datetime(year + 1, 1, 1)
    slots = discover_slots(root, start=start, end=end, retrievals=retrievals)

    found = pd.DataFrame({
        "scan_time": [s.scan_time for s in slots],
        "R10_primary":   [s.get("R10", "primary")   is not None for s in slots],
        "R10_secondary": [s.get("R10", "secondary") is not None for s in slots],
        "R11_primary":   [s.get("R11", "primary")   is not None for s in slots],
        "R11_secondary": [s.get("R11", "secondary") is not None for s in slots],
    })
    found["scan_time"] = pd.to_datetime(found["scan_time"], utc=True)

    merged = expected.merge(
        found, how="left",
        left_on="expected_scan_time", right_on="scan_time",
    )
    for col in ("R10_primary", "R10_secondary", "R11_primary", "R11_secondary"):
        merged[col] = merged[col].fillna(False)
    return merged
