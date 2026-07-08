"""Command-line interface for the validation module.

Subcommands:

- ``collocate``  : iterate A-EBD frames in a date range, match each to
  SEVIRI ORAC, and write per-frame matches CSVs.
- ``evaluate``   : concatenate per-frame CSVs and write a monthly
  stratified-stats table.
- ``figures``    : produce scatter, diagnostic, and bias-by-stratum
  PNGs from the concatenated CSV.

Designed to be parallel-safe at frame granularity — `collocate` skips
frames whose output CSV already exists, so a re-run resumes cleanly.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

import pandas as pd

from .collocate import match_track_to_seviri, open_seviri_at_matches
from .collocate_slstr import (
    GranuleCatalog, build_granule_catalog, match_track_to_slstr,
)
from .figures import bias_by_stratum, diagnostic_panel, scatter_panel
from .readers import read_aebd_track, read_accap_track, read_acth_track
from .reference import cot_cer_water_from_accap, cot_from_aebd, cth_from_acth
from .compare_figures import (
    bias_bar_compare, scatter_compare, scatter_compare_by_surface,
)
from . import cth_figures, water_cloud_figures
from .statistics import (
    CTH_QC_MODES, SYNERGY_QC_MODES, aggregate_to_pixel, aggregate_to_pixel_cth,
    aggregate_to_pixel_water, cer_water_report, cer_water_strata,
    cot_report, cot_water_report, cot_water_strata, cth_report, cth_strata,
    dedupe_to_sample, dedupe_to_sample_water, filter_water_sampling,
    homogeneity_sweep_stats, stratified_stats,
)
from .track_figures import track_panel

DEFAULT_DRIVER_DIR = {
    "A-EBD": "ATL_EBD_2A",
    "A-CTH": "ATL_CTH_2A",
    "ACM-CAP": "ACM_CAP_2B",
}

ORAC_COT_SATURATION = 100.0
DEFAULT_RETRIEVAL = "R11"
EARTHCARE_ROOT = Path("earthcare_data")
SEVIRI_ROOT_DEFAULT = Path("/gws/ssde/j25a/cloud_ecv/data_out/seviri")
SLSTR_ROOT_DEFAULT = Path(
    "/gws/ssde/j25a/cloud_ecv/data_out/slstr/v5.1_new_snowice/slstra/l2b"
)
SLSTR_DEFAULT_MAX_DT_MIN = 30.0
SLSTR_CATALOG_CACHE = Path("validation_data/_slstr_catalog")
FRAME_RE = re.compile(r"_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)_(\w+)\.h5$")


def _frame_metadata(path: Path) -> tuple[str, datetime] | None:
    """Return (frame_id, start_time) or None if filename doesn't parse."""
    m = FRAME_RE.search(path.name)
    if not m:
        return None
    start = datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return m.group(3), start


def _enumerate_frames(driver: str, start: datetime, end: datetime) -> list[Path]:
    """Find all driver frames in EARTHCARE_ROOT within [start, end)."""
    sub = DEFAULT_DRIVER_DIR.get(driver, driver.replace("-", "_") + "_2A")
    root = EARTHCARE_ROOT / sub
    if not root.exists():
        return []
    matches: list[tuple[datetime, Path]] = []
    for p in root.rglob("*.h5"):
        meta = _frame_metadata(p)
        if meta is None:
            continue
        _, t = meta
        if start <= t < end:
            matches.append((t, p))
    return [p for _, p in sorted(matches)]


# ---------------------------------------------------------------------------
# collocate
# ---------------------------------------------------------------------------

def _process_frame_cot(
    path: Path, seviri_root: Path, retrieval: str, out_dir: Path
) -> tuple[Path | None, str]:
    """Match one A-EBD frame and write matches_<frame_id>.csv.

    Returns ``(out_path, status)``. Status is one of: ``done`` (wrote new),
    ``skip`` (CSV existed), ``empty`` (no on-disk profiles), ``fail``.
    """
    frame_id = _frame_metadata(path)
    if frame_id is None:
        return None, "fail"
    fid = frame_id[0]
    out_csv = out_dir / f"matches_cot_{fid}.csv"
    if out_csv.exists() and out_csv.stat().st_size > 0:
        return out_csv, "skip"

    try:
        track = read_aebd_track(path)
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] read failed: {e}", file=sys.stderr)
        return None, "fail"

    cot, attenuated = cot_from_aebd(
        track["extinction"], track["height"], track["quality_status"]
    )

    try:
        matches = match_track_to_seviri(
            track["lat"], track["lon"], track["time"],
            seviri_root, retrieval=retrieval,
        )
    except RuntimeError as e:
        # No SEVIRI slots in window — common for off-disk-only frames.
        print(f"  [{fid}] no SEVIRI slots: {e}", file=sys.stderr)
        return None, "empty"
    except Exception as e:  # noqa: BLE001 — never let one bad frame kill the run
        print(f"  [{fid}] match failed ({type(e).__name__}): {e}", file=sys.stderr)
        return None, "fail"

    matches["cot_atlid"] = cot
    matches["attenuated"] = attenuated
    matches["frame_id"] = fid

    if not matches["valid_match"].any():
        # Frame fell entirely outside the SEVIRI footprint or time window.
        return None, "empty"

    try:
        matches = open_seviri_at_matches(
            matches, seviri_root, retrieval, ("cot", "cldmask", "lsflag", "phase")
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] ORAC sample failed ({type(e).__name__}): {e}", file=sys.stderr)
        return None, "fail"
    matches = matches.rename(columns={"cot": "cot_orac", "cldmask": "cldmask_orac",
                                       "lsflag": "lsflag_orac", "phase": "phase_orac"})
    matches["cot_orac_saturated"] = matches["cot_orac"] >= ORAC_COT_SATURATION

    out_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out_csv, index=False)
    return out_csv, "done"


def cmd_collocate(args: argparse.Namespace) -> int:
    start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    frames = _enumerate_frames(args.driver, start, end)
    print(f"Found {len(frames)} {args.driver} frames in [{start}, {end})")
    if not frames:
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "skip": 0, "empty": 0, "fail": 0}
    for i, path in enumerate(frames, 1):
        meta = _frame_metadata(path)
        fid = meta[0] if meta else "?"
        out_path, status = _process_frame_cot(
            path, Path(args.seviri_root), args.retrieval, out_dir
        )
        counts[status] += 1
        marker = {"done": "✓", "skip": "·", "empty": "○", "fail": "✗"}[status]
        print(f"  [{i:4d}/{len(frames)}] {marker} {fid:>8} → {status}")
    print(f"Summary: {counts}")
    return 0


# ---------------------------------------------------------------------------
# cth-collocate (A-CTH driver)
# ---------------------------------------------------------------------------

# ORAC vars sampled at each matched SEVIRI pixel for the CTH validation.
# Keep both raw and parallax-corrected so the corrected-vs-raw diagnostic is
# possible without re-running collocation.
_CTH_ORAC_VARS = (
    "cth", "cth_corrected", "cth_uncertainty", "cth_corrected_uncertainty",
    "cldmask", "lsflag", "phase",
)


def _process_frame_cth(
    path: Path, seviri_root: Path, retrieval: str, out_dir: Path
) -> tuple[Path | None, str]:
    """Match one A-CTH frame and write matches_cth_<frame_id>.csv.

    No QC is applied at this stage — the CSV carries the raw ATLID
    ``quality_status``, ``confidence``, ``cloud_class``, and per-profile
    ``tropopause_km`` so QC choices are made as strata at evaluate time.

    Returns ``(out_path, status)``. Status: ``done`` / ``skip`` / ``empty``
    / ``fail``, matching the cot path.
    """
    frame_id = _frame_metadata(path)
    if frame_id is None:
        return None, "fail"
    fid = frame_id[0]
    out_csv = out_dir / f"matches_cth_{fid}.csv"
    if out_csv.exists() and out_csv.stat().st_size > 0:
        return out_csv, "skip"

    try:
        track = read_acth_track(path)
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] read failed: {e}", file=sys.stderr)
        return None, "fail"

    cth_thick_km, cth_raw_km = cth_from_acth(track["cth_thick"], track["cth_raw"])

    try:
        matches = match_track_to_seviri(
            track["lat"], track["lon"], track["time"],
            seviri_root, retrieval=retrieval,
        )
    except RuntimeError as e:
        print(f"  [{fid}] no SEVIRI slots: {e}", file=sys.stderr)
        return None, "empty"
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] match failed ({type(e).__name__}): {e}", file=sys.stderr)
        return None, "fail"

    matches["cth_atlid_thick_km"] = cth_thick_km
    matches["cth_atlid_raw_km"] = cth_raw_km
    matches["quality_status_atlid"] = track["quality_status"]
    matches["confidence_atlid"] = track["cth_confidence"]
    matches["cloud_class_atlid"] = track["cloud_class"]
    matches["tropopause_km_atlid"] = track["tropopause_height_wmo"] / 1000.0
    matches["frame_id"] = fid

    if not matches["valid_match"].any():
        return None, "empty"

    try:
        matches = open_seviri_at_matches(
            matches, seviri_root, retrieval, _CTH_ORAC_VARS
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] ORAC sample failed ({type(e).__name__}): {e}", file=sys.stderr)
        return None, "fail"
    matches = matches.rename(columns={
        "cth": "cth_orac_km",
        "cth_corrected": "cth_orac_corrected_km",
        "cth_uncertainty": "cth_orac_uncertainty_km",
        "cth_corrected_uncertainty": "cth_orac_corrected_uncertainty_km",
        "cldmask": "cldmask_orac",
        "lsflag": "lsflag_orac",
        "phase": "phase_orac",
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out_csv, index=False)
    return out_csv, "done"


def cmd_cth_collocate(args: argparse.Namespace) -> int:
    start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    frames = _enumerate_frames("A-CTH", start, end)
    print(f"Found {len(frames)} A-CTH frames in [{start}, {end})")
    if not frames:
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "skip": 0, "empty": 0, "fail": 0}
    for i, path in enumerate(frames, 1):
        meta = _frame_metadata(path)
        fid = meta[0] if meta else "?"
        out_path, status = _process_frame_cth(
            path, Path(args.seviri_root), args.retrieval, out_dir
        )
        counts[status] += 1
        marker = {"done": "✓", "skip": "·", "empty": "○", "fail": "✗"}[status]
        print(f"  [{i:4d}/{len(frames)}] {marker} {fid:>8} → {status}")
    print(f"Summary: {counts}")
    return 0


# ---------------------------------------------------------------------------
# synergy-collocate (ACM-CAP driver — produces both COT and CER references)
# ---------------------------------------------------------------------------

# ORAC vars sampled at each matched SEVIRI pixel for the synergy validation.
# cot+cer+phase+cldmask+lsflag are needed for both cot-water and cer-water reports.
_SYNERGY_ORAC_VARS = ("cot", "cer", "cldmask", "lsflag", "phase")


def _process_frame_synergy(
    path: Path, seviri_root: Path, retrieval: str, out_dir: Path
) -> tuple[Path | None, str]:
    """Match one ACM-CAP frame and write matches_synergy_<frame_id>.csv.

    No QC is applied at this stage — the CSV carries the raw ACM-CAP
    quality fields (quality_status, convergence_status, synergy_status,
    cost_function, atlid_assim_status, cpr_assim_status) so QC choices
    are made as strata at evaluate time. Both cot_water and cer_water
    reference values are computed from the same per-bin arrays.
    """
    frame_id = _frame_metadata(path)
    if frame_id is None:
        return None, "fail"
    fid = frame_id[0]
    out_csv = out_dir / f"matches_synergy_{fid}.csv"
    if out_csv.exists() and out_csv.stat().st_size > 0:
        return out_csv, "skip"

    try:
        track = read_accap_track(path)
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] read failed: {e}", file=sys.stderr)
        return None, "fail"

    cot_w, cer_w, liq_present, ice_present = cot_cer_water_from_accap(
        track["liquid_optical_depth"], track["liquid_extinction"],
        track["liquid_eff_radius"], track["liquid_classification"],
        track["ice_water_content"], track["height"],
    )

    try:
        matches = match_track_to_seviri(
            track["lat"], track["lon"], track["time"],
            seviri_root, retrieval=retrieval,
        )
    except RuntimeError as e:
        print(f"  [{fid}] no SEVIRI slots: {e}", file=sys.stderr)
        return None, "empty"
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] match failed ({type(e).__name__}): {e}", file=sys.stderr)
        return None, "fail"

    matches["cot_water_atlid"] = cot_w
    matches["cer_water_atlid"] = cer_w
    matches["liquid_present_atlid"] = liq_present
    matches["ice_present_atlid"] = ice_present
    # Convenience derived flag retained for legacy strata; equivalent to
    # liquid_present & ~ice_present.
    matches["liquid_only_atlid"] = liq_present & ~ice_present
    matches["quality_status_atlid"] = track["quality_status"]
    matches["convergence_status_atlid"] = track["convergence_status"]
    matches["synergy_status_atlid"] = track["synergy_status"]
    matches["cost_function_atlid"] = track["cost_function"]
    matches["atlid_assim_status"] = track["atlid_assim_status"]
    matches["cpr_assim_status"] = track["cpr_assim_status"]
    matches["frame_id"] = fid

    if not matches["valid_match"].any():
        return None, "empty"

    try:
        matches = open_seviri_at_matches(
            matches, seviri_root, retrieval, _SYNERGY_ORAC_VARS
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] ORAC sample failed ({type(e).__name__}): {e}", file=sys.stderr)
        return None, "fail"
    matches = matches.rename(columns={
        "cot": "cot_orac", "cer": "cer_orac",
        "cldmask": "cldmask_orac", "lsflag": "lsflag_orac",
        "phase": "phase_orac",
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out_csv, index=False)
    return out_csv, "done"


def cmd_synergy_collocate(args: argparse.Namespace) -> int:
    start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    frames = _enumerate_frames("ACM-CAP", start, end)
    print(f"Found {len(frames)} ACM-CAP frames in [{start}, {end})")
    if not frames:
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "skip": 0, "empty": 0, "fail": 0}
    for i, path in enumerate(frames, 1):
        meta = _frame_metadata(path)
        fid = meta[0] if meta else "?"
        out_path, status = _process_frame_synergy(
            path, Path(args.seviri_root), args.retrieval, out_dir
        )
        counts[status] += 1
        marker = {"done": "✓", "skip": "·", "empty": "○", "fail": "✗"}[status]
        print(f"  [{i:4d}/{len(frames)}] {marker} {fid:>8} → {status}")
    print(f"Summary: {counts}")
    return 0


# ---------------------------------------------------------------------------
# SLSTR collocation (polar-orbiter swath — EarthCARE driver → SLSTR granule)
#
# Mirrors the SEVIRI cth / synergy / cot paths but uses ``match_track_to_slstr``
# (temporal-gated, crossing-limited). The output CSVs use the SAME column names
# as the SEVIRI matches, so cth-evaluate / cot-water-* / figures consume them
# unchanged. Two extra columns travel through: ``illum_orac`` (1 day / 2 twilight
# / 3 night) and ``sza_orac`` (solar zenith) — COT must be day-filtered because
# ORAC's solar retrieval defaults to a constant prior at night.
# ---------------------------------------------------------------------------

_SLSTR_CTH_VARS = ("cth", "cth_corrected", "cth_uncertainty",
                   "cth_corrected_uncertainty", "cldmask", "lsflag", "phase",
                   "illum", "solar_zenith_view_no1")
_SLSTR_SYNERGY_VARS = ("cot", "cer", "cldmask", "lsflag", "phase",
                       "illum", "solar_zenith_view_no1")
_SLSTR_COT_VARS = ("cot", "cldmask", "lsflag", "phase",
                   "illum", "solar_zenith_view_no1")
_SLSTR_ORAC_RENAME = {
    "cldmask": "cldmask_orac", "lsflag": "lsflag_orac", "phase": "phase_orac",
    "illum": "illum_orac", "solar_zenith_view_no1": "sza_orac",
}


def _load_or_build_slstr_catalog(
    slstr_root: Path, start: datetime, end: datetime, max_dt_s: float
) -> GranuleCatalog:
    """Build (and disk-cache) the corner-sampled granule catalogue for a run.

    The catalogue is expensive (a strided lat/lon read per granule) but shared
    across every frame and every product, so it is pickled under
    ``SLSTR_CATALOG_CACHE`` keyed by (root, window).
    """
    import pickle
    from datetime import timedelta

    from orac.slstr import discover_granules

    pad = timedelta(seconds=max_dt_s + 300)
    win_lo, win_hi = start - pad, end + pad
    key = f"{abs(hash((str(slstr_root), win_lo.isoformat(), win_hi.isoformat()))):x}"
    cache = SLSTR_CATALOG_CACHE.with_name(SLSTR_CATALOG_CACHE.name + f"_{key}.pkl")
    if cache.exists():
        with open(cache, "rb") as fh:
            cat = pickle.load(fh)
        print(f"Loaded granule catalogue ({len(cat.granules)} granules) from {cache}")
        return cat

    granules = discover_granules(slstr_root, win_lo, win_hi)
    print(f"Building granule catalogue over {len(granules)} SLSTR granules "
          f"in [{win_lo:%Y-%m-%d %H:%M}, {win_hi:%Y-%m-%d %H:%M}] ...")
    cat = build_granule_catalog(granules)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(cat, fh)
    print(f"  → cached to {cache} ({len(cat.granules)} usable granules)")
    return cat


def _process_frame_cth_slstr(
    path: Path, catalog: GranuleCatalog, slstr_root: Path, max_dt_s: float, out_dir: Path
) -> tuple[Path | None, str]:
    """Match one A-CTH frame to SLSTR ORAC; write matches_cth_<frame_id>.csv."""
    meta = _frame_metadata(path)
    if meta is None:
        return None, "fail"
    fid = meta[0]
    out_csv = out_dir / f"matches_cth_{fid}.csv"
    if out_csv.exists() and out_csv.stat().st_size > 0:
        return out_csv, "skip"
    try:
        track = read_acth_track(path)
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] read failed: {e}", file=sys.stderr)
        return None, "fail"

    cth_thick_km, cth_raw_km = cth_from_acth(track["cth_thick"], track["cth_raw"])
    try:
        matches = match_track_to_slstr(
            track["lat"], track["lon"], track["time"], slstr_root,
            orac_vars=_SLSTR_CTH_VARS, max_time_diff_seconds=max_dt_s,
            catalog=catalog,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] match failed ({type(e).__name__}): {e}", file=sys.stderr)
        return None, "fail"

    matches["cth_atlid_thick_km"] = cth_thick_km
    matches["cth_atlid_raw_km"] = cth_raw_km
    matches["quality_status_atlid"] = track["quality_status"]
    matches["confidence_atlid"] = track["cth_confidence"]
    matches["cloud_class_atlid"] = track["cloud_class"]
    matches["tropopause_km_atlid"] = track["tropopause_height_wmo"] / 1000.0
    matches["frame_id"] = fid
    if not matches["valid_match"].any():
        return None, "empty"

    matches = matches.rename(columns={
        "cth": "cth_orac_km", "cth_corrected": "cth_orac_corrected_km",
        "cth_uncertainty": "cth_orac_uncertainty_km",
        "cth_corrected_uncertainty": "cth_orac_corrected_uncertainty_km",
        **_SLSTR_ORAC_RENAME,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out_csv, index=False)
    return out_csv, "done"


def _process_frame_synergy_slstr(
    path: Path, catalog: GranuleCatalog, slstr_root: Path, max_dt_s: float, out_dir: Path
) -> tuple[Path | None, str]:
    """Match one ACM-CAP frame to SLSTR ORAC; write matches_synergy_<frame_id>.csv."""
    meta = _frame_metadata(path)
    if meta is None:
        return None, "fail"
    fid = meta[0]
    out_csv = out_dir / f"matches_synergy_{fid}.csv"
    if out_csv.exists() and out_csv.stat().st_size > 0:
        return out_csv, "skip"
    try:
        track = read_accap_track(path)
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] read failed: {e}", file=sys.stderr)
        return None, "fail"

    cot_w, cer_w, liq_present, ice_present = cot_cer_water_from_accap(
        track["liquid_optical_depth"], track["liquid_extinction"],
        track["liquid_eff_radius"], track["liquid_classification"],
        track["ice_water_content"], track["height"],
    )
    try:
        matches = match_track_to_slstr(
            track["lat"], track["lon"], track["time"], slstr_root,
            orac_vars=_SLSTR_SYNERGY_VARS, max_time_diff_seconds=max_dt_s,
            catalog=catalog,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] match failed ({type(e).__name__}): {e}", file=sys.stderr)
        return None, "fail"

    matches["cot_water_atlid"] = cot_w
    matches["cer_water_atlid"] = cer_w
    matches["liquid_present_atlid"] = liq_present
    matches["ice_present_atlid"] = ice_present
    matches["liquid_only_atlid"] = liq_present & ~ice_present
    matches["quality_status_atlid"] = track["quality_status"]
    matches["convergence_status_atlid"] = track["convergence_status"]
    matches["synergy_status_atlid"] = track["synergy_status"]
    matches["cost_function_atlid"] = track["cost_function"]
    matches["atlid_assim_status"] = track["atlid_assim_status"]
    matches["cpr_assim_status"] = track["cpr_assim_status"]
    matches["frame_id"] = fid
    if not matches["valid_match"].any():
        return None, "empty"

    matches = matches.rename(columns={
        "cot": "cot_orac", "cer": "cer_orac", **_SLSTR_ORAC_RENAME,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out_csv, index=False)
    return out_csv, "done"


def _process_frame_cot_slstr(
    path: Path, catalog: GranuleCatalog, slstr_root: Path, max_dt_s: float, out_dir: Path
) -> tuple[Path | None, str]:
    """Match one A-EBD frame to SLSTR ORAC (ice cot); write matches_cot_<frame_id>.csv."""
    meta = _frame_metadata(path)
    if meta is None:
        return None, "fail"
    fid = meta[0]
    out_csv = out_dir / f"matches_cot_{fid}.csv"
    if out_csv.exists() and out_csv.stat().st_size > 0:
        return out_csv, "skip"
    try:
        track = read_aebd_track(path)
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] read failed: {e}", file=sys.stderr)
        return None, "fail"

    cot, attenuated = cot_from_aebd(
        track["extinction"], track["height"], track["quality_status"]
    )
    try:
        matches = match_track_to_slstr(
            track["lat"], track["lon"], track["time"], slstr_root,
            orac_vars=_SLSTR_COT_VARS, max_time_diff_seconds=max_dt_s,
            catalog=catalog,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] match failed ({type(e).__name__}): {e}", file=sys.stderr)
        return None, "fail"

    matches["cot_atlid"] = cot
    matches["attenuated"] = attenuated
    matches["frame_id"] = fid
    if not matches["valid_match"].any():
        return None, "empty"

    matches = matches.rename(columns={"cot": "cot_orac", **_SLSTR_ORAC_RENAME})
    matches["cot_orac_saturated"] = matches["cot_orac"] >= ORAC_COT_SATURATION
    out_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out_csv, index=False)
    return out_csv, "done"


def _run_slstr_collocate(args: argparse.Namespace, driver: str, processor) -> int:
    start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    max_dt_s = args.max_time_diff_min * 60.0
    slstr_root = Path(args.slstr_root)

    frames = _enumerate_frames(driver, start, end)
    print(f"Found {len(frames)} {driver} frames in [{start}, {end})")
    if not frames:
        return 0

    catalog = _load_or_build_slstr_catalog(slstr_root, start, end, max_dt_s)
    if not catalog.granules:
        print("No SLSTR granules in the run window — nothing to match.", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "skip": 0, "empty": 0, "fail": 0}
    for i, path in enumerate(frames, 1):
        meta = _frame_metadata(path)
        fid = meta[0] if meta else "?"
        _, status = processor(path, catalog, slstr_root, max_dt_s, out_dir)
        counts[status] += 1
        marker = {"done": "✓", "skip": "·", "empty": "○", "fail": "✗"}[status]
        print(f"  [{i:4d}/{len(frames)}] {marker} {fid:>8} → {status}")
    print(f"Summary: {counts}")
    return 0


def cmd_slstr_cth_collocate(args: argparse.Namespace) -> int:
    return _run_slstr_collocate(args, "A-CTH", _process_frame_cth_slstr)


def cmd_slstr_synergy_collocate(args: argparse.Namespace) -> int:
    return _run_slstr_collocate(args, "ACM-CAP", _process_frame_synergy_slstr)


def cmd_slstr_collocate(args: argparse.Namespace) -> int:
    return _run_slstr_collocate(args, "A-EBD", _process_frame_cot_slstr)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> int:
    paths = sorted(glob(args.matches))
    if not paths:
        print(f"No matches CSVs at {args.matches}", file=sys.stderr)
        return 1
    parts = [pd.read_csv(p) for p in paths]
    matches = pd.concat(parts, ignore_index=True)
    print(f"Concatenated {len(paths)} CSVs → {len(matches)} rows")

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Cast types after concat (CSV loses datetime / dtypes).
    if "sev_scan_time" in matches.columns:
        matches["sev_scan_time"] = pd.to_datetime(matches["sev_scan_time"], errors="coerce")

    report = cot_report(matches)
    parts_out = [report[k].assign(view=k) for k in report]
    out = pd.concat(parts_out, ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(out)} rows; views={list(report)})")

    if args.write_concat:
        concat_path = out_csv.with_suffix(".matches.csv")
        matches.to_csv(concat_path, index=False)
        print(f"Wrote concatenated matches to {concat_path}")
    return 0


# ---------------------------------------------------------------------------
# cth-evaluate
# ---------------------------------------------------------------------------

def cmd_cth_evaluate(args: argparse.Namespace) -> int:
    paths = sorted(glob(args.matches))
    if not paths:
        print(f"No matches CSVs at {args.matches}", file=sys.stderr)
        return 1
    parts = [pd.read_csv(p) for p in paths]
    matches = pd.concat(parts, ignore_index=True)
    print(f"Concatenated {len(paths)} CSVs → {len(matches)} rows")

    if "sev_scan_time" in matches.columns:
        matches["sev_scan_time"] = pd.to_datetime(matches["sev_scan_time"], errors="coerce")

    out = cth_report(matches)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(out)} rows; "
          f"qc_modes={sorted(out['qc_mode'].unique())}, "
          f"views={sorted(out['view'].unique())})")

    if args.write_concat:
        concat_path = out_csv.with_suffix(".matches.csv")
        matches.to_csv(concat_path, index=False)
        print(f"Wrote concatenated matches to {concat_path}")
    return 0


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def cmd_figures(args: argparse.Namespace) -> int:
    paths = sorted(glob(args.matches))
    if not paths:
        print(f"No matches CSVs at {args.matches}", file=sys.stderr)
        return 1
    parts = [pd.read_csv(p) for p in paths]
    matches = pd.concat(parts, ignore_index=True)
    print(f"Loaded {len(matches)} rows from {len(paths)} CSVs")

    base_mask = (matches["valid_match"]
                 & (matches["cldmask_orac"] == 1)
                 & (matches["cot_atlid"] > 0)
                 & (~matches["cot_orac_saturated"])
                 & matches["cot_atlid"].notna()
                 & matches["cot_orac"].notna())
    if args.ice_only and "phase_orac" in matches.columns:
        n_pre = int(base_mask.sum())
        base_mask &= matches["phase_orac"] == 2
        n_post = int(base_mask.sum())
        print(f"Ice filter: {n_pre} → {n_post} rows (phase_orac == 2)")
    base_all = matches[base_mask].copy()
    # Headline view drops attenuated (τ lower bounds, not point-comparable
    # to ORAC). Diagnostic panel keeps them so the attenuated class is
    # visible alongside non-attenuated.
    if "attenuated" in base_all.columns:
        att_mask = base_all["attenuated"].fillna(False).astype(bool)
        base = base_all[~att_mask]
        n_att = int(att_mask.sum())
        print(f"Base (all): {len(base_all)}  attenuated dropped: {n_att}  "
              f"headline base: {len(base)}")
    else:
        base = base_all
    pix = aggregate_to_pixel(base, "cot_atlid", "cot_orac")
    print(f"Sample-level: {len(base)}  Pixel-aggregate: {len(pix)}")

    out_dir = Path(args.out)
    suptitle = args.label or "cot validation"
    scatter_panel(base, pix, suptitle=f"{suptitle} — scatter",
                  out=out_dir / "cot_scatter.png")
    diagnostic_panel(base_all, suptitle=f"{suptitle} — diagnostic",
                     out=out_dir / "cot_diagnostic.png")

    sample_stats = stratified_stats(base, "cot_atlid", "cot_orac")
    pix_stats = stratified_stats(pix, "cot_atlid", "cot_orac")
    bias_by_stratum(sample_stats, metric="bias",
                    title=f"{suptitle} — bias by stratum (sample)",
                    out=out_dir / "cot_bias_by_stratum_sample.png")
    bias_by_stratum(pix_stats, metric="bias",
                    title=f"{suptitle} — bias by stratum (pixel)",
                    out=out_dir / "cot_bias_by_stratum_pixel.png")
    bias_by_stratum(pix_stats, metric="r",
                    title=f"{suptitle} — R by stratum (pixel)",
                    out=out_dir / "cot_r_by_stratum_pixel.png")
    print(f"Wrote 5 PNGs to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# cth-figures
# ---------------------------------------------------------------------------

def _load_cth_matches(matches_glob: str) -> pd.DataFrame:
    paths = sorted(glob(matches_glob))
    if not paths:
        raise FileNotFoundError(f"No matches CSVs at {matches_glob}")
    parts = [pd.read_csv(p) for p in paths]
    df = pd.concat(parts, ignore_index=True)
    if "sev_scan_time" in df.columns:
        df["sev_scan_time"] = pd.to_datetime(df["sev_scan_time"], errors="coerce")
    return df


def _cth_base_filter(d: pd.DataFrame) -> pd.Series:
    return (
        d["valid_match"]
        & (d["cldmask_orac"] == 1)
        & d["cth_atlid_thick_km"].notna()
        & d["cth_orac_corrected_km"].notna()
    )


def cmd_cth_figures(args: argparse.Namespace) -> int:
    matches = _load_cth_matches(args.matches)
    print(f"Loaded {len(matches)} rows")

    base = matches[_cth_base_filter(matches)].copy()
    qc_mask = CTH_QC_MODES[args.qc_mode](base).fillna(False)
    headline = base[qc_mask]
    print(f"Base after qc='{args.qc_mode}': {len(headline)} rows "
          f"(from {len(base)} cloudy+finite, {len(matches)} raw)")

    sample = dedupe_to_sample(headline)
    pixel = aggregate_to_pixel_cth(headline)
    print(f"Sample-level (nearest ATLID per pixel): {len(sample)}  "
          f"Pixel-aggregate (mean cloudy ATLID): {len(pixel)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suptitle = args.label or f"cth validation ({args.qc_mode})"

    cth_figures.scatter_panel(
        sample, pixel, suptitle=f"{suptitle} — scatter",
        out=out_dir / "cth_scatter.png",
    )
    cth_figures.diagnostic_panel(
        sample, suptitle=f"{suptitle} — diagnostic",
        out=out_dir / "cth_diagnostic.png",
    )

    sample_stats = stratified_stats(sample, "cth_atlid_thick_km",
                                    "cth_orac_corrected_km", strata=cth_strata())
    pixel_stats = stratified_stats(pixel, "cth_atlid_thick_km",
                                   "cth_orac_corrected_km", strata=cth_strata())
    cth_figures.bias_by_stratum(
        sample_stats, metric="bias",
        title=f"{suptitle} — bias by stratum (sample)",
        out=out_dir / "cth_bias_by_stratum_sample.png",
    )
    cth_figures.bias_by_stratum(
        pixel_stats, metric="bias",
        title=f"{suptitle} — bias by stratum (pixel)",
        out=out_dir / "cth_bias_by_stratum_pixel.png",
    )
    cth_figures.bias_by_stratum(
        pixel_stats, metric="r",
        title=f"{suptitle} — R by stratum (pixel)",
        out=out_dir / "cth_r_by_stratum_pixel.png",
    )

    # QC sensitivity uses cth_report on the full matches table.
    qc_stats = cth_report(matches)
    cth_figures.qc_sensitivity_panel(
        qc_stats,
        title=f"{suptitle.split(' (')[0]} — QC sensitivity (all-stratum)",
        out=out_dir / "cth_qc_sensitivity.png",
    )
    print(f"Wrote 6 PNGs to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# compare (R10 vs R11)
# ---------------------------------------------------------------------------

def _load_filtered(matches_glob: str, ice_only: bool) -> pd.DataFrame:
    paths = sorted(glob(matches_glob))
    if not paths:
        raise FileNotFoundError(f"No matches CSVs at {matches_glob}")
    parts = [pd.read_csv(p) for p in paths]
    df = pd.concat(parts, ignore_index=True)
    mask = (df["valid_match"]
            & (df["cldmask_orac"] == 1)
            & (df["cot_atlid"] > 0)
            & (~df["cot_orac_saturated"])
            & df["cot_atlid"].notna()
            & df["cot_orac"].notna())
    if "attenuated" in df.columns:
        mask &= ~df["attenuated"].fillna(False).astype(bool)
    if ice_only and "phase_orac" in df.columns:
        mask &= df["phase_orac"] == 2
    return df[mask].copy()


def cmd_compare(args: argparse.Namespace) -> int:
    base_r10 = _load_filtered(args.matches_r10, args.ice_only)
    base_r11 = _load_filtered(args.matches_r11, args.ice_only)
    print(f"R10 base: {len(base_r10)}  R11 base: {len(base_r11)}  "
          f"(ice_only={args.ice_only})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suptitle = args.label or ("ice-only cot validation" if args.ice_only else "cot validation")

    scatter_compare(base_r10, base_r11,
                    suptitle=f"{suptitle} — R10 vs R11 (sample-level)",
                    out=out_dir / "compare_R10_R11_scatter_sample.png")
    scatter_compare_by_surface(base_r10, base_r11,
                               suptitle=f"{suptitle} — R10 vs R11 by surface (sample-level)",
                               out=out_dir / "compare_R10_R11_scatter_sample_by_surface.png")

    pix_r10 = aggregate_to_pixel(base_r10, "cot_atlid", "cot_orac")
    pix_r11 = aggregate_to_pixel(base_r11, "cot_atlid", "cot_orac")
    scatter_compare(pix_r10, pix_r11,
                    suptitle=f"{suptitle} — R10 vs R11 (pixel-aggregate)",
                    out=out_dir / "compare_R10_R11_scatter_pixel.png")
    scatter_compare_by_surface(pix_r10, pix_r11,
                               suptitle=f"{suptitle} — R10 vs R11 by surface (pixel-aggregate)",
                               out=out_dir / "compare_R10_R11_scatter_pixel_by_surface.png")

    s10 = stratified_stats(base_r10, "cot_atlid", "cot_orac")
    s11 = stratified_stats(base_r11, "cot_atlid", "cot_orac")
    bias_bar_compare(s10, s11, metric="bias",
                     title=f"{suptitle} — R10 vs R11 bias (sample)",
                     out=out_dir / "compare_R10_R11_bias_sample.png")
    bias_bar_compare(s10, s11, metric="r_log",
                     title=f"{suptitle} — R10 vs R11 R_log (sample)",
                     out=out_dir / "compare_R10_R11_r_log_sample.png")

    p10 = stratified_stats(pix_r10, "cot_atlid", "cot_orac")
    p11 = stratified_stats(pix_r11, "cot_atlid", "cot_orac")
    bias_bar_compare(p10, p11, metric="bias",
                     title=f"{suptitle} — R10 vs R11 bias (pixel)",
                     out=out_dir / "compare_R10_R11_bias_pixel.png")
    bias_bar_compare(p10, p11, metric="r_log",
                     title=f"{suptitle} — R10 vs R11 R_log (pixel)",
                     out=out_dir / "compare_R10_R11_r_log_pixel.png")

    # Combined stats CSV.
    out_stats = pd.concat([
        s10.assign(view="sample", retrieval="R10"),
        s11.assign(view="sample", retrieval="R11"),
        p10.assign(view="pixel",  retrieval="R10"),
        p11.assign(view="pixel",  retrieval="R11"),
    ], ignore_index=True)
    out_stats.to_csv(out_dir / "compare_R10_R11_stats.csv", index=False)
    print(f"Wrote 8 PNGs and stats CSV to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# cth-compare (R10 vs R11)
# ---------------------------------------------------------------------------

def cmd_cth_compare(args: argparse.Namespace) -> int:
    raw_r10 = _load_cth_matches(args.matches_r10)
    raw_r11 = _load_cth_matches(args.matches_r11)
    print(f"R10 raw: {len(raw_r10)}  R11 raw: {len(raw_r11)}")

    qc_fn = CTH_QC_MODES[args.qc_mode]
    base_r10 = raw_r10[_cth_base_filter(raw_r10) & qc_fn(raw_r10).fillna(False)].copy()
    base_r11 = raw_r11[_cth_base_filter(raw_r11) & qc_fn(raw_r11).fillna(False)].copy()
    print(f"R10 after qc='{args.qc_mode}': {len(base_r10)}  "
          f"R11 after qc='{args.qc_mode}': {len(base_r11)}")

    sample_r10 = dedupe_to_sample(base_r10)
    sample_r11 = dedupe_to_sample(base_r11)
    pixel_r10 = aggregate_to_pixel_cth(base_r10)
    pixel_r11 = aggregate_to_pixel_cth(base_r11)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suptitle = args.label or f"cth validation R10 vs R11 ({args.qc_mode})"

    cth_figures.scatter_compare(
        sample_r10, sample_r11,
        suptitle=f"{suptitle} — sample-level (nearest ATLID)",
        out=out_dir / "compare_R10_R11_scatter_sample.png",
    )
    cth_figures.scatter_compare(
        pixel_r10, pixel_r11,
        suptitle=f"{suptitle} — pixel-aggregate (mean cloudy ATLID)",
        out=out_dir / "compare_R10_R11_scatter_pixel.png",
    )
    cth_figures.scatter_compare_by_surface(
        sample_r10, sample_r11,
        suptitle=f"{suptitle} — sample-level by surface",
        out=out_dir / "compare_R10_R11_scatter_sample_by_surface.png",
    )
    cth_figures.scatter_compare_by_surface(
        pixel_r10, pixel_r11,
        suptitle=f"{suptitle} — pixel-aggregate by surface",
        out=out_dir / "compare_R10_R11_scatter_pixel_by_surface.png",
    )

    s10 = stratified_stats(sample_r10, "cth_atlid_thick_km",
                           "cth_orac_corrected_km", strata=cth_strata())
    s11 = stratified_stats(sample_r11, "cth_atlid_thick_km",
                           "cth_orac_corrected_km", strata=cth_strata())
    p10 = stratified_stats(pixel_r10, "cth_atlid_thick_km",
                           "cth_orac_corrected_km", strata=cth_strata())
    p11 = stratified_stats(pixel_r11, "cth_atlid_thick_km",
                           "cth_orac_corrected_km", strata=cth_strata())

    cth_figures.bias_bar_compare(s10, s11, metric="bias",
        title=f"{suptitle} — R10 vs R11 bias (sample)",
        out=out_dir / "compare_R10_R11_bias_sample.png")
    cth_figures.bias_bar_compare(p10, p11, metric="bias",
        title=f"{suptitle} — R10 vs R11 bias (pixel)",
        out=out_dir / "compare_R10_R11_bias_pixel.png")
    cth_figures.bias_bar_compare(p10, p11, metric="r",
        title=f"{suptitle} — R10 vs R11 R (pixel)",
        out=out_dir / "compare_R10_R11_r_pixel.png")
    cth_figures.bias_bar_compare(p10, p11, metric="rmse",
        title=f"{suptitle} — R10 vs R11 RMSE (pixel)",
        out=out_dir / "compare_R10_R11_rmse_pixel.png")

    out_stats = pd.concat([
        s10.assign(view="sample", retrieval="R10"),
        s11.assign(view="sample", retrieval="R11"),
        p10.assign(view="pixel",  retrieval="R10"),
        p11.assign(view="pixel",  retrieval="R11"),
    ], ignore_index=True).assign(qc_mode=args.qc_mode)
    out_stats.to_csv(out_dir / "compare_R10_R11_stats.csv", index=False)
    print(f"Wrote 6 PNGs and stats CSV to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# water-cloud (cot_water, cer_water) evaluate / figures / compare
# ---------------------------------------------------------------------------

def _load_synergy_matches(matches_glob: str) -> pd.DataFrame:
    paths = sorted(glob(matches_glob))
    if not paths:
        raise FileNotFoundError(f"No matches CSVs at {matches_glob}")
    parts = [pd.read_csv(p) for p in paths]
    df = pd.concat(parts, ignore_index=True)
    if "sev_scan_time" in df.columns:
        df["sev_scan_time"] = pd.to_datetime(df["sev_scan_time"], errors="coerce")
    return df


def _water_base_filter(d: pd.DataFrame, var_atlid: str, var_orac: str) -> pd.Series:
    return (
        d["valid_match"]
        & (d["cldmask_orac"] == 1)
        & d[var_atlid].notna()
        & d[var_orac].notna()
    )


def _cmd_water_evaluate(args, *, var_atlid, var_orac, report_fn) -> int:
    matches = _load_synergy_matches(args.matches)
    print(f"Concatenated {len(matches)} rows")
    out = report_fn(
        matches,
        min_n_liquid_only=args.min_n_liquid_only,
        min_n_total=args.min_n_total,
    )
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(out)} rows; "
          f"qc_modes={sorted(out['qc_mode'].unique())}, "
          f"views={sorted(out['view'].unique())})")
    if args.write_concat:
        concat_path = out_csv.with_suffix(".matches.csv")
        matches.to_csv(concat_path, index=False)
        print(f"Wrote concatenated matches to {concat_path}")
    return 0


def cmd_cot_water_evaluate(args: argparse.Namespace) -> int:
    return _cmd_water_evaluate(args, var_atlid="cot_water_atlid",
                                var_orac="cot_orac", report_fn=cot_water_report)


def cmd_cer_water_evaluate(args: argparse.Namespace) -> int:
    return _cmd_water_evaluate(args, var_atlid="cer_water_atlid",
                                var_orac="cer_orac", report_fn=cer_water_report)


def _cmd_water_figures(args, *, mode, var_atlid, var_orac, strata_fn, report_fn,
                        prefix: str) -> int:
    matches = _load_synergy_matches(args.matches)
    print(f"Loaded {len(matches)} rows")
    base = matches[_water_base_filter(matches, var_atlid, var_orac)].copy()
    qc_mask = SYNERGY_QC_MODES[args.qc_mode](base).fillna(False)
    headline = base[qc_mask]
    print(f"Base after qc='{args.qc_mode}': {len(headline)} rows "
          f"(from {len(base)} cloudy+finite, {len(matches)} raw)")
    headline_annotated, pixel = filter_water_sampling(
        headline, var_atlid, var_orac,
        min_n_liquid_only=args.min_n_liquid_only,
        min_n_total=args.min_n_total,
    )
    sample = dedupe_to_sample_water(headline_annotated) if not headline_annotated.empty else headline_annotated
    print(f"Sampling filter: n_liquid_only >= {args.min_n_liquid_only}, n_total >= {args.min_n_total}")
    print(f"Sample-level: {len(sample)}  Pixel-aggregate: {len(pixel)}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    suptitle = args.label or f"{prefix} validation ({args.qc_mode})"

    water_cloud_figures.scatter_panel(
        sample, pixel, mode=mode, x=var_atlid, y=var_orac,
        suptitle=f"{suptitle} — scatter",
        out=out_dir / f"{prefix}_scatter.png",
    )

    sample_stats = stratified_stats(sample, var_atlid, var_orac, strata=strata_fn())
    pixel_stats = stratified_stats(pixel, var_atlid, var_orac, strata=strata_fn())
    water_cloud_figures.bias_by_stratum(
        sample_stats, metric="bias",
        title=f"{suptitle} — bias by stratum (sample)",
        out=out_dir / f"{prefix}_bias_by_stratum_sample.png")
    water_cloud_figures.bias_by_stratum(
        pixel_stats, metric="bias",
        title=f"{suptitle} — bias by stratum (pixel)",
        out=out_dir / f"{prefix}_bias_by_stratum_pixel.png")
    water_cloud_figures.bias_by_stratum(
        pixel_stats, metric="r",
        title=f"{suptitle} — R by stratum (pixel)",
        out=out_dir / f"{prefix}_r_by_stratum_pixel.png")

    qc_stats = report_fn(
        matches,
        min_n_liquid_only=args.min_n_liquid_only,
        min_n_total=args.min_n_total,
    )
    water_cloud_figures.qc_sensitivity_panel(
        qc_stats, title=f"{suptitle.split(' (')[0]} — QC sensitivity",
        out=out_dir / f"{prefix}_qc_sensitivity.png")
    print(f"Wrote 5 PNGs to {out_dir}")
    return 0


def cmd_cot_water_figures(args: argparse.Namespace) -> int:
    mode = "cot_linear" if getattr(args, "scale", "log") == "linear" else "cot"
    return _cmd_water_figures(args, mode=mode,
                               var_atlid="cot_water_atlid", var_orac="cot_orac",
                               strata_fn=cot_water_strata, report_fn=cot_water_report,
                               prefix="cot_water")


def cmd_cer_water_figures(args: argparse.Namespace) -> int:
    return _cmd_water_figures(args, mode="cer",
                               var_atlid="cer_water_atlid", var_orac="cer_orac",
                               strata_fn=cer_water_strata, report_fn=cer_water_report,
                               prefix="cer_water")


def _cmd_water_compare(args, *, mode, var_atlid, var_orac, strata_fn, prefix) -> int:
    raw_r10 = _load_synergy_matches(args.matches_r10)
    raw_r11 = _load_synergy_matches(args.matches_r11)
    print(f"R10 raw: {len(raw_r10)}  R11 raw: {len(raw_r11)}")
    qc_fn = SYNERGY_QC_MODES[args.qc_mode]
    base_r10 = raw_r10[_water_base_filter(raw_r10, var_atlid, var_orac)
                       & qc_fn(raw_r10).fillna(False)].copy()
    base_r11 = raw_r11[_water_base_filter(raw_r11, var_atlid, var_orac)
                       & qc_fn(raw_r11).fillna(False)].copy()
    print(f"R10 after qc='{args.qc_mode}': {len(base_r10)}  R11: {len(base_r11)}")

    ann_r10, pixel_r10 = filter_water_sampling(
        base_r10, var_atlid, var_orac,
        min_n_liquid_only=args.min_n_liquid_only,
        min_n_total=args.min_n_total,
    )
    ann_r11, pixel_r11 = filter_water_sampling(
        base_r11, var_atlid, var_orac,
        min_n_liquid_only=args.min_n_liquid_only,
        min_n_total=args.min_n_total,
    )
    sample_r10 = dedupe_to_sample_water(ann_r10) if not ann_r10.empty else ann_r10
    sample_r11 = dedupe_to_sample_water(ann_r11) if not ann_r11.empty else ann_r11
    print(f"Sampling filter: n_liquid_only >= {args.min_n_liquid_only}, n_total >= {args.min_n_total}")
    print(f"R10 retained: sample={len(sample_r10)} pixel={len(pixel_r10)}  "
          f"R11 retained: sample={len(sample_r11)} pixel={len(pixel_r11)}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    suptitle = args.label or f"{prefix} validation R10 vs R11 ({args.qc_mode})"

    water_cloud_figures.scatter_compare(
        sample_r10, sample_r11, mode=mode, x=var_atlid, y=var_orac,
        suptitle=f"{suptitle} — sample-level (nearest ATLID)",
        out=out_dir / f"compare_R10_R11_scatter_sample.png")
    water_cloud_figures.scatter_compare(
        pixel_r10, pixel_r11, mode=mode, x=var_atlid, y=var_orac,
        suptitle=f"{suptitle} — pixel-aggregate (mean cloudy ATLID)",
        out=out_dir / f"compare_R10_R11_scatter_pixel.png")
    water_cloud_figures.scatter_compare_by_surface(
        sample_r10, sample_r11, mode=mode, x=var_atlid, y=var_orac,
        suptitle=f"{suptitle} — sample-level by surface",
        out=out_dir / f"compare_R10_R11_scatter_sample_by_surface.png")
    water_cloud_figures.scatter_compare_by_surface(
        pixel_r10, pixel_r11, mode=mode, x=var_atlid, y=var_orac,
        suptitle=f"{suptitle} — pixel-aggregate by surface",
        out=out_dir / f"compare_R10_R11_scatter_pixel_by_surface.png")

    s10 = stratified_stats(sample_r10, var_atlid, var_orac, strata=strata_fn())
    s11 = stratified_stats(sample_r11, var_atlid, var_orac, strata=strata_fn())
    p10 = stratified_stats(pixel_r10, var_atlid, var_orac, strata=strata_fn())
    p11 = stratified_stats(pixel_r11, var_atlid, var_orac, strata=strata_fn())
    for name, a, b, view in [("bias_sample", s10, s11, "sample"),
                              ("bias_pixel",  p10, p11, "pixel"),
                              ("r_pixel",     p10, p11, "pixel"),
                              ("rmse_pixel",  p10, p11, "pixel")]:
        metric = name.split("_")[0]
        water_cloud_figures.bias_bar_compare(
            a, b, metric=metric,
            title=f"{suptitle} — {metric} ({view})",
            out=out_dir / f"compare_R10_R11_{name}.png")

    out_stats = pd.concat([
        s10.assign(view="sample", retrieval="R10"),
        s11.assign(view="sample", retrieval="R11"),
        p10.assign(view="pixel",  retrieval="R10"),
        p11.assign(view="pixel",  retrieval="R11"),
    ], ignore_index=True).assign(qc_mode=args.qc_mode)
    out_stats.to_csv(out_dir / "compare_R10_R11_stats.csv", index=False)
    print(f"Wrote 8 PNGs and stats CSV to {out_dir}")
    return 0


def cmd_cot_water_compare(args: argparse.Namespace) -> int:
    mode = "cot_linear" if getattr(args, "scale", "log") == "linear" else "cot"
    return _cmd_water_compare(args, mode=mode,
                               var_atlid="cot_water_atlid", var_orac="cot_orac",
                               strata_fn=cot_water_strata, prefix="cot_water")


def cmd_cer_water_compare(args: argparse.Namespace) -> int:
    return _cmd_water_compare(args, mode="cer",
                               var_atlid="cer_water_atlid", var_orac="cer_orac",
                               strata_fn=cer_water_strata, prefix="cer_water")


# ---------------------------------------------------------------------------
# homogeneity sweep
# ---------------------------------------------------------------------------

def _cmd_water_homogeneity(args, *, var_atlid: str, var_orac: str, prefix: str,
                            r_metric: str = "r", r_label: str = "Pearson R") -> int:
    n_cuts = tuple(int(x) for x in args.n_cuts)
    cv_edges = tuple(float(x) for x in args.cv_edges)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    qc_fn = SYNERGY_QC_MODES[args.qc_mode]

    sources: list[tuple[str, str]] = []
    if args.matches_r10:
        sources.append(("R10", args.matches_r10))
    if args.matches_r11:
        sources.append(("R11", args.matches_r11))
    if not sources:
        print("Need at least --matches-r10 or --matches-r11", file=sys.stderr)
        return 1

    all_stats: list[pd.DataFrame] = []
    for label, matches_glob in sources:
        raw = _load_synergy_matches(matches_glob)
        base = raw[_water_base_filter(raw, var_atlid, var_orac)
                   & qc_fn(raw).fillna(False)].copy()
        pixel = aggregate_to_pixel_water(base, var_atlid, var_orac)
        print(f"{label}: raw={len(raw)} qc='{args.qc_mode}'={len(base)} "
              f"pixel={len(pixel)}")

        sweep = homogeneity_sweep_stats(
            pixel, var_atlid, var_orac, n_cuts=n_cuts, cv_edges=cv_edges,
        )
        sweep_out = sweep.assign(retrieval=label, qc_mode=args.qc_mode)
        all_stats.append(sweep_out)

        suptitle = (args.label or f"{prefix} homogeneity sweep") + f" — {label} ({args.qc_mode})"
        water_cloud_figures.homogeneity_sweep(
            sweep, r_metric=r_metric, r_label=r_label,
            title=suptitle,
            out=out_dir / f"{prefix}_homogeneity_{label}.png",
        )

    stats = pd.concat(all_stats, ignore_index=True)
    stats.to_csv(out_dir / f"{prefix}_homogeneity_stats.csv", index=False)
    print(f"Wrote {len(sources)} PNG(s) and stats CSV to {out_dir}")
    return 0


def cmd_cot_water_homogeneity(args: argparse.Namespace) -> int:
    return _cmd_water_homogeneity(args, var_atlid="cot_water_atlid",
                                   var_orac="cot_orac", prefix="cot_water",
                                   r_metric="r_log", r_label="Pearson R (log COT)")


def cmd_cer_water_homogeneity(args: argparse.Namespace) -> int:
    return _cmd_water_homogeneity(args, var_atlid="cer_water_atlid",
                                   var_orac="cer_orac", prefix="cer_water")


# ---------------------------------------------------------------------------
# track-plot
# ---------------------------------------------------------------------------

def cmd_track(args: argparse.Namespace) -> int:
    matches_csv = Path(args.matches_dir) / f"matches_cot_{args.frame}.csv"
    if not matches_csv.exists():
        print(f"matches CSV not found: {matches_csv}", file=sys.stderr)
        return 1
    df = pd.read_csv(matches_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    track_panel(
        args.frame, df, args.seviri_root, retrieval=args.retrieval,
        out=out_path,
    )
    print(f"Wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="validation")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collocate", help="Match A-EBD frames to SEVIRI ORAC.")
    c.add_argument("--driver", default="A-EBD", choices=("A-EBD", "A-CTH"))
    c.add_argument("--start", required=True, help="ISO start, e.g. 2026-02-01")
    c.add_argument("--end", required=True, help="ISO end (exclusive)")
    c.add_argument("--seviri-root", default=str(SEVIRI_ROOT_DEFAULT))
    c.add_argument("--retrieval", default=DEFAULT_RETRIEVAL, choices=("R10", "R11"))
    c.add_argument("--out", required=True, help="Per-frame matches CSV directory")
    c.set_defaults(func=cmd_collocate)

    e = sub.add_parser("evaluate", help="Concatenate matches CSVs + write stats.")
    e.add_argument("--matches", required=True, help="Glob, e.g. 'validation_data/2026-02/matches_*.csv'")
    e.add_argument("--out", required=True, help="Output stats CSV path")
    e.add_argument("--write-concat", action="store_true",
                   help="Also write the concatenated matches CSV alongside.")
    e.set_defaults(func=cmd_evaluate)

    f = sub.add_parser("figures", help="Make scatter / diagnostic / bar charts.")
    f.add_argument("--matches", required=True, help="Glob over matches CSVs")
    f.add_argument("--out", required=True, help="Output figure directory")
    f.add_argument("--label", default="", help="Title prefix")
    f.add_argument("--all-phases", dest="ice_only", action="store_false",
                   help="Skip the ice-phase filter (default: ice-only).")
    f.set_defaults(func=cmd_figures, ice_only=True)

    cmp = sub.add_parser("compare", help="R10 vs R11 ORAC retrieval comparison.")
    cmp.add_argument("--matches-r10", required=True, help="Glob over R10 matches CSVs")
    cmp.add_argument("--matches-r11", required=True, help="Glob over R11 matches CSVs")
    cmp.add_argument("--out", required=True, help="Output figure directory")
    cmp.add_argument("--label", default="", help="Title prefix")
    cmp.add_argument("--all-phases", dest="ice_only", action="store_false",
                     help="Skip the ice-phase filter (default: ice-only).")
    cmp.set_defaults(func=cmd_compare, ice_only=True)

    cc = sub.add_parser("cth-collocate", help="Match A-CTH frames to SEVIRI ORAC (no QC applied).")
    cc.add_argument("--start", required=True, help="ISO start, e.g. 2026-02-01")
    cc.add_argument("--end", required=True, help="ISO end (exclusive)")
    cc.add_argument("--seviri-root", default=str(SEVIRI_ROOT_DEFAULT))
    cc.add_argument("--retrieval", default=DEFAULT_RETRIEVAL, choices=("R10", "R11"))
    cc.add_argument("--out", required=True, help="Per-frame matches CSV directory")
    cc.set_defaults(func=cmd_cth_collocate)

    ce = sub.add_parser("cth-evaluate", help="Concatenate cth matches CSVs + write QC × view × stratum stats.")
    ce.add_argument("--matches", required=True, help="Glob, e.g. 'validation_data/cth_2026-02_R11/matches_cth_*.csv'")
    ce.add_argument("--out", required=True, help="Output stats CSV path")
    ce.add_argument("--write-concat", action="store_true",
                    help="Also write the concatenated matches CSV alongside.")
    ce.set_defaults(func=cmd_cth_evaluate)

    cf = sub.add_parser("cth-figures", help="Make cth scatter / diagnostic / bias-by-stratum / QC-sensitivity PNGs.")
    cf.add_argument("--matches", required=True, help="Glob over cth matches CSVs")
    cf.add_argument("--out", required=True, help="Output figure directory")
    cf.add_argument("--qc-mode", default="qc_strict",
                    choices=tuple(CTH_QC_MODES),
                    help="QC base filter for scatter / diagnostic / bias-by-stratum panels.")
    cf.add_argument("--label", default="", help="Title prefix")
    cf.set_defaults(func=cmd_cth_figures)

    sc = sub.add_parser("synergy-collocate",
                        help="Match ACM-CAP frames to SEVIRI ORAC (no QC; both cot+cer references).")
    sc.add_argument("--start", required=True, help="ISO start, e.g. 2026-02-01")
    sc.add_argument("--end", required=True, help="ISO end (exclusive)")
    sc.add_argument("--seviri-root", default=str(SEVIRI_ROOT_DEFAULT))
    sc.add_argument("--retrieval", default=DEFAULT_RETRIEVAL, choices=("R10", "R11"))
    sc.add_argument("--out", required=True, help="Per-frame matches CSV directory")
    sc.set_defaults(func=cmd_synergy_collocate)

    # SLSTR collocation (polar-orbiter swath). Same output schema as SEVIRI, so
    # cth-evaluate / cot-water-* / figures consume the CSVs unchanged.
    for name, fn, helptext in (
        ("slstr-cth-collocate", cmd_slstr_cth_collocate,
         "Match A-CTH frames to SLSTR ORAC (temporal-gated crossing match)."),
        ("slstr-synergy-collocate", cmd_slstr_synergy_collocate,
         "Match ACM-CAP frames to SLSTR ORAC (cot+cer water references)."),
        ("slstr-collocate", cmd_slstr_collocate,
         "Match A-EBD frames to SLSTR ORAC (ice cot reference)."),
    ):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--start", required=True, help="ISO start, e.g. 2025-12-01")
        sp.add_argument("--end", required=True, help="ISO end (exclusive)")
        sp.add_argument("--slstr-root", default=str(SLSTR_ROOT_DEFAULT))
        sp.add_argument("--max-time-diff-min", type=float, default=SLSTR_DEFAULT_MAX_DT_MIN,
                        help="Temporal match tolerance in minutes (default 30).")
        sp.add_argument("--out", required=True, help="Per-frame matches CSV directory")
        sp.set_defaults(func=fn)

    for var, ev_fn, fig_fn, cmp_fn, hom_fn in (
        ("cot-water", cmd_cot_water_evaluate, cmd_cot_water_figures, cmd_cot_water_compare, cmd_cot_water_homogeneity),
        ("cer-water", cmd_cer_water_evaluate, cmd_cer_water_figures, cmd_cer_water_compare, cmd_cer_water_homogeneity),
    ):
        ev = sub.add_parser(f"{var}-evaluate",
                            help=f"Concatenate synergy matches CSVs + write {var} stats.")
        ev.add_argument("--matches", required=True, help="Glob over synergy matches CSVs")
        ev.add_argument("--out", required=True, help="Output stats CSV path")
        ev.add_argument("--write-concat", action="store_true",
                        help="Also write concatenated matches CSV alongside.")
        ev.add_argument("--min-n-liquid-only", type=int, default=1,
                        help="Require at least this many liquid-only EarthCARE profiles in an ORAC pixel.")
        ev.add_argument("--min-n-total", type=int, default=1,
                        help="Require at least this many total EarthCARE profiles in an ORAC pixel.")
        ev.set_defaults(func=ev_fn)

        fg = sub.add_parser(f"{var}-figures", help=f"Make {var} figures.")
        fg.add_argument("--matches", required=True, help="Glob over synergy matches CSVs")
        fg.add_argument("--out", required=True, help="Output figure directory")
        fg.add_argument("--qc-mode", default="qc_strict",
                        choices=tuple(SYNERGY_QC_MODES),
                        help="QC base filter for the headline panels.")
        if var == "cot-water":
            fg.add_argument("--scale", default="log", choices=("log", "linear"),
                            help="COT scatter axis scale.")
        fg.add_argument("--min-n-liquid-only", type=int, default=1,
                        help="Require at least this many liquid-only EarthCARE profiles in an ORAC pixel.")
        fg.add_argument("--min-n-total", type=int, default=1,
                        help="Require at least this many total EarthCARE profiles in an ORAC pixel.")
        fg.add_argument("--label", default="", help="Title prefix")
        fg.set_defaults(func=fig_fn)

        cmp = sub.add_parser(f"{var}-compare",
                             help=f"R10 vs R11 ORAC retrieval comparison for {var}.")
        cmp.add_argument("--matches-r10", required=True, help="Glob over R10 synergy matches CSVs")
        cmp.add_argument("--matches-r11", required=True, help="Glob over R11 synergy matches CSVs")
        cmp.add_argument("--out", required=True, help="Output figure directory")
        cmp.add_argument("--qc-mode", default="qc_strict",
                         choices=tuple(SYNERGY_QC_MODES),
                         help="QC base filter applied to both R10 and R11.")
        if var == "cot-water":
            cmp.add_argument("--scale", default="log", choices=("log", "linear"),
                             help="COT scatter axis scale.")
        cmp.add_argument("--min-n-liquid-only", type=int, default=1,
                         help="Require at least this many liquid-only EarthCARE profiles in an ORAC pixel.")
        cmp.add_argument("--min-n-total", type=int, default=1,
                         help="Require at least this many total EarthCARE profiles in an ORAC pixel.")
        cmp.add_argument("--label", default="", help="Title prefix")
        cmp.set_defaults(func=cmp_fn)

        hom = sub.add_parser(
            f"{var}-homogeneity",
            help=f"Sweep {var} agreement across homogeneity bins for n>=1, 3, 5 sample cuts.",
        )
        hom.add_argument("--matches-r10", default="",
                         help="Glob over R10 synergy matches CSVs (optional)")
        hom.add_argument("--matches-r11", default="",
                         help="Glob over R11 synergy matches CSVs (optional)")
        hom.add_argument("--out", required=True, help="Output figure directory")
        hom.add_argument("--qc-mode", default="qc_strict",
                         choices=tuple(SYNERGY_QC_MODES),
                         help="QC base filter applied before pixel aggregation.")
        hom.add_argument("--n-cuts", nargs="+", type=int, default=[1, 3, 5],
                         help="Minimum-n_liquid_only cuts to overlay.")
        hom.add_argument("--cv-edges", nargs="+", type=float,
                         default=[0.0, 0.25, 0.75, float("inf")],
                         help="ref_cv_atlid bin edges.")
        hom.add_argument("--label", default="", help="Title prefix")
        hom.set_defaults(func=hom_fn)

    cmpcth = sub.add_parser("cth-compare", help="R10 vs R11 ORAC retrieval comparison for CTH.")
    cmpcth.add_argument("--matches-r10", required=True, help="Glob over R10 cth matches CSVs")
    cmpcth.add_argument("--matches-r11", required=True, help="Glob over R11 cth matches CSVs")
    cmpcth.add_argument("--out", required=True, help="Output figure directory")
    cmpcth.add_argument("--qc-mode", default="qc_strict",
                        choices=tuple(CTH_QC_MODES),
                        help="QC base filter applied to both R10 and R11.")
    cmpcth.add_argument("--label", default="", help="Title prefix")
    cmpcth.set_defaults(func=cmd_cth_compare)

    t = sub.add_parser("track-plot", help="Per-orbit case-study figure for one frame.")
    t.add_argument("--frame", required=True, help="A-EBD frame ID, e.g. 09737D")
    t.add_argument("--matches-dir", required=True,
                   help="Dir holding matches_cot_<frame>.csv files")
    t.add_argument("--seviri-root", default=str(SEVIRI_ROOT_DEFAULT))
    t.add_argument("--retrieval", default=DEFAULT_RETRIEVAL, choices=("R10", "R11"))
    t.add_argument("--out", required=True, help="Output PNG path")
    t.set_defaults(func=cmd_track)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
