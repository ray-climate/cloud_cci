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
from .figures import bias_by_stratum, diagnostic_panel, scatter_panel
from .readers import read_aebd_track
from .reference import cot_from_aebd
from .statistics import aggregate_to_pixel, cot_report, stratified_stats
from .track_figures import track_panel

DEFAULT_DRIVER_DIR = {
    "A-EBD": "ATL_EBD_2A",
    "A-CTH": "ATL_CTH_2A",
}

ORAC_COT_SATURATION = 100.0
DEFAULT_RETRIEVAL = "R11"
EARTHCARE_ROOT = Path("earthcare_data")
SEVIRI_ROOT_DEFAULT = Path("/gws/ssde/j25a/cloud_ecv/data_out/seviri")
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
            matches, seviri_root, retrieval, ("cot", "cldmask", "lsflag")
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [{fid}] ORAC sample failed ({type(e).__name__}): {e}", file=sys.stderr)
        return None, "fail"
    matches = matches.rename(columns={"cot": "cot_orac", "cldmask": "cldmask_orac",
                                       "lsflag": "lsflag_orac"})
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

    base_all = matches[
        matches["valid_match"]
        & (matches["cldmask_orac"] == 1)
        & (matches["cot_atlid"] > 0)
        & (~matches["cot_orac_saturated"])
        & matches["cot_atlid"].notna()
        & matches["cot_orac"].notna()
    ].copy()
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
    f.set_defaults(func=cmd_figures)

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
