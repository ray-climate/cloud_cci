"""Crossing-geometry Δt sweep for EarthCARE × ORAC-SLSTR collocation.

EarthCARE and Sentinel-3A are both polar orbiters, so coincidences are
crossing-limited and the temporal window is the binding design parameter. This
script quantifies the trade-off so we can pick ``--max-time-diff-min`` for the
full run.

Method: collocate a subset of A-CTH frames ONCE at a wide window (the matches
CSVs record the true ``time_diff_s`` per matched profile), then sweep candidate
windows by post-hoc filtering — no re-matching. For each window it reports the
valid-match count, the |latitude| distribution (crossings concentrate toward the
poles), and the median pixel distance.

Usage:
    # 1. collocate a week at a wide window (run once, data permitting):
    python -m validation slstr-cth-collocate \
        --start 2025-12-01 --end 2025-12-08 --max-time-diff-min 120 \
        --out validation_data/slstr_dtsweep_cth

    # 2. sweep + figure:
    python scripts/slstr_dt_sweep.py \
        --matches 'validation_data/slstr_dtsweep_cth/matches_cth_*.csv' \
        --out figures/slstr_dt_sweep
"""
from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WINDOWS_MIN = [5, 10, 15, 20, 30, 45, 60, 90, 120]


def load_matches(matches_glob: str) -> pd.DataFrame:
    paths = sorted(glob(matches_glob))
    if not paths:
        raise SystemExit(f"No matches CSVs at {matches_glob}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(paths)} frames")
    # Spatial hits only (a pixel was found within the swath); time gate applied
    # per-window below.
    return df[df["on_disk"].fillna(False)].copy()


def sweep(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    abslat = df["ec_lat"].abs()
    for w in WINDOWS_MIN:
        m = df["time_diff_s"] <= w * 60.0
        sub = df[m]
        rows.append({
            "window_min": w,
            "n_valid": int(m.sum()),
            "n_frames": int(sub["frame_id"].nunique()) if "frame_id" in sub else np.nan,
            "median_abslat": float(abslat[m].median()) if m.any() else np.nan,
            "frac_polar_ge60": float((abslat[m] >= 60).mean()) if m.any() else np.nan,
            "frac_tropics_lt30": float((abslat[m] < 30).mean()) if m.any() else np.nan,
            "median_dist_km": float(df.loc[m, "distance_km"].median()) if m.any() else np.nan,
            "median_tdiff_s": float(df.loc[m, "time_diff_s"].median()) if m.any() else np.nan,
        })
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, table: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot(table["window_min"], table["n_valid"], "o-", color="#1565c0")
    ax[0].set_xlabel("Δt window (min)")
    ax[0].set_ylabel("valid matches (profiles)")
    ax[0].set_title("Match count vs temporal window")
    ax[0].grid(alpha=0.3)

    ax[1].plot(table["window_min"], table["median_abslat"], "o-", color="#c0392b",
               label="median |lat|")
    ax[1].plot(table["window_min"], 100 * table["frac_polar_ge60"], "s--",
               color="#2e7d32", label="% polar (|lat|≥60)")
    ax[1].set_xlabel("Δt window (min)")
    ax[1].set_ylabel("degrees  /  %")
    ax[1].set_title("Latitude concentration of matches")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    # |lat| histogram of all spatial hits, coloured by whether they fall inside a
    # reference 30-min window.
    abslat = df["ec_lat"].abs()
    in30 = df["time_diff_s"] <= 30 * 60.0
    bins = np.arange(0, 91, 5)
    ax[2].hist(abslat[in30], bins=bins, color="#1565c0", alpha=0.8, label="Δt ≤ 30 min")
    ax[2].hist(abslat[~in30], bins=bins, color="#b0bec5", alpha=0.6, label="Δt > 30 min")
    ax[2].set_xlabel("|latitude| (deg)")
    ax[2].set_ylabel("matches")
    ax[2].set_title("Where crossings happen")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "slstr_dt_sweep.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matches", required=True, help="Glob over wide-window A-CTH matches CSVs")
    ap.add_argument("--out", required=True, help="Output figure directory")
    args = ap.parse_args()

    df = load_matches(args.matches)
    table = sweep(df)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "slstr_dt_sweep_stats.csv", index=False)
    png = make_figure(df, table, out_dir)

    pd.set_option("display.width", 120)
    print("\n=== Δt sweep ===")
    print(table.to_string(index=False))
    print(f"\nWrote {png}")
    print(f"Wrote {out_dir / 'slstr_dt_sweep_stats.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
