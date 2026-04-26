"""End-to-end cot validation smoke test on one frame.

Reads ATL_EBD_2A frame 09541B (2026-02-01 03:09–03:48 UTC), integrates
particle extinction at 355 nm to a column τ per profile, matches each
profile to its nearest SEVIRI ORAC pixel in the closest 15-min slot,
and writes:

    validation_data/matches_cot_2026-02-01_09541B.csv
    figures/validation/cot_smoke_scatter.png

Run::

    python scripts/validate_cot_smoke.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validation import cot_from_aebd, match_track_to_seviri, read_aebd_track
from validation.collocate import open_seviri_at_matches

AEBD_FILE = Path(
    "earthcare_data/ATL_EBD_2A/2026/02/01/"
    "ECA_EXBC_ATL_EBD_2A_20260201T030940Z_20260201T034848Z_09541B.h5"
)
SEVIRI_ROOT = Path("/gws/ssde/j25a/cloud_ecv/data_out/seviri")
RETRIEVAL = "R11"

OUT_CSV = Path("validation_data/matches_cot_2026-02-01_09541B.csv")
OUT_PNG = Path("figures/validation/cot_smoke_scatter.png")


def main() -> None:
    print(f"Reading {AEBD_FILE.name}")
    track = read_aebd_track(AEBD_FILE)
    n_prof = len(track["lat"])
    print(f"  {n_prof} ATLID profiles, {track['extinction'].shape[1]} bins")
    print(f"  lat {track['lat'].min():.2f}–{track['lat'].max():.2f},"
          f"  lon {track['lon'].min():.2f}–{track['lon'].max():.2f}")
    print(f"  time {pd.Timestamp(track['time'].min())} – {pd.Timestamp(track['time'].max())}")

    print("Integrating extinction → cot")
    cot_atlid, attenuated = cot_from_aebd(
        track["extinction"], track["height"], track["quality_status"]
    )
    print(f"  cot range {cot_atlid.min():.3f}–{cot_atlid.max():.3f}, median {np.median(cot_atlid):.3f}")
    print(f"  attenuated profiles: {attenuated.sum()}/{n_prof} ({100*attenuated.mean():.1f}%)")

    print("Matching to SEVIRI")
    matches = match_track_to_seviri(
        track["lat"], track["lon"], track["time"],
        SEVIRI_ROOT, retrieval=RETRIEVAL,
    )
    matches["cot_atlid"] = cot_atlid
    matches["attenuated"] = attenuated
    matches["frame_id"] = track["frame_id"]

    n_valid = int(matches["valid_match"].sum())
    n_unique = matches.loc[matches["valid_match"], "sev_pixel_id"].nunique()
    print(f"  on_disk: {int(matches['on_disk'].sum())}/{n_prof}")
    print(f"  valid_match (≤ 7.5 min): {n_valid}/{n_prof}")
    print(f"  unique SEVIRI pixels hit: {n_unique}")
    print(f"  median |Δt|: {matches.loc[matches['valid_match'], 'time_diff_s'].median():.1f} s")
    print(f"  median dist: {matches.loc[matches['valid_match'], 'distance_km'].median():.2f} km")

    print("Sampling ORAC cot + cldmask at matches")
    matches = open_seviri_at_matches(
        matches, SEVIRI_ROOT, RETRIEVAL, ("cot", "cldmask")
    )
    matches = matches.rename(columns={"cot": "cot_orac", "cldmask": "cldmask_orac"})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    # Filter for the smoke-test scatter:
    # - both finite
    # - ORAC says cloudy (cot is undefined for clear-sky pixels)
    # - drop ORAC cot ≥ 200 (saturation rail, unconverged retrieval)
    # NOTE: `attenuated` flag is over-aggressive (fires on any QS=3 bin in column,
    # including high-altitude noise) — not used for filtering yet. To-do: detect
    # full attenuation via cumulative-τ + bottom-of-column QS=3.
    sample = matches[
        matches["valid_match"]
        & np.isfinite(matches["cot_atlid"])
        & np.isfinite(matches["cot_orac"])
        & (matches["cldmask_orac"] == 1)
        & (matches["cot_orac"] < 200)
        & (matches["cot_atlid"] > 0)
    ].copy()
    print(f"Sample-level scatter rows (cloudy, ORAC<200, ATLID>0): {len(sample)}")
    if len(sample) < 50:
        print("Too few rows — skipping figure.")
        return

    # Pixel-aggregate view: groupby sev_pixel_id, mean cot on both sides.
    pix = (
        sample.groupby("sev_pixel_id")
        .agg(
            cot_atlid=("cot_atlid", "mean"),
            cot_orac=("cot_orac", "mean"),
            n_atlid=("cot_atlid", "size"),
        )
        .reset_index()
    )
    print(f"Pixel-aggregate rows: {len(pix)} (median n_atlid={int(pix['n_atlid'].median())})")

    def _stats(d):
        b = (d["cot_orac"] - d["cot_atlid"]).mean()
        rmse = np.sqrt(((d["cot_orac"] - d["cot_atlid"]) ** 2).mean())
        r = np.corrcoef(d["cot_orac"], d["cot_atlid"])[0, 1]
        return b, rmse, r

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
    log_lim = (-1.5, 2.2)
    log_ticks = [-1, 0, 1, 2]
    log_labels = ["0.1", "1", "10", "100"]
    for ax, d, title in [
        (axes[0], sample, f"sample-level (N={len(sample)})"),
        (axes[1], pix, f"pixel-aggregate (N={len(pix)})"),
    ]:
        b, rmse, r = _stats(d)
        x = np.log10(d["cot_atlid"].clip(lower=0.05))
        y = np.log10(d["cot_orac"].clip(lower=0.05))
        hb = ax.hexbin(x, y, gridsize=50, mincnt=1, bins="log", cmap="viridis")
        fig.colorbar(hb, ax=ax, label="count (log)")
        ax.plot(log_lim, log_lim, "k--", lw=0.8)
        ax.set_xlim(log_lim); ax.set_ylim(log_lim)
        ax.set_xticks(log_ticks); ax.set_yticks(log_ticks)
        ax.set_xticklabels(log_labels); ax.set_yticklabels(log_labels)
        ax.set_aspect("equal")
        ax.set_xlabel("ATLID column τ₃₅₅")
        ax.set_ylabel("ORAC SEVIRI cot")
        ax.set_title(f"{title}\nbias={b:+.2f}  RMSE={rmse:.2f}  R={r:.2f}")

    fig.suptitle(
        f"cot smoke test — frame {track['frame_id']}, {RETRIEVAL}  "
        "(cloudy pixels, cot<200, non-attenuated)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
