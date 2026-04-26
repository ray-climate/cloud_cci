"""End-to-end cot validation smoke test on one frame.

Reads ATL_EBD_2A frame 09541B (2026-02-01), integrates particle extinction
at 355 nm to a column τ per profile, matches each profile to its nearest
SEVIRI ORAC pixel in the closest 15-min slot, and writes:

    validation_data/matches_cot_2026-02-01_09541B.csv
    figures/validation/cot_smoke_scatter.png        # 2-panel sample vs pixel
    figures/validation/cot_smoke_diagnostic.png     # 2x2 colour-coded panels

Run::

    PYTHONPATH=. python scripts/validate_cot_smoke.py
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

# ORAC retrieval upper rail. Values at/above this are unconverged (LUT edge);
# tracked as a separate stratum, not dropped silently.
ORAC_COT_SATURATION = 100.0

OUT_CSV = Path("validation_data/matches_cot_2026-02-01_09541B.csv")
OUT_SCATTER = Path("figures/validation/cot_smoke_scatter.png")
OUT_DIAG = Path("figures/validation/cot_smoke_diagnostic.png")

LOG_LIM = (-1.5, 2.2)
LOG_TICKS = [-1, 0, 1, 2]
LOG_LABELS = ["0.1", "1", "10", "100"]


def _stats(d):
    b = (d["cot_orac"] - d["cot_atlid"]).mean()
    rmse = np.sqrt(((d["cot_orac"] - d["cot_atlid"]) ** 2).mean())
    r = np.corrcoef(d["cot_orac"], d["cot_atlid"])[0, 1]
    return b, rmse, r


def _log_axes(ax):
    ax.plot(LOG_LIM, LOG_LIM, "k--", lw=0.8)
    ax.set_xlim(LOG_LIM); ax.set_ylim(LOG_LIM)
    ax.set_xticks(LOG_TICKS); ax.set_yticks(LOG_TICKS)
    ax.set_xticklabels(LOG_LABELS); ax.set_yticklabels(LOG_LABELS)
    ax.set_aspect("equal")
    ax.set_xlabel("ATLID column τ₃₅₅")
    ax.set_ylabel("ORAC SEVIRI cot")


def main() -> None:
    print(f"Reading {AEBD_FILE.name}")
    track = read_aebd_track(AEBD_FILE)
    n_prof = len(track["lat"])
    print(f"  {n_prof} ATLID profiles, {track['extinction'].shape[1]} bins")
    print(f"  lat {track['lat'].min():.2f}–{track['lat'].max():.2f},"
          f"  lon {track['lon'].min():.2f}–{track['lon'].max():.2f}")

    print("Integrating extinction → cot")
    cot_atlid, attenuated = cot_from_aebd(
        track["extinction"], track["height"], track["quality_status"]
    )
    print(f"  cot range {cot_atlid.min():.3f}–{cot_atlid.max():.3f}, "
          f"median {np.median(cot_atlid):.3f}")
    print(f"  attenuated profiles: {attenuated.sum()}/{n_prof} "
          f"({100*attenuated.mean():.1f}%)")

    print("Matching to SEVIRI")
    matches = match_track_to_seviri(
        track["lat"], track["lon"], track["time"],
        SEVIRI_ROOT, retrieval=RETRIEVAL,
    )
    matches["cot_atlid"] = cot_atlid
    matches["attenuated"] = attenuated
    matches["frame_id"] = track["frame_id"]
    print(f"  on_disk: {int(matches['on_disk'].sum())}/{n_prof}")
    print(f"  valid_match: {int(matches['valid_match'].sum())}/{n_prof}")
    print(f"  unique SEVIRI pixels hit: "
          f"{matches.loc[matches['valid_match'], 'sev_pixel_id'].nunique()}")

    print("Sampling ORAC cot + cldmask at matches")
    matches = open_seviri_at_matches(
        matches, SEVIRI_ROOT, RETRIEVAL, ("cot", "cldmask")
    )
    matches = matches.rename(columns={"cot": "cot_orac", "cldmask": "cldmask_orac"})
    matches["cot_orac_saturated"] = matches["cot_orac"] >= ORAC_COT_SATURATION
    print(f"  ORAC saturated (cot >= {ORAC_COT_SATURATION:.0f}): "
          f"{int(matches['cot_orac_saturated'].sum())}/{n_prof}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    # ------------------------------------------------------------------
    # Filter: cloudy, both finite, ATLID > 0. Keep saturated rows in the
    # CSV but exclude from the primary scatter (they'd be on the cot=100+
    # rail and meaningless).
    # ------------------------------------------------------------------
    base = matches[
        matches["valid_match"]
        & np.isfinite(matches["cot_atlid"])
        & np.isfinite(matches["cot_orac"])
        & (matches["cldmask_orac"] == 1)
        & (matches["cot_atlid"] > 0)
        & (~matches["cot_orac_saturated"])
    ].copy()
    print(f"Sample-level rows (cloudy, ATLID>0, ORAC unsaturated): {len(base)}")
    if len(base) < 50:
        print("Too few rows — skipping figures.")
        return

    # Pixel-aggregate
    pix = (
        base.groupby("sev_pixel_id")
        .agg(
            cot_atlid=("cot_atlid", "mean"),
            cot_orac=("cot_orac", "mean"),
            ec_lat=("ec_lat", "mean"),
            distance_km=("distance_km", "mean"),
            time_diff_s=("time_diff_s", "mean"),
            attenuated=("attenuated", "any"),
            n_atlid=("cot_atlid", "size"),
        )
        .reset_index()
    )
    print(f"Pixel-aggregate rows: {len(pix)} "
          f"(median n_atlid={int(pix['n_atlid'].median())})")

    # ------------------------------------------------------------------
    # Figure 1 — sample-level vs pixel-aggregate (the headline scatter).
    # ------------------------------------------------------------------
    OUT_SCATTER.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
    for ax, d, title in [
        (axes[0], base, f"sample-level (N={len(base)})"),
        (axes[1], pix, f"pixel-aggregate (N={len(pix)})"),
    ]:
        b, rmse, r = _stats(d)
        x = np.log10(d["cot_atlid"].clip(lower=0.05))
        y = np.log10(d["cot_orac"].clip(lower=0.05))
        hb = ax.hexbin(x, y, gridsize=50, mincnt=1, bins="log", cmap="viridis")
        fig.colorbar(hb, ax=ax, label="count (log)")
        _log_axes(ax)
        ax.set_title(f"{title}\nbias={b:+.2f}  RMSE={rmse:.2f}  R={r:.2f}")
    fig.suptitle(
        f"cot smoke test — frame {track['frame_id']}, {RETRIEVAL}  "
        "(cloudy, ATLID>0, ORAC unsaturated)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_SCATTER, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_SCATTER}")

    # ------------------------------------------------------------------
    # Figure 2 — diagnostic 2x2: scatter coloured by lat / distance /
    # time-diff / attenuated, all sample-level. Helps interpret the
    # "horizontal tail" (ATLID-thin / ORAC-thick).
    # ------------------------------------------------------------------
    OUT_DIAG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    x = np.log10(base["cot_atlid"].clip(lower=0.05))
    y = np.log10(base["cot_orac"].clip(lower=0.05))

    # (a) lat
    ax = axes[0, 0]
    sc = ax.scatter(x, y, c=base["ec_lat"], cmap="viridis", s=4, alpha=0.6,
                    vmin=20, vmax=70)
    fig.colorbar(sc, ax=ax, label="latitude [deg]")
    _log_axes(ax)
    ax.set_title("(a) coloured by latitude")

    # (b) distance
    ax = axes[0, 1]
    sc = ax.scatter(x, y, c=base["distance_km"], cmap="plasma", s=4, alpha=0.6,
                    vmin=0, vmax=6)
    fig.colorbar(sc, ax=ax, label="dist to SEVIRI pixel [km]")
    _log_axes(ax)
    ax.set_title("(b) coloured by match distance")

    # (c) time diff
    ax = axes[1, 0]
    sc = ax.scatter(x, y, c=base["time_diff_s"], cmap="cividis", s=4, alpha=0.6,
                    vmin=0, vmax=450)
    fig.colorbar(sc, ax=ax, label="|Δt| [s]")
    _log_axes(ax)
    ax.set_title("(c) coloured by time offset")

    # (d) attenuated
    ax = axes[1, 1]
    not_att = base[~base["attenuated"]]
    att = base[base["attenuated"]]
    ax.scatter(np.log10(not_att["cot_atlid"].clip(lower=0.05)),
               np.log10(not_att["cot_orac"].clip(lower=0.05)),
               c="0.7", s=4, alpha=0.5, label=f"normal (N={len(not_att)})")
    ax.scatter(np.log10(att["cot_atlid"].clip(lower=0.05)),
               np.log10(att["cot_orac"].clip(lower=0.05)),
               c="tab:red", s=8, alpha=0.7,
               label=f"attenuated (N={len(att)})")
    _log_axes(ax)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("(d) attenuated profiles (τ is lower bound)")

    fig.suptitle(
        f"cot smoke diagnostic — frame {track['frame_id']}, {RETRIEVAL}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIAG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIAG}")


if __name__ == "__main__":
    main()
