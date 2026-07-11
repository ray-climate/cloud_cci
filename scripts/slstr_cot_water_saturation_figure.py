"""Attribute the water-COT median underestimate: ORAC's passive liquid optical
depth SATURATES (~7-8) over polar bright surfaces at high sun-zenith, while the
ACM-CAP synergy (radar-aided) retrieves the true, higher tau. Two opposing
effects — ORAC over-retrieves thin cloud, saturates on thick cloud.

Phase-agree liquid subset (ORAC & ACM-CAP both liquid), qc_strict, daytime.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("figures/slstr_cot_water_2025-12")


def main() -> int:
    cols = ["valid_match", "quality_status_atlid", "cot_orac", "cot_water_atlid",
            "phase_orac", "liquid_only_atlid", "cldmask_orac"]
    paths = sorted(glob.glob("validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv"))
    d = pd.concat([pd.read_csv(p, usecols=lambda c: c in cols) for p in paths],
                  ignore_index=True)
    m = ((d["valid_match"] == True) & (d["quality_status_atlid"] == 0)
         & (d["liquid_only_atlid"] == True) & (d["cldmask_orac"] == 1)
         & (d["phase_orac"] == 1) & np.isfinite(d["cot_orac"])
         & np.isfinite(d["cot_water_atlid"]) & (d["cot_orac"] > 0) & (d["cot_water_atlid"] > 0))
    d = d[m].copy()

    # bin by ACM-CAP tau (log-spaced), median ORAC tau per bin
    edges = np.array([0.3, 1, 2, 3, 5, 7, 10, 15, 22, 32, 50])
    d["b"] = pd.cut(d["cot_water_atlid"], edges)
    g = d.groupby("b", observed=True).agg(
        acmcap=("cot_water_atlid", "median"), orac=("cot_orac", "median"),
        bias=("cot_orac", lambda s: np.nan), n=("cot_orac", "size"))
    g["bias"] = d.groupby("b", observed=True).apply(
        lambda x: (x["cot_orac"] - x["cot_water_atlid"]).median())

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # (a) ORAC median tau vs ACM-CAP tau — the saturation ceiling
    ax[0].plot([0.3, 50], [0.3, 50], "k--", lw=1, label="1:1 (perfect)")
    ax[0].plot(g["acmcap"], g["orac"], "o-", color="#c0392b", lw=2, ms=6,
               label="ORAC liquid τ (median)")
    ax[0].axhline(8, color="0.5", ls=":", lw=1)
    ax[0].text(0.4, 8.4, "ORAC saturates ≈ 8", fontsize=9, color="0.4")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlim(0.3, 50); ax[0].set_ylim(0.3, 50)
    ax[0].set_xlabel("ACM-CAP liquid optical depth τ (synergy, radar-aided)")
    ax[0].set_ylabel("ORAC liquid optical depth τ (passive)")
    ax[0].set_title("(a) ORAC passive τ saturates while synergy τ climbs")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3, which="both")

    # (b) median bias vs ACM-CAP tau — the crossover
    xc = np.arange(len(g))
    colors = ["#c0392b" if b < 0 else "#1565c0" for b in g["bias"]]
    ax[1].bar(xc, g["bias"], color=colors, edgecolor="0.3")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(xc)
    ax[1].set_xticklabels([f"{v:.0f}" for v in g["acmcap"]], rotation=0)
    ax[1].set_xlabel("ACM-CAP liquid τ (bin median)")
    ax[1].set_ylabel("median bias  ORAC − ACM-CAP")
    ax[1].set_title("(b) Over-retrieves thin cloud, saturates on thick")
    ax[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Why water-COT under-reads on the median: ORAC passive liquid τ "
                 "saturates (~8) over polar bright surfaces\n"
                 "phase-agree liquid, Antarctic daytime Dec-2025", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "cot_water_saturation.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)
    print(g[["acmcap", "orac", "bias", "n"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
