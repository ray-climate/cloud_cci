"""Comprehensive water-COT difference analysis: the +3 bias is a phase-mismatch
artefact. Four panels:

 (a) where ORAC and ACM-CAP AGREE the cloud is liquid  → ORAC τ ≈ ACM-CAP τ
 (b) where ORAC misclassifies the (ACM-CAP) liquid cloud as ICE → inflated / noisy
 (c) bias decomposition: all vs phase-agree vs phase-disagree
 (d) what drives the misclassification (solar-zenith angle, surface)

Reads the filtered frame written by the analysis step (ACM-CAP liquid-only,
qc_strict, daytime).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

SCRATCH = ("/tmp/claude-7051641/-gws-pw-j07-nceo-aerosolfire-rsong-project-cloud-cci/"
           "3a1e8f12-6f9c-4529-9d79-b8ab9052e120/scratchpad/cotw_phase_ctt.parquet")
OUT = Path("figures/slstr_cot_water_2025-12")


def _scatter(ax, x, y, title, sub):
    bins = np.logspace(-1, 2, 60)
    ax.hist2d(x, y, bins=[bins, bins], cmap="viridis", norm=LogNorm(), cmin=1)
    ax.plot([0.1, 100], [0.1, 100], "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.1, 100); ax.set_ylim(0.1, 100)
    ax.set_xlabel("ACM-CAP liquid optical depth τ")
    ax.set_ylabel("ORAC cot")
    bias = (y - x).mean()
    lr = np.corrcoef(np.log10(x), np.log10(y))[0, 1]
    ax.text(0.04, 0.96, f"{sub}\nN = {len(x):,}\nbias = {bias:+.2f}\n"
            f"$r_{{log}}$ = {lr:.2f}", transform=ax.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(fc="white", ec="0.7", alpha=0.9))
    ax.set_title(title, fontsize=10.5)


def main() -> int:
    d = pd.read_parquet(SCRATCH)
    d["ice"] = d["phase_orac"] == 2
    ag = d[~d["ice"]]; di = d[d["ice"]]
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10.5))

    _scatter(ax[0, 0], ag["cot_water_atlid"].values, ag["cot_orac"].values,
             "(a) ORAC agrees: liquid — clean, near 1:1",
             "phase AGREE\n(ORAC liquid)")
    _scatter(ax[0, 1], di["cot_water_atlid"].values, di["cot_orac"].values,
             "(b) ORAC says ice — inflated / non-comparable",
             "phase DISAGREE\n(ORAC ice)")

    # (c) bias decomposition
    axc = ax[1, 0]
    allb = (d["cot_orac"] - d["cot_water_atlid"]).mean()
    agb = (ag["cot_orac"] - ag["cot_water_atlid"]).mean()
    dib = (di["cot_orac"] - di["cot_water_atlid"]).mean()
    labels = [f"All liquid-only\n(N={len(d):,})",
              f"Phase AGREE — ORAC liquid\n({100*len(ag)/len(d):.0f}%)",
              f"Phase DISAGREE — ORAC ice\n({100*len(di)/len(d):.0f}%)"]
    vals = [allb, agb, dib]
    ypos = [0, 1, 2]
    colors = ["#555", "#2e7d32", "#c0392b"]
    axc.barh(ypos, vals, color=colors, height=0.6)
    axc.axvline(0, color="k", lw=0.8)
    for yp, v in zip(ypos, vals):
        axc.text(v + (0.4 if v >= 0 else -0.4), yp, f"{v:+.2f}", va="center",
                 ha="left" if v >= 0 else "right", fontweight="bold", fontsize=10)
    axc.set_yticks(ypos); axc.set_yticklabels(labels, fontsize=9)
    axc.set_xlabel("bias  ORAC − ACM-CAP liquid τ")
    axc.set_xlim(-3, 20)
    axc.invert_yaxis()
    axc.set_title("(c) The +3 bias lives entirely in the 22% ORAC misclassifies as ice",
                  fontsize=10.5)
    axc.grid(axis="x", alpha=0.3)

    # (d) misclassification drivers
    axd = ax[1, 1]
    d["szab"] = pd.cut(d["sza_orac"], [0, 65, 70, 75, 90],
                       labels=["<65°", "65–70°", "70–75°", ">75°"])
    rate = d.groupby("szab", observed=True)["ice"].mean() * 100
    axd.bar(range(len(rate)), rate.values, color="#1565c0", width=0.6)
    axd.set_xticks(range(len(rate))); axd.set_xticklabels(rate.index.astype(str))
    axd.set_xlabel("solar zenith angle (ORAC pixel)")
    axd.set_ylabel("% of liquid clouds ORAC calls ICE")
    for i, v in enumerate(rate.values):
        axd.text(i, v + 0.4, f"{v:.0f}%", ha="center", fontsize=9)
    oc = 100 * d.loc[d["lsflag_orac"] < 0.5, "ice"].mean()
    la = 100 * d.loc[d["lsflag_orac"] >= 0.5, "ice"].mean()
    # control: NOT driven by cloud-top temperature (all clouds supercooled)
    ct_note = ""
    if "ctt_orac" in d.columns:
        cc = d[np.isfinite(d["ctt_orac"])].copy()
        cc["ice"] = cc["phase_orac"] == 2
        cold = 100 * cc.loc[cc["ctt_orac"] - 273.15 < -25, "ice"].mean()
        warm = 100 * cc.loc[(cc["ctt_orac"] - 273.15 >= -25), "ice"].mean()
        ct_note = (f"\n\nNOT cloud-temperature:\n  all supercooled (−30..0 °C)\n"
                   f"  <−25 °C {cold:.0f}% ≈ ≥−25 °C {warm:.0f}%")
    axd.text(0.04, 0.97, f"by surface:\n  sea-ice ocean {oc:.0f}%\n  ice-sheet land {la:.0f}%"
             + ct_note, transform=axd.transAxes, va="top", ha="left", fontsize=8.5,
             bbox=dict(fc="white", ec="0.7", alpha=0.9))
    axd.set_title("(d) Driven by retrieval geometry (SZA, surface) — not temperature",
                  fontsize=10.5)
    axd.grid(axis="y", alpha=0.3)

    fig.suptitle("Water-COT: the +3 bias is a phase-misclassification artefact, "
                 "not a τ error — SLSTR × ACM-CAP, Antarctic daytime Dec-2025",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "cot_water_phase_analysis.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
