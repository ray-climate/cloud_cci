"""Cloud-mask and cloud-phase validation of ORAC SLSTR against EarthCARE A-TC
(ATLID Target Classification) — the two-way contingency (POD_ice included) that the
liquid-centric ACM-CAP flags could not provide.

Cloud mask : ORAC cldmask vs A-TC any-cloud-in-column (usable columns only).
Phase      : where both see cloud, ORAC phase (1 liq / 2 ice) vs A-TC cloud-TOP
             phase (top_class 1|2 -> liquid, 3 -> ice).
Stratified by sun-zenith and surface (sea-ice/ice-sheet/open water).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRATCH = ("/tmp/claude-7051641/-gws-pw-j07-nceo-aerosolfire-rsong-project-cloud-cci/"
           "3a1e8f12-6f9c-4529-9d79-b8ab9052e120/scratchpad")
OUT = Path("figures/slstr_phase_2025-12")


def contingency_mask(d: pd.DataFrame) -> dict:
    # usable A-TC columns only (exclude fully attenuated / missing)
    u = d[d["atc_usable"]].copy()
    orac_cloud = u["cldmask_orac"] == 1
    atc_cloud = u["atc_cloud"]
    hit = int((orac_cloud & atc_cloud).sum())
    miss = int((~orac_cloud & atc_cloud).sum())
    fa = int((orac_cloud & ~atc_cloud).sum())
    cn = int((~orac_cloud & ~atc_cloud).sum())
    tot = hit + miss + fa + cn
    return dict(N=tot, hit=hit, miss=miss, false_alarm=fa, correct_neg=cn,
                POD=hit / (hit + miss) if hit + miss else np.nan,
                FAR=fa / (hit + fa) if hit + fa else np.nan,
                accuracy=(hit + cn) / tot if tot else np.nan,
                bias_ratio=(hit + fa) / (hit + miss) if hit + miss else np.nan)


def _atc_phase(top_class: pd.Series) -> pd.Series:
    return np.where(top_class.isin([1, 2]), "liquid",
                    np.where(top_class == 3, "ice", "none"))


def contingency_phase(d: pd.DataFrame) -> tuple:
    # both detect cloud; ORAC has a definite phase
    both = d[(d["cldmask_orac"] == 1) & d["phase_orac"].isin([1, 2])
             & d["atc_cloud"]].copy()
    both["atc_phase"] = _atc_phase(both["atc_top_class"])
    both["orac_phase"] = np.where(both["phase_orac"] == 1, "liquid", "ice")
    ct = pd.crosstab(both["orac_phase"], both["atc_phase"])
    for c in ["liquid", "ice"]:
        if c not in ct.columns:
            ct[c] = 0
    liq = both[both["atc_phase"] == "liquid"]
    ice = both[both["atc_phase"] == "ice"]
    pod_liq = (liq["orac_phase"] == "liquid").mean() if len(liq) else np.nan
    pod_ice = (ice["orac_phase"] == "ice").mean() if len(ice) else np.nan
    acc = (both["orac_phase"] == both["atc_phase"]).mean()
    return both, ct, pod_liq, pod_ice, acc


def main() -> int:
    d = pd.read_parquet(f"{SCRATCH}/atc_phase.parquet")
    d = d[d["atc_top_class"] >= 0].copy()   # augmented rows only
    print(f"A-TC-augmented matches: {len(d)}")

    m = contingency_mask(d)
    print("\n===== CLOUD MASK  (ORAC vs A-TC, usable columns) =====")
    print(f"N={m['N']}  hit={m['hit']} miss={m['miss']} false_alarm={m['false_alarm']} "
          f"correct_neg={m['correct_neg']}")
    print(f"POD(cloud)={m['POD']:.3f}  FAR={m['FAR']:.3f}  accuracy={m['accuracy']:.3f}  "
          f"bias_ratio={m['bias_ratio']:.2f}")

    both, ct, pod_liq, pod_ice, acc = contingency_phase(d)
    print("\n===== CLOUD PHASE  (both cloudy; ORAC vs A-TC cloud-top) =====")
    print(f"N(both cloudy, ORAC has phase) = {len(both)}")
    print(ct.to_string())
    print(f"\nPOD_liquid = {pod_liq*100:.1f}%   POD_ice = {pod_ice*100:.1f}%   "
          f"overall phase accuracy = {acc*100:.1f}%")

    # supercooled vs warm liquid detection (A-TC distinguishes them)
    liqtop = both[both["atc_top_class"].isin([1, 2])]
    for lc, name in [(1, "warm liquid"), (2, "supercooled")]:
        s = liqtop[liqtop["atc_top_class"] == lc]
        if len(s):
            print(f"  A-TC {name:12s}: ORAC calls liquid {100*(s.orac_phase=='liquid').mean():.1f}%  (N={len(s)})")

    # by SZA
    print("\n  phase accuracy & POD_ice by SZA:")
    both["szb"] = pd.cut(both["sza_orac"], [60, 70, 75, 80, 85, 92])
    for b, g in both.groupby("szb", observed=True):
        ice = g[g.atc_phase == "ice"]
        pi = (ice.orac_phase == "ice").mean()*100 if len(ice) else np.nan
        print(f"    SZA {str(b):>12}: acc {100*(g.orac_phase==g.atc_phase).mean():4.1f}%  "
              f"POD_ice {pi:4.1f}%  (N={len(g)})")

    # by surface type (full cryosphere split: sea-ice / open water / ice-sheet)
    print("\n  phase skill by surface type (stemp split at 271.35 K):")
    ocean = both["lsflag_orac"] < 0.5
    st = both["stemp_orac"]
    surf = pd.Series(index=both.index, dtype=object)
    surf[ocean & (st < 271.35)] = "sea-ice"
    surf[ocean & (st >= 271.35)] = "open water"
    surf[~ocean] = "snow / ice-sheet"
    both["surf"] = surf
    surf_order = ["open water", "sea-ice", "snow / ice-sheet"]
    surf_pod = {}
    for s in surf_order:
        g = both[both["surf"] == s]
        if not len(g):
            continue
        liq = g[g.atc_phase == "liquid"]; ice = g[g.atc_phase == "ice"]
        pl = (liq.orac_phase == "liquid").mean()*100 if len(liq) else np.nan
        pi = (ice.orac_phase == "ice").mean()*100 if len(ice) else np.nan
        icefrac = 100*(g.atc_phase == "ice").mean()   # cloud composition per surface
        surf_pod[s] = (pl, pi, len(g))
        print(f"    {s:18s}: POD_liq {pl:4.1f}%  POD_ice {pi:4.1f}%  "
              f"acc {100*(g.orac_phase==g.atc_phase).mean():4.1f}%  "
              f"(N={len(g):>6}, A-TC ice-frac {icefrac:.0f}%)")
    nan_s = both["surf"].isna().sum()
    if nan_s:
        print(f"    (stemp missing on {nan_s} rows -> unclassified)")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5))
    # (a) phase confusion matrix, column-normalised so the diagonal == POD
    cm = ct.reindex(index=["liquid", "ice"], columns=["liquid", "ice"]).fillna(0).values
    cmn = 100 * cm / cm.sum(axis=0, keepdims=True)   # per-truth-column
    ax[0].imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["liquid", "ice"])
    ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["liquid", "ice"])
    ax[0].set_xlabel("EarthCARE A-TC cloud-top phase (truth)")
    ax[0].set_ylabel("ORAC SLSTR phase")
    for i in range(2):
        for j in range(2):
            tag = "  ← POD" if i == j else ""
            ax[0].text(j, i, f"{cmn[i,j]:.0f}%{tag}\n(N={int(cm[i,j])})", ha="center",
                       va="center", color="white" if cmn[i, j] > 50 else "black", fontsize=10)
    ax[0].set_title(f"(a) Phase confusion (column = truth, normalised)\nPOD_liq "
                    f"{pod_liq*100:.0f}%  POD_ice {pod_ice*100:.0f}%  acc {acc*100:.0f}%")

    # (b) cloud-mask contingency bars
    labels = ["hit\n(both cloud)", "miss\n(ORAC clear,\nATLID cloud)",
              "false alarm\n(ORAC cloud,\nATLID clear)", "correct\nnegative"]
    vals = [m["hit"], m["miss"], m["false_alarm"], m["correct_neg"]]
    cols = ["#1b7837", "#d9a441", "#c0392b", "#95a5a6"]
    ax[1].bar(range(4), vals, color=cols, edgecolor="0.3")
    for i, v in enumerate(vals):
        ax[1].text(i, v, f"{v}", ha="center", va="bottom", fontsize=9)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_title(f"(b) Cloud mask vs A-TC\nPOD {m['POD']:.2f}  FAR {m['FAR']:.2f}  "
                    f"acc {m['accuracy']:.2f}")
    ax[1].set_ylabel("pixels")

    # (c) POD_liquid / POD_ice by surface type — the answer to "does phase skill
    #     depend on surface?"  POD_ice is flat; POD_liquid dips over sea-ice.
    slist = [s for s in surf_order if s in surf_pod]
    x = np.arange(len(slist)); w = 0.38
    pl_v = [surf_pod[s][0] for s in slist]
    pi_v = [surf_pod[s][1] for s in slist]
    ax[2].bar(x - w/2, pl_v, w, label="POD_liquid", color="#c0392b", edgecolor="0.3")
    ax[2].bar(x + w/2, pi_v, w, label="POD_ice", color="#4a90d9", edgecolor="0.3")
    for xi, (pl, pi, nn) in zip(x, [surf_pod[s] for s in slist]):
        ax[2].text(xi - w/2, pl + 1, f"{pl:.0f}", ha="center", fontsize=9)
        ax[2].text(xi + w/2, pi + 1, f"{pi:.0f}", ha="center", fontsize=9)
    ax[2].axhline(pod_ice*100, color="#4a90d9", ls=":", lw=1)
    ax[2].axhline(pod_liq*100, color="#c0392b", ls=":", lw=1)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([f"{s}\n(N={surf_pod[s][2]//1000}k)" for s in slist], fontsize=8)
    ax[2].set_ylim(0, 100); ax[2].set_ylabel("POD (%)")
    ax[2].legend(fontsize=9, loc="lower right")
    ax[2].set_title("(c) Phase skill by surface type\nPOD_ice flat; POD_liq dips over sea-ice")
    ax[2].grid(axis="y", alpha=0.3)

    fig.suptitle("ORAC SLSTR cloud mask & phase vs EarthCARE A-TC — Antarctic "
                 "daytime, Dec-2025", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "phase_contingency.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("\nwrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
