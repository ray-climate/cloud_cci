"""Cloud liquid water-path validation: ORAC SLSTR `cwp` vs ACM-CAP LWP
(radar+lidar liquid_water_content integrated). Scatter + surface stratification,
matching the COT/CER surface split (open water / sea-ice / snow-ice-sheet).

CWP is not independent of COT — ORAC cwp ~ (5/9) rho tau r_e — so this tests
whether the tau saturation (report 3d/3e) propagates into the water-path product
against an independent (water-content) reference.
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
OUT = Path("figures/slstr_cwp_2025-12")
SEA_ICE_T = 271.35
SURF_ORDER = ["open water", "sea-ice", "snow / ice-sheet"]
SURF_COL = {"open water": "#1b4f72", "sea-ice": "#4a90d9", "snow / ice-sheet": "#b0b0b0"}


def main() -> int:
    d = pd.read_parquet(f"{SCRATCH}/cwp_pairs.parquet")
    m = ((d["quality_status_atlid"] == 0) & (d["liquid_only_atlid"] == True)
         & (d["cldmask_orac"] == 1) & (d["phase_orac"] == 1)
         & np.isfinite(d["cwp_orac"]) & np.isfinite(d["lwp_atlid"])
         & (d["cwp_orac"] > 0) & (d["lwp_atlid"] > 0))
    v = d[m].copy()
    # surface (ocean split only; land = ice sheet)
    ocean = v["lsflag_orac"] < 0.5
    # stemp not in cwp parquet; reconstruct sea-ice via lat? -> use lsflag only:
    surf = pd.Series(index=v.index, dtype=object)
    surf[ocean] = "ocean (sea-ice / water)"
    surf[~ocean] = "snow / ice-sheet"
    v["surf"] = surf

    diff = (v["cwp_orac"] - v["lwp_atlid"]).values
    print(f"LIQUID CWP N={len(v)}")
    print(f" ORAC cwp median {np.median(v['cwp_orac']):.0f}  LWP median {np.median(v['lwp_atlid']):.0f} g/m2")
    print(f" median bias {np.median(diff):+.0f}  mean {np.mean(diff):+.0f}  g/m2")
    rlog = np.corrcoef(np.log(v['cwp_orac']), np.log(v['lwp_atlid']))[0, 1]
    print(f" r_log {rlog:.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    # (a) scatter (log-log)
    ax[0].hexbin(v["lwp_atlid"], v["cwp_orac"], gridsize=40, bins="log",
                 xscale="log", yscale="log", cmap="viridis", mincnt=1)
    lim = [5, 5000]
    ax[0].plot(lim, lim, "w--", lw=1)
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("ACM-CAP LWP  (∫ liquid_water_content, g m⁻²)")
    ax[0].set_ylabel("ORAC SLSTR cwp  (g m⁻²)")
    ax[0].set_title(f"(a) liquid water path  N={len(v)}\nmedian bias "
                    f"{np.median(diff):+.0f} g m⁻²,  r_log {rlog:.2f}")

    # (b) median bias by surface (ocean vs ice sheet)
    labs, meds, ns = [], [], []
    for s in ["ocean (sea-ice / water)", "snow / ice-sheet"]:
        g = v[v["surf"] == s]
        if len(g) == 0:
            continue
        labs.append(s); meds.append(float((g["cwp_orac"] - g["lwp_atlid"]).median()))
        ns.append(len(g))
    cols = ["#2e6da4", "#b0b0b0"][:len(labs)]
    ax[1].bar(range(len(labs)), meds, color=cols, edgecolor="0.3")
    ymax = max(abs(np.array(meds)).max(), 1)
    for i, (mv, nn) in enumerate(zip(meds, ns)):
        ax[1].text(i, mv + (0.04 if mv >= 0 else -0.04) * ymax,
                   f"{mv:+.0f}\nN={nn//1000}k", ha="center",
                   va="bottom" if mv >= 0 else "top", fontsize=9)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(range(len(labs)))
    ax[1].set_xticklabels(labs, fontsize=9)
    ax[1].set_ylabel("median bias  ORAC − ACM-CAP  (g m⁻²)")
    ax[1].set_title("(b) water-path bias by surface")
    ax[1].grid(axis="y", alpha=0.3)
    lo, hi = min(0, min(meds)), max(0, max(meds))
    rng = max(hi - lo, 1)
    ax[1].set_ylim(lo - 0.35 * rng, hi + 0.35 * rng)

    fig.suptitle("ORAC SLSTR liquid water path vs EarthCARE ACM-CAP — Antarctic "
                 "daytime, phase-agree liquid, Dec-2025", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "cwp_validation.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
