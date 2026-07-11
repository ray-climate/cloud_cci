"""Surface-type stratification of SLSTR ORAC solar retrievals vs EarthCARE.

The v5.1 "new snowice" retrieval is meant to improve cloud detection/retrieval
over bright cryospheric surfaces. This splits the validation by the ORAC surface
class at each matched pixel:

  sea-ice        : ocean (lsflag=0) & stemp < 271.35 K  (seawater freezing point)
  open water     : ocean (lsflag=0) & stemp >= 271.35 K
  snow / ice-sheet: land (lsflag=1) & stemp <  273.15 K
  snow-free land : land (lsflag=1) & stemp >= 273.15 K

Produces per-surface median-bias / RMSE / R for water-COT, CER (ACM-CAP) and
ice-COT (A-EBD), a grouped bar figure, and printed tables for the report.
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
OUT = Path("figures/slstr_surface_2025-12")

SEA_ICE_T = 271.35   # K, seawater freezing

SURF_ORDER = ["open water", "sea-ice", "snow / ice-sheet"]
SURF_COL = {"open water": "#1b4f72", "sea-ice": "#4a90d9",
            "snow / ice-sheet": "#b0b0b0"}


def classify(d: pd.DataFrame) -> pd.Series:
    # In the Antarctic-summer daytime sample 100% of pixels are sub-freezing and
    # all land is ice sheet, so the only cryosphere contrast is within the ocean:
    # sea-ice (frozen) vs open water (>271.35 K). Land is uniformly snow/ice-sheet.
    ocean = d["lsflag_orac"] < 0.5
    st = d["stemp_orac"]
    s = pd.Series(index=d.index, dtype=object)
    s[ocean & (st < SEA_ICE_T)] = "sea-ice"
    s[ocean & (st >= SEA_ICE_T)] = "open water"
    s[~ocean] = "snow / ice-sheet"
    return s


def _stats(diff: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    n = len(diff)
    out = dict(n=n, median_bias=np.nan, mean_bias=np.nan, rmse=np.nan, r=np.nan)
    if n < 1:
        return out
    out["median_bias"] = float(np.median(diff))
    out["mean_bias"] = float(np.mean(diff))
    out["rmse"] = float(np.sqrt(np.mean(diff ** 2)))
    if n >= 3 and np.std(x) > 0 and np.std(y) > 0:
        out["r"] = float(np.corrcoef(x, y)[0, 1])
    return out


def surface_table(d: pd.DataFrame, xcol: str, ycol: str, label: str) -> pd.DataFrame:
    d = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol]) & np.isfinite(d["stemp_orac"])
          & (d[xcol] > 0) & (d[ycol] > 0)].copy()
    d["surf"] = classify(d)
    rows = []
    for s in SURF_ORDER + ["ALL"]:
        g = d if s == "ALL" else d[d["surf"] == s]
        if len(g) == 0:
            continue
        diff = (g[xcol] - g[ycol]).values
        st = _stats(diff, g[ycol].values, g[xcol].values)
        rows.append(dict(surface=s, **st))
    t = pd.DataFrame(rows)
    print(f"\n===== {label}  (ORAC {xcol}  vs  {ycol}) =====")
    print(t.to_string(index=False,
          formatters={c: (lambda v: f"{v:7.2f}") for c in
                      ["median_bias", "mean_bias", "rmse", "r"]}))
    return t


def main() -> int:
    syn = pd.read_parquet(f"{SCRATCH}/surface_synergy.parquet")
    cot = pd.read_parquet(f"{SCRATCH}/surface_cot.parquet")

    # phase-agree liquid, qc-strict, daytime, cloudy
    lw = syn[(syn["quality_status_atlid"] == 0) & (syn["liquid_only_atlid"] == True)
             & (syn["cldmask_orac"] == 1) & (syn["phase_orac"] == 1)].copy()
    t_cot = surface_table(lw, "cot_orac", "cot_water_atlid", "WATER-COT by surface")
    t_cer = surface_table(lw, "cer_orac", "cer_water_atlid", "CER (water) by surface")

    ice = cot[(cot["cldmask_orac"] == 1) & (cot["phase_orac"] == 2)].copy()
    t_ice = surface_table(ice, "cot_orac", "cot_atlid", "ICE-COT by surface")

    # ---- figure: median bias per surface, three panels ----
    panels = [("water-COT  (ORAC − ACM-CAP)", t_cot),
              ("CER µm  (ORAC − ACM-CAP)", t_cer),
              ("ice-COT  (ORAC − A-EBD)", t_ice)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, t) in zip(axes, panels):
        tt = t[t["surface"] != "ALL"]
        surf = list(tt["surface"])
        y = tt["median_bias"].values
        cols = [SURF_COL[s] for s in surf]
        ax.bar(range(len(surf)), y, color=cols, edgecolor="0.3")
        ymin, ymax = min(0.0, float(y.min())), max(0.0, float(y.max()))
        rng = max(ymax - ymin, 1.0)
        ax.set_ylim(ymin - 0.42 * rng, ymax + 0.42 * rng)
        pad = 0.03 * rng
        for i, (v, nn, rr) in enumerate(zip(y, tt["n"], tt["r"])):
            ax.text(i, v + (pad if v >= 0 else -pad),
                    f"{v:.1f}\nr={rr:.2f}\nN={int(nn/1000)}k", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(surf)))
        ax.set_xticklabels(surf, rotation=18, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11, pad=26)
        ax.set_ylabel("median bias")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("SLSTR ORAC solar-retrieval skill by surface type — Antarctic "
                 "daytime, Dec-2025\nocean split into sea-ice / open water at "
                 "271.35 K (ORAC stemp); all polar land is snow / ice-sheet", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "surface_type_bias.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("\nwrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
