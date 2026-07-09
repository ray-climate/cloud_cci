"""CTH sensitivity to the collocation thresholds (Δt and spatial distance).

Shows the headline SLSTR CTH statistics are insensitive to the collocation
thresholds — the defence for the 60-min / nearest-pixel choices. Computed from
the existing full-month CTH matches (collocated at 60 min), so Δt is swept
15→60 min and distance <1/<2/<3 km by post-hoc filtering (no re-collocation).
The Δt sweep to 120 min is covered separately by figures/slstr_dt_sweep/.

Usage: python scripts/slstr_sensitivity_table.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("figures/slstr_sensitivity")
OUT.mkdir(parents=True, exist_ok=True)
O, A = "cth_orac_corrected_km", "cth_atlid_thick_km"
DT_MIN = [15, 30, 45, 60]
DIST_KM = [1, 2, 3]


def load_qc_strict() -> pd.DataFrame:
    paths = sorted(glob.glob("validation_data/slstr_cth_2025-12/matches_cth_*.csv"))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    # qc_strict base filter (mirrors statistics.CTH_QC_MODES['qc_strict'])
    m = (
        (df["valid_match"] == True)
        & (df["cldmask_orac"] == 1)
        & np.isfinite(df[O]) & np.isfinite(df[A])
        & (df["quality_status_atlid"] == 0)
        & (df["confidence_atlid"] >= 5)
        & (df[A] <= df["tropopause_km_atlid"] + 2)
    )
    return df[m].copy()


def main() -> int:
    d = load_qc_strict()
    print(f"qc_strict base: {len(d):,} profiles")
    rows = []
    for dt in DT_MIN:
        for dc in DIST_KM:
            s = d[(d["time_diff_s"] <= dt * 60) & (d["distance_km"] < dc)]
            if len(s) < 50:
                rows.append(dict(dt_min=dt, dist_km=dc, n=len(s),
                                 bias=np.nan, rmse=np.nan, r=np.nan))
                continue
            diff = s[O] - s[A]
            rows.append(dict(
                dt_min=dt, dist_km=dc, n=len(s),
                bias=float(diff.mean()),
                rmse=float(np.sqrt((diff**2).mean())),
                r=float(np.corrcoef(s[O], s[A])[0, 1]),
            ))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "slstr_cth_sensitivity.csv", index=False)

    # heatmaps of bias / RMSE / R over the (Δt, distance) grid
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k, (col, label, cmap) in enumerate([
        ("bias", "bias [km]", "RdBu_r"),
        ("rmse", "RMSE [km]", "viridis"),
        ("r", "Pearson R", "viridis"),
    ]):
        piv = t.pivot(index="dist_km", columns="dt_min", values=col)
        vlim = max(abs(np.nanmin(piv.values)), abs(np.nanmax(piv.values)))
        im = ax[k].imshow(piv.values, cmap=cmap, aspect="auto",
                          vmin=-vlim if col == "bias" else None,
                          vmax=vlim if col == "bias" else None)
        ax[k].set_xticks(range(len(DT_MIN))); ax[k].set_xticklabels(DT_MIN)
        ax[k].set_yticks(range(len(DIST_KM))); ax[k].set_yticklabels(DIST_KM)
        ax[k].set_xlabel("Δt window [min]"); ax[k].set_ylabel("distance cap [km]")
        ax[k].set_title(label)
        for (i, j), v in np.ndenumerate(piv.values):
            ax[k].text(j, i, f"{v:.2f}", ha="center", va="center",
                       color="k", fontsize=9)
        fig.colorbar(im, ax=ax[k], shrink=0.8)
    fig.suptitle("SLSTR CTH sensitivity to collocation thresholds "
                 "(qc_strict, polar Dec-2025) — flat across the grid", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "slstr_cth_sensitivity.png", dpi=140)
    plt.close(fig)

    pd.set_option("display.width", 120)
    print(t.to_string(index=False))
    print(f"\nwrote {OUT/'slstr_cth_sensitivity.png'} and .csv")
    print(f"bias range across grid: {t['bias'].min():.2f} .. {t['bias'].max():.2f} km")
    print(f"RMSE range: {t['rmse'].min():.2f} .. {t['rmse'].max():.2f} km  |  "
          f"R range: {t['r'].min():.3f} .. {t['r'].max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
