"""Quick-look checks for a SLSTR-pixel × ACM-CAP collocation CSV
(``validation slstr-pixel-collocate``): where/when the matches are and whether
the ORAC and ACM-CAP columns are broadly comparable. A sanity figure for the
people receiving the CSV, not a validation result.

    python scripts/slstr_pixel_quicklook.py <pixels.csv> <out.png> [max_dt_min]
"""
from __future__ import annotations

import sys

import cartopy.crs as ccrs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8f93"


def _loglog(ax, x, y, lim, xlabel, ylabel, title):
    ax.scatter(x, y, s=3, color=BLUE, alpha=0.25, linewidths=0)
    ax.plot(lim, lim, color=GREY, lw=1)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel=xlabel, ylabel=ylabel, title=f"{title}  (n = {len(x):,})")


def main(csv: str, out: str, max_dt_min: float = 10.0) -> int:
    d = pd.read_csv(csv)
    dt_min = d["dt_s"] / 60.0
    near = dt_min.abs() <= max_dt_min
    day = d["orac_illum"] == 1
    good = d["acmcap_quality_status"] == 0

    fig = plt.figure(figsize=(15, 9.5))
    for i, (proj, extent, name) in enumerate((
        (ccrs.NorthPolarStereo(), [-180, 180, 65, 90], "Arctic"),
        (ccrs.SouthPolarStereo(), [-180, 180, -90, -65], "Antarctic"),
    )):
        ax = fig.add_subplot(2, 4, i + 1, projection=proj)
        ax.set_extent(extent, ccrs.PlateCarree())
        ax.coastlines(lw=0.5, color=GREY)
        ax.gridlines(lw=0.3, color=GREY, alpha=0.5)
        for mask, col, lab in ((~near, GREY, f"> {max_dt_min:g} min"),
                               (near, ORANGE, f"≤ {max_dt_min:g} min")):
            ax.scatter(d.loc[mask, "slstr_lon"], d.loc[mask, "slstr_lat"], s=1.5,
                       color=col, transform=ccrs.PlateCarree(), label=lab, linewidths=0)
        ax.set_title(f"{name}: matched SLSTR pixels")
        if i == 0:
            ax.legend(loc="lower left", markerscale=6, fontsize=8, frameon=False)

    ax = fig.add_subplot(2, 4, 3)
    bins = np.arange(-60, 61, 5)
    ax.hist(dt_min, bins=bins, color=GREY, label="all")
    ax.hist(dt_min[day], bins=bins, color=BLUE, label="ORAC daytime")
    ax.axvspan(-max_dt_min, max_dt_min, color=ORANGE, alpha=0.15, lw=0)
    ax.set(xlabel="EarthCARE − SLSTR time (min)", ylabel="pixels",
           title=f"Time difference  (n = {len(d):,})")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(2, 4, 4)
    ax.hist(d["distance_km"], bins=np.arange(0, 1.55, 0.05), color=BLUE)
    ax.set(xlabel="nearest profile → pixel centre (km)", ylabel="pixels",
           title="Match distance")

    # A-CTH heights have a spurious stratospheric tail; cap as in the CTH reports.
    cap = d["acth_tropopause_km"] + 2
    m = near & (d["acth_quality_status"] == 0) & d["orac_cth"].notna()
    for j, (col, lab) in enumerate((("acth_cth_thick_km", "A-CTH thick-cloud top"),
                                    ("acth_cth_km", "A-CTH lidar top"))):
        ax = fig.add_subplot(2, 4, 5 + j)
        mm = m & (d[col] <= cap)
        ax.scatter(d.loc[mm, col], d.loc[mm, "orac_cth"], s=3, color=BLUE,
                   alpha=0.25, linewidths=0)
        ax.plot([0, 13], [0, 13], color=GREY, lw=1)
        bias = (d.loc[mm, "orac_cth"] - d.loc[mm, col]).median()
        ax.set(xlim=(0, 13), ylim=(0, 13), xlabel=f"{lab} (km)", ylabel="ORAC cth (km)",
               title=f"CTH, median diff {bias:+.2f} km  (n = {mm.sum():,})")

    m = (near & good & day & (d["orac_cot"] > 0) & (d["acmcap_cot_total"] > 0))
    _loglog(fig.add_subplot(2, 4, 7), d.loc[m, "acmcap_cot_total"], d.loc[m, "orac_cot"],
            (0.05, 300), "ACM-CAP total COT", "ORAC cot", "COT, daytime")

    # Phase: ORAC phase (0 clear, 1 liquid, 2 ice) vs ACM-CAP column phase.
    ax = fig.add_subplot(2, 4, 8)
    m = near & good & d["orac_phase"].notna() & d["acmcap_phase_col"].notna()
    table = pd.crosstab(d.loc[m, "acmcap_phase_col"].astype(int),
                        d.loc[m, "orac_phase"].astype(int)).reindex(
        index=[0, 1, 2, 3], columns=[0, 1, 2], fill_value=0)
    frac = table.div(table.sum(axis=1).replace(0, np.nan), axis=0)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for r in range(4):
        for c in range(3):
            ax.text(c, r, f"{table.iat[r, c]:,}\n{frac.iat[r, c]:.0%}", ha="center",
                    va="center", fontsize=9,
                    color="white" if frac.iat[r, c] > 0.55 else "black")
    ax.set(xticks=range(3), yticks=range(4), xticklabels=["clear", "liquid", "ice"],
           yticklabels=["clear", "liquid only", "ice only", "both"],
           xlabel="ORAC phase", ylabel="ACM-CAP column phase",
           title=f"Phase, row-normalised  (n = {int(table.values.sum()):,})")

    fig.suptitle(f"SLSTR-pixel × ACM-CAP collocation quick-look — {csv.split('/')[-1]}\n"
                 f"Lower row: |Δt| ≤ {max_dt_min:g} min, reference quality_status = 0; "
                 f"A-CTH ≤ tropopause + 2 km",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], *(float(a) for a in sys.argv[3:4])))
