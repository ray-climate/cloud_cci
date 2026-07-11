"""CTH results broken down by cloud type — a self-explanatory replacement for the
generic bias-by-stratum bar chart.

Shows, for each cloud-top-height band and each ATLID cloud class, the three
headline metrics side by side (bias, RMSE, Pearson R) with full descriptive
names and the sample size N annotated — so a reader sees at a glance which cloud
types ORAC SLSTR CTH gets right and which it does not.

Usage: python scripts/slstr_cth_cloudtype_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATS = "validation_data/slstr_cth_2025-12.csv"
OUT = Path("figures/slstr_cth_2025-12")

# (stratum key, full name, group). Order is bottom→top on the plot.
ROWS = [
    ("cth_high", "High (≥ 7 km)",         "Cloud-top\nheight"),
    ("cth_mid",  "Mid (3–7 km)",          "Cloud-top\nheight"),
    ("cth_low",  "Low (< 3 km)",          "Cloud-top\nheight"),
    ("class_thick_over_thick", "Thick over thick", "ATLID\ncloud class"),
    ("class_thin_over_thick",  "Thin over thick",  "ATLID\ncloud class"),
    ("class_thin",             "Thin single",      "ATLID\ncloud class"),
    ("class_thick",            "Thick single",     "ATLID\ncloud class"),
]


def main() -> int:
    s = pd.read_csv(STATS)
    h = s[(s["qc_mode"] == "qc_strict") & (s["view"] == "pixel")].set_index("stratum")

    labels, groups, bias, rmse, r, n = [], [], [], [], [], []
    for key, name, grp in ROWS:
        labels.append(name); groups.append(grp)
        if key in h.index and np.isfinite(h.loc[key, "n"]) and h.loc[key, "n"] > 0:
            bias.append(h.loc[key, "bias"]); rmse.append(h.loc[key, "rmse"])
            r.append(h.loc[key, "r"]); n.append(int(h.loc[key, "n"]))
        else:
            bias.append(np.nan); rmse.append(np.nan); r.append(np.nan); n.append(0)
    bias, rmse, r = np.array(bias), np.array(rmse), np.array(r)

    # y positions with a gap between the two groups
    y, ticks, cur = [], [], 0.0
    for i, g in enumerate(groups):
        if i > 0 and g != groups[i - 1]:
            cur += 1.0
        y.append(cur); ticks.append(cur); cur += 1.0
    y = np.array(y)

    fig, ax = plt.subplots(1, 3, figsize=(14, 5.4), sharey=True,
                           gridspec_kw=dict(wspace=0.08))

    # Panel 1 — bias (diverging by sign). Labels always sit just to the RIGHT of
    # the bar tip (white inside negative bars) so they never collide with the
    # long cloud-type names on the left.
    colors = ["#c0392b" if (np.isfinite(b) and b < 0) else "#1565c0" for b in bias]
    ax[0].barh(y, np.nan_to_num(bias), color=colors, height=0.62, zorder=2)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_xlim(-4.6, 0.9)
    ax[0].set_xlabel("bias  ORAC − ATLID  [km]")
    ax[0].set_title("(a) Bias", fontsize=11)
    for yi, b in zip(y, bias):
        if np.isfinite(b):
            inside = b < -0.4
            ax[0].text(b + 0.1, yi, f"{b:+.2f}", va="center", ha="left",
                       fontsize=8.5, color="white" if inside else "black",
                       fontweight="bold" if inside else "normal")
        else:
            ax[0].text(0.05, yi, "no data", va="center", ha="left",
                       fontsize=8.5, color="grey", style="italic")

    # Panel 2 — RMSE
    ax[1].barh(y, np.nan_to_num(rmse), color="#7e57c2", height=0.62, zorder=2)
    ax[1].set_xlabel("RMSE  [km]")
    ax[1].set_title("(b) RMSE", fontsize=11)
    for yi, v in zip(y, rmse):
        if np.isfinite(v):
            ax[1].text(v + 0.08, yi, f"{v:.2f}", va="center", ha="left", fontsize=8.5)

    # Panel 3 — Pearson R (0–1), coloured by value
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    norm = Normalize(0, 0.8)
    rc = [cm.viridis(norm(v)) if np.isfinite(v) else "#dddddd" for v in r]
    ax[2].barh(y, np.nan_to_num(r), color=rc, height=0.62, zorder=2)
    ax[2].set_xlim(0, 1.0)
    ax[2].set_xlabel("Pearson R")
    ax[2].set_title("(c) Correlation", fontsize=11)
    for yi, v, nn in zip(y, r, n):
        if np.isfinite(v):
            ax[2].text(v + 0.02, yi, f"R={v:.2f}\nN={nn:,}", va="center",
                       ha="left", fontsize=7.8)

    ax[0].set_yticks(ticks)
    ax[0].set_yticklabels(labels, fontsize=9.5)
    ax[0].set_ylim(-0.7, max(y) + 0.7)
    for a in ax:
        a.grid(axis="x", alpha=0.3, zorder=0)

    # group labels in the far-left margin (clear of the — now short — tick labels)
    for grp in dict.fromkeys(groups):
        ys = [yy for yy, gg in zip(y, groups) if gg == grp]
        ax[0].text(-0.30, np.mean(ys), grp, transform=ax[0].get_yaxis_transform(),
                   rotation=90, va="center", ha="center", fontsize=9.5,
                   fontweight="bold", color="#444", linespacing=0.9)

    fig.suptitle("ORAC SLSTR cloud-top height by cloud type — December 2025 "
                 "(polar, qc_strict, pixel)\n"
                 "single-layer / low cloud is unbiased; multi-layer & high cloud "
                 "are underestimated by ~4 km", fontsize=12)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.87, bottom=0.11, wspace=0.08)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "cth_by_cloud_type.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
