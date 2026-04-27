"""Plotting helpers for cth validation reports.

Linear-axis variants of :mod:`validation.figures` — CTH is roughly linear
on [0, 18] km, so log-clipping (used for cot) would compress the
informative high-cirrus regime.

Three reusable panels:

- :func:`scatter_panel`     : sample-level + pixel-aggregate hexbin.
- :func:`diagnostic_panel`  : 2×2 scatter coloured by lat / dist / Δt
                              / ATLID cloud_class.
- :func:`qc_sensitivity_panel` : bar chart of bias and N across QC modes,
                              built directly from a :func:`cth_report`
                              stats table.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CTH_LIM = (0, 18)
CTH_TICKS = (0, 3, 6, 9, 12, 15, 18)


def _stats(d: pd.DataFrame, x: str, y: str) -> tuple[int, float, float, float]:
    if len(d) < 2:
        return len(d), np.nan, np.nan, np.nan
    diff = d[y] - d[x]
    return (
        len(d),
        float(diff.mean()),
        float(np.sqrt((diff ** 2).mean())),
        float(np.corrcoef(d[x], d[y])[0, 1]) if d[x].std() > 0 and d[y].std() > 0 else np.nan,
    )


def _setup_axes(ax) -> None:
    ax.plot(CTH_LIM, CTH_LIM, "k--", lw=0.8)
    ax.set_xlim(CTH_LIM); ax.set_ylim(CTH_LIM)
    ax.set_xticks(CTH_TICKS); ax.set_yticks(CTH_TICKS)
    ax.set_aspect("equal")
    ax.set_xlabel("ATLID cloud top height [km]")
    ax.set_ylabel("ORAC SEVIRI cth_corrected [km]")


def scatter_panel(
    sample: pd.DataFrame,
    pixel: pd.DataFrame,
    x: str = "cth_atlid_thick_km",
    y: str = "cth_orac_corrected_km",
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """Side-by-side hexbin: sample-level vs pixel-aggregate, linear km."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
    for ax, d, label in [
        (axes[0], sample, "sample-level (nearest ATLID)"),
        (axes[1], pixel, "pixel-aggregate (mean cloudy ATLID)"),
    ]:
        d2 = d[[x, y]].dropna()
        n, bias, rmse, r = _stats(d2, x, y)
        if n >= 2:
            hb = ax.hexbin(d2[x].values, d2[y].values,
                           gridsize=40, mincnt=1, bins="log",
                           extent=(*CTH_LIM, *CTH_LIM), cmap="viridis")
            fig.colorbar(hb, ax=ax, label="count (log)")
        _setup_axes(ax)
        ax.set_title(f"{label}  (N={n})\nbias={bias:+.2f}  RMSE={rmse:.2f}  R={r:.2f}")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def diagnostic_panel(
    sample: pd.DataFrame,
    x: str = "cth_atlid_thick_km",
    y: str = "cth_orac_corrected_km",
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """2×2 scatter coloured by lat / dist / Δt / ATLID cloud class."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    cols = [x, y, "ec_lat", "distance_km", "time_diff_s", "cloud_class_atlid"]
    d = sample[cols].dropna(subset=[x, y])
    xv, yv = d[x].values, d[y].values

    sc = axes[0, 0].scatter(xv, yv, c=d["ec_lat"], cmap="viridis",
                            s=4, alpha=0.6, vmin=-60, vmax=60)
    fig.colorbar(sc, ax=axes[0, 0], label="latitude [deg]")
    axes[0, 0].set_title("(a) coloured by latitude")

    sc = axes[0, 1].scatter(xv, yv, c=d["distance_km"], cmap="plasma",
                            s=4, alpha=0.6, vmin=0, vmax=6)
    fig.colorbar(sc, ax=axes[0, 1], label="dist to SEVIRI pixel [km]")
    axes[0, 1].set_title("(b) coloured by match distance")

    sc = axes[1, 0].scatter(xv, yv, c=d["time_diff_s"], cmap="cividis",
                            s=4, alpha=0.6, vmin=0, vmax=450)
    fig.colorbar(sc, ax=axes[1, 0], label="|Δt| [s]")
    axes[1, 0].set_title("(c) coloured by time offset")

    # Cloud-class colouring with categorical legend.
    class_names = {1: "thick", 2: "thin", 3: "thin/thick", 4: "thick/thick", 5: "thin/thin"}
    palette = {1: "tab:blue", 2: "tab:orange", 3: "tab:green",
               4: "tab:red", 5: "tab:purple"}
    for cls, name in class_names.items():
        sub = d[d["cloud_class_atlid"] == cls]
        if len(sub) == 0:
            continue
        axes[1, 1].scatter(sub[x], sub[y], s=6, alpha=0.7,
                           c=palette[cls], label=f"{name} (N={len(sub)})")
    axes[1, 1].legend(loc="lower right", fontsize=8)
    axes[1, 1].set_title("(d) coloured by ATLID cloud class")

    for ax in axes.flat:
        _setup_axes(ax)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def bias_by_stratum(
    stats: pd.DataFrame,
    *,
    metric: str = "bias",
    title: str = "Bias by stratum",
    out: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of one metric across strata. Mirrors :func:`validation.figures.bias_by_stratum`."""
    d = stats.dropna(subset=[metric]).copy()
    d = d[d["stratum"] != "all"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["tab:blue" if v < 0 else "tab:red" for v in d[metric]]
    ax.bar(d["stratum"], d[metric], color=colors, edgecolor="0.3")
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")

    ymax = d[metric].abs().max() * 1.15
    for i, (val, n) in enumerate(zip(d[metric].values, d["n"].values)):
        offset = (0.02 * ymax) if val >= 0 else (-0.04 * ymax)
        ax.text(i, val + offset, f"N={int(n)}",
                ha="center", fontsize=8, color="0.25")

    fig.tight_layout()
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def scatter_compare(
    d_r10: pd.DataFrame,
    d_r11: pd.DataFrame,
    x: str = "cth_atlid_thick_km",
    y: str = "cth_orac_corrected_km",
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """1×2 hexbin scatter, R10 left vs R11 right, linear km."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
    for ax, d, label in [(axes[0], d_r10, "R10"), (axes[1], d_r11, "R11")]:
        d2 = d[[x, y]].dropna()
        n, bias, rmse, r = _stats(d2, x, y)
        if n >= 2:
            hb = ax.hexbin(d2[x].values, d2[y].values,
                           gridsize=40, mincnt=1, bins="log",
                           extent=(*CTH_LIM, *CTH_LIM), cmap="viridis")
            fig.colorbar(hb, ax=ax, label="count (log)")
        _setup_axes(ax)
        ax.set_title(f"{label}  (N={n})\nbias={bias:+.2f}  RMSE={rmse:.2f}  R={r:.2f}")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def bias_bar_compare(
    stats_r10: pd.DataFrame,
    stats_r11: pd.DataFrame,
    *,
    metric: str = "bias",
    title: str = "R10 vs R11 — bias by stratum",
    out: str | Path | None = None,
) -> plt.Figure:
    """Dual-bar chart of one metric across strata, R10 vs R11.

    ``stats_*`` are tables from :func:`validation.statistics.stratified_stats`
    (or one ``view`` slice from :func:`cth_report`).
    """
    s10 = stats_r10.set_index("stratum")
    s11 = stats_r11.set_index("stratum")
    common = [s for s in s11.index if s in s10.index and s != "all"]
    common = [s for s in common
              if pd.notna(s10.loc[s, metric]) and pd.notna(s11.loc[s, metric])]

    v10 = s10.loc[common, metric].values
    v11 = s11.loc[common, metric].values
    n10 = s10.loc[common, "n"].values
    n11 = s11.loc[common, "n"].values

    x = np.arange(len(common))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w / 2, v10, w, color="tab:gray",   edgecolor="0.3", label="R10")
    ax.bar(x + w / 2, v11, w, color="tab:orange", edgecolor="0.3", label="R11")
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(common, rotation=35, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)

    yspan = max(np.nanmax(np.abs(np.r_[v10, v11])) * 1.18, 1e-3)
    for i, (a, b, na, nb) in enumerate(zip(v10, v11, n10, n11)):
        ax.text(i - w / 2, a + np.sign(a or 1) * 0.02 * yspan,
                f"N={int(na)}", ha="center", fontsize=7, color="0.25")
        ax.text(i + w / 2, b + np.sign(b or 1) * 0.02 * yspan,
                f"N={int(nb)}", ha="center", fontsize=7, color="tab:red")
    fig.tight_layout()
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def qc_sensitivity_panel(
    stats: pd.DataFrame,
    *,
    title: str = "QC sensitivity",
    out: str | Path | None = None,
) -> plt.Figure:
    """Bias / RMSE / R / N versus QC mode, sample-level vs pixel-aggregate.

    Built directly from the tidy table emitted by
    :func:`validation.statistics.cth_report`. Only the ``stratum=='all'``
    rows are used.
    """
    a = stats[stats["stratum"] == "all"].copy()
    if a.empty:
        raise ValueError("stats has no 'all' stratum rows")

    qc_order = ["qc_off", "qc_no_trop_cap", "qc_relaxed", "qc_strict"]
    a["qc_mode"] = pd.Categorical(a["qc_mode"], categories=qc_order, ordered=True)
    a = a.sort_values(["qc_mode", "view"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [("bias", "bias [km]"), ("rmse", "RMSE [km]"),
               ("r", "Pearson R"), ("n", "N")]
    for ax, (m, ylabel) in zip(axes.flat, metrics):
        for view, off, c in (("sample", -0.18, "tab:blue"),
                              ("pixel", +0.18, "tab:orange")):
            sub = a[a["view"] == view].sort_values("qc_mode")
            xpos = np.arange(len(sub)) + off
            ax.bar(xpos, sub[m].values, width=0.36, color=c,
                   edgecolor="0.3", label=view)
        ax.set_xticks(np.arange(len(qc_order)))
        ax.set_xticklabels(qc_order, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(m)
        if m == "bias":
            ax.axhline(0, color="0.3", lw=0.7)
    axes[0, 0].legend(loc="best", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig
