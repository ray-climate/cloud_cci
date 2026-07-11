"""Plotting helpers for water-cloud cot and cer validation against ACM-CAP.

Re-uses the publishable density-plot style from :mod:`validation.cth_figures`:
LogNorm 2D-histogram, height-matched colourbar, on-top 1:1 line, stat box.

Two axis modes:

- **COT** : log10-axis on [0.1, 100], heavy-tailed.
- **CER** : linear µm on [0, 30], roughly Gaussian.

Both modes share `scatter_panel` / `scatter_compare` /
`scatter_compare_by_surface` / `bias_by_stratum` / `bias_bar_compare` /
`qc_sensitivity_panel` entry points — pick the right one via the
``mode`` argument or use the COT- and CER-specific wrappers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cth_figures import (
    DENSITY_BINS, DENSITY_CMAP, _attach_colorbar, _density_image,
    _save, _stat_text, _stats,
)

COT_LOG_LIM = (-1.0, 2.0)    # log10 of cot (0.1 .. 100)
COT_LOG_TICKS = (-1, 0, 1, 2)
COT_LOG_TLABELS = ("0.1", "1", "10", "100")
COT_FLOOR = 0.1              # display floor; clip <0.1 to 0.1 for scatter

COT_LIN_LIM = (0.0, 60.0)    # linear cot
COT_LIN_TICKS = (0, 10, 20, 30, 40, 50, 60)

CER_LIM = (0.0, 30.0)        # µm
CER_TICKS = (0, 5, 10, 15, 20, 25, 30)


def _logclip(s: pd.Series, floor: float = COT_FLOOR) -> np.ndarray:
    return np.log10(s.clip(lower=floor).values)


def _setup_cot_axes(ax, scale: str = "log") -> None:
    """COT axes for ATLID (x) vs ORAC (y). ``scale`` is 'log' or 'linear'."""
    if scale == "log":
        lim, ticks, labels = COT_LOG_LIM, COT_LOG_TICKS, COT_LOG_TLABELS
    elif scale == "linear":
        lim, ticks, labels = COT_LIN_LIM, COT_LIN_TICKS, None
    else:
        raise ValueError(f"unknown cot scale {scale!r}; expected 'log' or 'linear'")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    if labels is not None:
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="in", top=True, right=True, length=4)
    ax.set_xlabel(r"ACM-CAP liquid optical depth $\tau$")
    ax.set_ylabel(r"ORAC cot")
    ax.plot(lim, lim, color="0.2", lw=0.9, ls="--", zorder=3)


def _setup_cer_axes(ax) -> None:
    """Linear µm axes for cer validation."""
    ax.set_xlim(CER_LIM); ax.set_ylim(CER_LIM)
    ax.set_xticks(CER_TICKS); ax.set_yticks(CER_TICKS)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="in", top=True, right=True, length=4)
    ax.set_xlabel(r"ACM-CAP liquid effective radius [$\mu$m]")
    ax.set_ylabel(r"ORAC cer [$\mu$m]")
    ax.plot(CER_LIM, CER_LIM, color="0.2", lw=0.9, ls="--", zorder=3)


# ---------------------------------------------------------------------------
# Internal mode-aware helpers
# ---------------------------------------------------------------------------
# ``mode`` accepts 'cot' (log), 'cot_linear' (linear cot), 'cer' (linear µm).

def _setup_axes(ax, mode: str) -> None:
    if mode == "cot":
        _setup_cot_axes(ax, scale="log")
    elif mode == "cot_linear":
        _setup_cot_axes(ax, scale="linear")
    elif mode == "cer":
        _setup_cer_axes(ax)
    else:
        raise ValueError(
            f"unknown mode {mode!r}; expected 'cot', 'cot_linear', or 'cer'"
        )


def _xy_for_mode(d: pd.DataFrame, x: str, y: str, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return numeric (x, y) arrays in axis space.

    'cot' uses log10(clip≥0.1); 'cot_linear' and 'cer' are passed through.
    """
    if mode == "cot":
        return _logclip(d[x]), _logclip(d[y])
    return d[x].values, d[y].values


def _lim_for_mode(mode: str) -> tuple[float, float]:
    return {
        "cot": COT_LOG_LIM,
        "cot_linear": COT_LIN_LIM,
        "cer": CER_LIM,
    }[mode]


# ---------------------------------------------------------------------------
# Public panels — generic over ``mode in {'cot', 'cer'}``
# ---------------------------------------------------------------------------

def scatter_panel(
    sample: pd.DataFrame,
    pixel: pd.DataFrame,
    *,
    mode: str,
    x: str,
    y: str,
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """Side-by-side density panel: sample-level vs pixel-aggregate."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
    for ax, d, label in [
        (axes[0], sample, "sample-level (nearest ATLID)"),
        (axes[1], pixel, "pixel-aggregate (mean cloudy ATLID)"),
    ]:
        d2 = d[[x, y]].dropna()
        n, bias, rmse, r = _stats(d2, x, y)
        _setup_axes(ax, mode)
        if n >= 2:
            xv, yv = _xy_for_mode(d2, x, y, mode)
            im = _density_image(ax, xv, yv, lim=_lim_for_mode(mode))
            _attach_colorbar(fig, ax, im, label="count")
        _stat_text(ax, n, bias, rmse, r, median_bias=(float((d2[y] - d2[x]).median()) if n >= 2 else None),
                   unit=("" if mode.startswith("cot") else r"$\mu$m"))
        ax.set_title(label, pad=6)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, out)


def scatter_compare(
    d_r10: pd.DataFrame,
    d_r11: pd.DataFrame,
    *,
    mode: str,
    x: str,
    y: str,
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """1×2 density scatter, R10 left vs R11 right."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
    for ax, d, label in [(axes[0], d_r10, "R10"), (axes[1], d_r11, "R11")]:
        d2 = d[[x, y]].dropna()
        n, bias, rmse, r = _stats(d2, x, y)
        _setup_axes(ax, mode)
        if n >= 2:
            xv, yv = _xy_for_mode(d2, x, y, mode)
            im = _density_image(ax, xv, yv, lim=_lim_for_mode(mode))
            _attach_colorbar(fig, ax, im, label="count")
        _stat_text(ax, n, bias, rmse, r, median_bias=(float((d2[y] - d2[x]).median()) if n >= 2 else None),
                   unit=("" if mode.startswith("cot") else r"$\mu$m"))
        ax.set_title(label, pad=6)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, out)


def scatter_compare_by_surface(
    d_r10: pd.DataFrame,
    d_r11: pd.DataFrame,
    *,
    mode: str,
    x: str,
    y: str,
    lsflag_col: str = "lsflag_orac",
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """2×2 density scatter — rows: ocean / land, cols: R10 / R11."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5))
    surfaces = (
        ("Ocean", lambda d: d[lsflag_col] < 0.5),
        ("Land",  lambda d: d[lsflag_col] >= 0.5),
    )
    retrievals = (("R10", d_r10), ("R11", d_r11))
    for i, (s_name, mask_fn) in enumerate(surfaces):
        for j, (r_name, d) in enumerate(retrievals):
            ax = axes[i, j]
            d2 = d.loc[mask_fn(d).fillna(False), [x, y]].dropna()
            n, bias, rmse, r = _stats(d2, x, y)
            _setup_axes(ax, mode)
            if n >= 2:
                xv, yv = _xy_for_mode(d2, x, y, mode)
                im = _density_image(ax, xv, yv, lim=_lim_for_mode(mode))
                _attach_colorbar(fig, ax, im, label="count")
            _stat_text(ax, n, bias, rmse, r, median_bias=(float((d2[y] - d2[x]).median()) if n >= 2 else None),
                   unit=("" if mode.startswith("cot") else r"$\mu$m"))
            ax.set_title(f"{r_name} — {s_name}", pad=6)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.0)
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Bar charts (mode-agnostic)
# ---------------------------------------------------------------------------

def bias_by_stratum(
    stats: pd.DataFrame, *, metric: str = "bias",
    title: str = "", out: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of ``metric`` across strata. Mirrors cth_figures' helper."""
    d = stats.dropna(subset=[metric]).copy()
    d = d[d["stratum"] != "all"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["tab:blue" if v < 0 else "tab:red" for v in d[metric]]
    ax.bar(d["stratum"], d[metric], color=colors, edgecolor="0.3")
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_ylabel(metric); ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    for tick in ax.get_xticklabels(): tick.set_horizontalalignment("right")
    if d.empty:
        return _save(fig, out)
    ymax = d[metric].abs().max() * 1.15
    for i, (val, n) in enumerate(zip(d[metric].values, d["n"].values)):
        offset = (0.02 * ymax) if val >= 0 else (-0.04 * ymax)
        ax.text(i, val + offset, f"N={int(n)}",
                ha="center", fontsize=8, color="0.25")
    fig.tight_layout()
    return _save(fig, out)


def bias_bar_compare(
    stats_r10: pd.DataFrame, stats_r11: pd.DataFrame, *,
    metric: str = "bias", title: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """R10 vs R11 dual-bar chart of one metric across strata."""
    s10 = stats_r10.set_index("stratum"); s11 = stats_r11.set_index("stratum")
    common = [s for s in s11.index if s in s10.index and s != "all"]
    common = [s for s in common
              if pd.notna(s10.loc[s, metric]) and pd.notna(s11.loc[s, metric])]
    v10 = s10.loc[common, metric].values; v11 = s11.loc[common, metric].values
    n10 = s10.loc[common, "n"].values; n11 = s11.loc[common, "n"].values
    x = np.arange(len(common)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w/2, v10, w, color="tab:gray",   edgecolor="0.3", label="R10")
    ax.bar(x + w/2, v11, w, color="tab:orange", edgecolor="0.3", label="R11")
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(common, rotation=35, ha="right")
    ax.set_ylabel(metric); ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    if not common:
        return _save(fig, out)
    yspan = max(np.nanmax(np.abs(np.r_[v10, v11])) * 1.18, 1e-3)
    for i, (a, b, na, nb) in enumerate(zip(v10, v11, n10, n11)):
        ax.text(i - w/2, a + np.sign(a or 1) * 0.02 * yspan,
                f"N={int(na)}", ha="center", fontsize=7, color="0.25")
        ax.text(i + w/2, b + np.sign(b or 1) * 0.02 * yspan,
                f"N={int(nb)}", ha="center", fontsize=7, color="tab:red")
    fig.tight_layout()
    return _save(fig, out)


def qc_sensitivity_panel(
    stats: pd.DataFrame, *, title: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """Bias / RMSE / R / N versus QC mode, sample vs pixel."""
    a = stats[stats["stratum"] == "all"].copy()
    if a.empty:
        raise ValueError("stats has no 'all' stratum rows")
    qc_order = ["qc_off", "qc_relaxed", "qc_strict"]
    a["qc_mode"] = pd.Categorical(a["qc_mode"], categories=qc_order, ordered=True)
    a = a.sort_values(["qc_mode", "view"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [("bias", "bias"), ("rmse", "RMSE"), ("r", "Pearson R"), ("n", "N")]
    for ax, (m, ylabel) in zip(axes.flat, metrics):
        for view, off, c in (("sample", -0.18, "tab:blue"),
                              ("pixel", +0.18, "tab:orange")):
            sub = a[a["view"] == view].sort_values("qc_mode")
            xpos = np.arange(len(sub)) + off
            ax.bar(xpos, sub[m].values, width=0.36, color=c, edgecolor="0.3", label=view)
        ax.set_xticks(np.arange(len(qc_order)))
        ax.set_xticklabels(qc_order, rotation=20, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(m)
        if m == "bias":
            ax.axhline(0, color="0.3", lw=0.7)
    axes[0, 0].legend(loc="best", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return _save(fig, out)


def homogeneity_sweep(
    sweep_stats: pd.DataFrame, *,
    r_metric: str = "r",
    r_label: str = "Pearson R",
    title: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """1x3 panel of bias / RMSE / correlation as a function of homogeneity bin.

    One curve per ``n_cut`` (minimum liquid-only ATLID profiles per ORAC
    pixel). The input is the table produced by
    :func:`validation.statistics.homogeneity_sweep_stats`. For heavy-tailed
    quantities like COT, pass ``r_metric="r_log"`` so the third panel shows
    the log-space correlation that is not dominated by a few extreme points.
    """
    d = sweep_stats.copy()
    bin_order = d.sort_values("cv_lo")["cv_bin"].drop_duplicates().tolist()
    n_cuts = sorted(d["n_cut"].unique().tolist())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(0.15 + 0.7 * i / max(len(n_cuts) - 1, 1)) for i in range(len(n_cuts))]

    for metric, ax, ylabel in [
        ("bias",   axes[0], "bias (ORAC − ATLID)"),
        ("rmse",   axes[1], "RMSE"),
        (r_metric, axes[2], r_label),
    ]:
        for n_cut, color in zip(n_cuts, colors):
            sub = (d[d["n_cut"] == n_cut]
                   .set_index("cv_bin").reindex(bin_order).reset_index())
            ax.plot(bin_order, sub[metric].values,
                    marker="o", color=color, lw=1.8,
                    label=f"n ≥ {n_cut}")
            for i, (val, n) in enumerate(zip(sub[metric].values, sub["n"].values)):
                if np.isfinite(val) and pd.notna(n) and n > 0:
                    ax.annotate(f"N={int(n)}", xy=(i, val), xytext=(0, 6),
                                textcoords="offset points",
                                ha="center", fontsize=7, color=color)
        ax.set_xlabel("homogeneity (ref_cv_atlid bin)")
        ax.set_ylabel(ylabel)
        if metric == "bias":
            ax.axhline(0, color="0.3", lw=0.7)
        ax.grid(alpha=0.3)

    axes[0].legend(loc="best", fontsize=9, frameon=False)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return _save(fig, out)
