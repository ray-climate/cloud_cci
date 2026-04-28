"""R10 vs R11 ORAC retrieval comparison against the same ATLID reference.

Same publication-quality density-scatter style as
:mod:`validation.cth_figures`, adapted for cot's log10 axes.

Panels:

- :func:`scatter_compare`            : 1×2 R10 vs R11 density.
- :func:`scatter_compare_by_surface` : 2×2 R10/R11 × ocean/land density.
- :func:`bias_bar_compare`           : R10 vs R11 dual-bar chart.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figures import (
    LOG_LIM,
    _attach_colorbar,
    _density_image,
    _logclip,
    _save,
    _setup_log_axes,
    _stat_text,
    _stats,
)


def scatter_compare(
    d_r10: pd.DataFrame,
    d_r11: pd.DataFrame,
    x: str = "cot_atlid",
    y: str = "cot_orac",
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """1×2 density scatter, R10 left vs R11 right."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
    for ax, d, label in [(axes[0], d_r10, "R10"), (axes[1], d_r11, "R11")]:
        d2 = d[[x, y]].dropna()
        n, bias, rmse, _, r_log = _stats(d2, x, y)
        _setup_log_axes(ax)
        if n >= 2:
            im = _density_image(ax, _logclip(d2[x]), _logclip(d2[y]))
            _attach_colorbar(fig, ax, im, label="count")
        _stat_text(ax, n, bias, rmse, r_log)
        ax.set_title(label, pad=6)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, out)


def scatter_compare_by_surface(
    d_r10: pd.DataFrame,
    d_r11: pd.DataFrame,
    x: str = "cot_atlid",
    y: str = "cot_orac",
    lsflag_col: str = "lsflag_orac",
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """2×2 density scatter — rows: ocean / land, cols: R10 / R11.

    The land/ocean split uses ORAC ``lsflag_orac`` (0 = sea, 1 = land).
    Each panel reports its own N, bias, RMSE, R_log.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5))
    surfaces = (
        ("Ocean", lambda d: d[lsflag_col] < 0.5),
        ("Land",  lambda d: d[lsflag_col] >= 0.5),
    )
    retrievals = (("R10", d_r10), ("R11", d_r11))
    for i, (s_name, mask_fn) in enumerate(surfaces):
        for j, (r_name, d) in enumerate(retrievals):
            ax = axes[i, j]
            if lsflag_col in d.columns:
                d2 = d.loc[mask_fn(d).fillna(False), [x, y]].dropna()
            else:
                d2 = d.iloc[0:0][[x, y]]
            n, bias, rmse, _, r_log = _stats(d2, x, y)
            _setup_log_axes(ax)
            if n >= 2:
                im = _density_image(ax, _logclip(d2[x]), _logclip(d2[y]))
                _attach_colorbar(fig, ax, im, label="count")
            _stat_text(ax, n, bias, rmse, r_log)
            ax.set_title(f"{r_name} — {s_name}", pad=6)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.0)
    fig.tight_layout()
    return _save(fig, out)


def bias_bar_compare(
    stats_r10: pd.DataFrame,
    stats_r11: pd.DataFrame,
    *,
    metric: str = "bias",
    title: str = "R10 vs R11 — bias by stratum",
    out: str | Path | None = None,
) -> plt.Figure:
    """Dual-bar chart of one metric across strata, R10 vs R11."""
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
