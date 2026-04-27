"""R10 vs R11 ORAC retrieval comparison against the same ATLID reference.

Two figures:

- :func:`scatter_compare` : sample-level hexbin scatter of ORAC cot vs
  ATLID column τ for R10 and R11 side by side. Reports N, bias, RMSE,
  R_log per panel.
- :func:`bias_bar_compare` : bias-by-stratum dual bar chart with R10
  and R11 plotted next to each other for every stratum. Used to see
  per-regime improvement of R11 over R10.

The two input DataFrames must come from collocation runs that share
the same ATLID frame set (which they do for a given month) — the join
is along the ``frame_id`` × pixel match, so the comparison is
apples-to-apples.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figures import COT_FLOOR, LOG_LABELS, LOG_LIM, LOG_TICKS, _logclip, _stats


def _setup_log_axes(ax) -> None:
    ax.plot(LOG_LIM, LOG_LIM, "k--", lw=0.8)
    ax.set_xlim(LOG_LIM); ax.set_ylim(LOG_LIM)
    ax.set_xticks(LOG_TICKS); ax.set_yticks(LOG_TICKS)
    ax.set_xticklabels(LOG_LABELS); ax.set_yticklabels(LOG_LABELS)
    ax.set_aspect("equal")
    ax.set_xlabel("ATLID column τ₃₅₅")
    ax.set_ylabel("ORAC SEVIRI cot")


def _r_log(d: pd.DataFrame, x: str, y: str) -> float:
    if len(d) < 2:
        return np.nan
    lx = np.log10(np.clip(d[x].values, COT_FLOOR, None))
    ly = np.log10(np.clip(d[y].values, COT_FLOOR, None))
    if lx.std() == 0 or ly.std() == 0:
        return np.nan
    return float(np.corrcoef(lx, ly)[0, 1])


def scatter_compare(
    d_r10: pd.DataFrame,
    d_r11: pd.DataFrame,
    x: str = "cot_atlid",
    y: str = "cot_orac",
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """1×2 hexbin scatter: R10 left, R11 right. R_log shown in title."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
    for ax, d, label in [
        (axes[0], d_r10, "R10"),
        (axes[1], d_r11, "R11"),
    ]:
        d2 = d[[x, y]].dropna()
        n, bias, rmse, _ = _stats(d2, x, y)
        rlog = _r_log(d2, x, y)
        if n >= 2:
            hb = ax.hexbin(_logclip(d2[x]), _logclip(d2[y]),
                           gridsize=50, mincnt=1, bins="log", cmap="viridis")
            fig.colorbar(hb, ax=ax, label="count (log)")
        _setup_log_axes(ax)
        ax.set_title(f"{label}  (N={n})\nbias={bias:+.2f}  RMSE={rmse:.2f}  R_log={rlog:.2f}")
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

    Each stratum gets two adjacent bars. ``stats_*`` are tables from
    :func:`validation.statistics.stratified_stats`.
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
    ax.legend(loc="upper right", fontsize=9)

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
