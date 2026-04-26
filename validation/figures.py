"""Plotting helpers for cot validation reports.

Three reusable panel makers:

- :func:`scatter_panel` : sample-level + pixel-aggregate hexbin scatters
  side by side, both on log-log axes (cot range 0.05–100). Reports
  N, bias, RMSE, R in each title.
- :func:`diagnostic_panel` : 2×2 scatter coloured by latitude,
  match distance, time offset, and attenuated flag. Used to
  diagnose whether the residuals are driven by geometry/QC.
- :func:`bias_by_stratum` : bar chart of bias per stratum from a
  :func:`validation.statistics.stratified_stats` table.

All figures use a uniform log-axis convention so they compare directly
across frames / months.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_LIM = (-1.5, 2.2)
LOG_TICKS = (-1, 0, 1, 2)
LOG_LABELS = ("0.1", "1", "10", "100")
COT_FLOOR = 0.05  # display floor — ATLID/ORAC values below this are clipped


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


def _setup_log_axes(ax) -> None:
    ax.plot(LOG_LIM, LOG_LIM, "k--", lw=0.8)
    ax.set_xlim(LOG_LIM); ax.set_ylim(LOG_LIM)
    ax.set_xticks(LOG_TICKS); ax.set_yticks(LOG_TICKS)
    ax.set_xticklabels(LOG_LABELS); ax.set_yticklabels(LOG_LABELS)
    ax.set_aspect("equal")
    ax.set_xlabel("ATLID column τ₃₅₅")
    ax.set_ylabel("ORAC SEVIRI cot")


def _logclip(s: pd.Series) -> np.ndarray:
    return np.log10(s.clip(lower=COT_FLOOR).values)


def scatter_panel(
    sample: pd.DataFrame,
    pixel: pd.DataFrame,
    x: str = "cot_atlid",
    y: str = "cot_orac",
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """Side-by-side hexbin scatter: sample-level vs pixel-aggregate.

    Both panels show ``log10(y)`` vs ``log10(x)`` over τ ∈ [0.05, 100].
    Title strip carries N, bias, RMSE, R for each view.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
    for ax, d, label in [
        (axes[0], sample, "sample-level"),
        (axes[1], pixel, "pixel-aggregate"),
    ]:
        d2 = d[[x, y]].dropna()
        n, bias, rmse, r = _stats(d2, x, y)
        if n >= 2:
            hb = ax.hexbin(_logclip(d2[x]), _logclip(d2[y]),
                           gridsize=50, mincnt=1, bins="log", cmap="viridis")
            fig.colorbar(hb, ax=ax, label="count (log)")
        _setup_log_axes(ax)
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
    x: str = "cot_atlid",
    y: str = "cot_orac",
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """2×2 scatter coloured by lat / dist / time-diff / attenuated flag.

    Used to diagnose where the residuals come from. ``sample`` is the
    sample-level DataFrame after the standard base filter
    (cloudy + ATLID > 0 + not saturated).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    d = sample[[x, y, "ec_lat", "distance_km", "time_diff_s", "attenuated"]].dropna(
        subset=[x, y]
    )
    xv = _logclip(d[x]); yv = _logclip(d[y])

    sc = axes[0, 0].scatter(xv, yv, c=d["ec_lat"], cmap="viridis",
                            s=4, alpha=0.6, vmin=20, vmax=70)
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

    not_att = d[~d["attenuated"]]
    att = d[d["attenuated"]]
    axes[1, 1].scatter(_logclip(not_att[x]), _logclip(not_att[y]),
                       c="0.7", s=4, alpha=0.5, label=f"normal (N={len(not_att)})")
    axes[1, 1].scatter(_logclip(att[x]), _logclip(att[y]),
                       c="tab:red", s=8, alpha=0.7,
                       label=f"attenuated (N={len(att)})")
    axes[1, 1].legend(loc="lower right", fontsize=9)
    axes[1, 1].set_title("(d) attenuated profiles (τ is lower bound)")

    for ax in axes.flat:
        _setup_log_axes(ax)
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
    """Bar chart of one metric across strata, labelling N per bar.

    ``stats`` is a table from :func:`validation.statistics.stratified_stats`;
    ``metric`` is one of ``bias / rmse / mae / r``.
    """
    d = stats.dropna(subset=[metric]).copy()
    d = d[d["stratum"] != "all"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["tab:blue" if v < 0 else "tab:red" for v in d[metric]]
    ax.bar(d["stratum"], d[metric], color=colors, edgecolor="0.3")
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=35)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")

    # N labels above bars
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
