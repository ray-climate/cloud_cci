"""Plotting helpers for cot validation reports.

Same publication-quality conventions as :mod:`validation.cth_figures`,
adapted for cot's log-distributed nature: 2D-histogram density on
``imshow`` with ``LogNorm``, log10 axes spanning [0.05, 100], the 1:1
line drawn above the density layer, and an inset stats text box.

Reusable panels:

- :func:`scatter_panel` : sample + pixel side-by-side density.
- :func:`diagnostic_panel` : 2×2 scatter coloured by latitude / match
  distance / time offset / ATLID attenuated flag.
- :func:`bias_by_stratum` : bar chart of one metric across strata from
  a :func:`validation.statistics.stratified_stats` table.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

LOG_LIM = (-1.5, 2.2)
LOG_TICKS = (-1, 0, 1, 2)
LOG_LABELS = ("0.1", "1", "10", "100")
COT_FLOOR = 0.05  # display floor — ATLID/ORAC values below this are clipped
DENSITY_BINS = 60
DENSITY_CMAP = "viridis"


def median_ci95(diff) -> tuple[float, float]:
    """Distribution-free 95% CI for the median of ``diff`` (order statistics)."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return (np.nan, np.nan)
    z = 1.959964
    ds = np.sort(d)
    lo = int(np.clip(np.floor(n / 2 - z * np.sqrt(n) / 2) - 1, 0, n - 1))
    hi = int(np.clip(np.ceil(n / 2 + z * np.sqrt(n) / 2) - 1, 0, n - 1))
    return (float(ds[lo]), float(ds[hi]))


def _stats(d: pd.DataFrame, x: str, y: str) -> tuple[int, float, float, float, float]:
    """Return ``(n, bias, rmse, r, r_log)``. ``r_log`` is Pearson R on
    ``log10(clip(., COT_FLOOR))`` — the metric the cot literature uses
    because raw-space R is dominated by the heavy upper tail."""
    if len(d) < 2:
        return len(d), np.nan, np.nan, np.nan, np.nan
    diff = d[y] - d[x]
    if d[x].std() > 0 and d[y].std() > 0:
        r = float(np.corrcoef(d[x], d[y])[0, 1])
        lx = np.log10(np.clip(d[x].values, COT_FLOOR, None))
        ly = np.log10(np.clip(d[y].values, COT_FLOOR, None))
        r_log = float(np.corrcoef(lx, ly)[0, 1]) if lx.std() > 0 and ly.std() > 0 else np.nan
    else:
        r = r_log = np.nan
    return (
        len(d),
        float(diff.mean()),
        float(np.sqrt((diff ** 2).mean())),
        r,
        r_log,
    )


def _logclip(s: pd.Series) -> np.ndarray:
    return np.log10(s.clip(lower=COT_FLOOR).values)


def _setup_log_axes(ax) -> None:
    """Set up log10 axes and draw the 1:1 line on top of the data layer.

    Mirrors the cth axes convention: ticks-in on all four sides, the
    diagonal at zorder 3 so it stays visible across the dense ridge.
    """
    ax.set_xlim(LOG_LIM); ax.set_ylim(LOG_LIM)
    ax.set_xticks(LOG_TICKS); ax.set_yticks(LOG_TICKS)
    ax.set_xticklabels(LOG_LABELS); ax.set_yticklabels(LOG_LABELS)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="in", top=True, right=True, length=4)
    ax.set_xlabel(r"ATLID column $\tau_{355}$")
    ax.set_ylabel("ORAC cot")
    ax.plot(LOG_LIM, LOG_LIM, color="0.2", lw=0.9, ls="--", zorder=3)


def _density_image(ax, x_log: np.ndarray, y_log: np.ndarray,
                   lim: tuple[float, float] = LOG_LIM,
                   bins: int = DENSITY_BINS, cmap: str = DENSITY_CMAP):
    """2D-histogram density on ``ax`` with ``LogNorm`` colour scale.

    Inputs are already in log10-space so the histogram domain matches
    the axes set by :func:`_setup_log_axes`. Zero-count cells are masked
    so the background shows through.
    """
    H, xedges, yedges = np.histogram2d(x_log, y_log, bins=bins, range=[lim, lim])
    H = H.T  # imshow expects (y, x)
    Hm = np.ma.masked_where(H == 0, H)
    extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
    vmin = 1.0
    vmax = max(2.0, float(Hm.max())) if Hm.count() else 2.0
    im = ax.imshow(
        Hm, origin="lower", extent=extent, cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest", aspect="equal", zorder=2,
    )
    return im


def _attach_colorbar(fig, ax, im, label: str = "count"):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label)
    cb.ax.tick_params(direction="in", length=3)
    return cb


def _stat_text(ax, n: int, bias: float, rmse: float, r_log: float, *,
               median_bias: float | None = None,
               median_ci: tuple[float, float] | None = None,
               loc: tuple[float, float] = (0.04, 0.96)) -> None:
    """Stats annotation in the upper-left of ``ax``.

    Reports ``R_log`` rather than raw ``R`` — for cot the log-space
    correlation is the meaningful metric (Karlsson 2013 / PVIR convention).
    For heavy-tailed cot, ``median_bias`` (if given) is the headline and the
    mean is flagged as skew-sensitive; ``median_ci`` adds a 95% interval.
    """
    if median_bias is not None:
        ci = ""
        if median_ci is not None and np.isfinite(median_ci[0]):
            ci = f" [{median_ci[0]:+.2f}, {median_ci[1]:+.2f}]"
        bias_lines = (f"median bias = {median_bias:+.2f}{ci}\n"
                      f"mean bias = {bias:+.2f} (skewed)\n")
    else:
        bias_lines = f"bias = {bias:+.2f}\n"
    txt = (
        f"$N$ = {n:,}\n"
        f"{bias_lines}"
        f"RMSE = {rmse:.2f}\n"
        f"$R_{{\\mathrm{{log}}}}$ = {r_log:.2f}"
    )
    ax.text(
        loc[0], loc[1], txt, transform=ax.transAxes,
        ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.32", fc="white",
                  ec="0.5", lw=0.6, alpha=0.85),
        zorder=4,
    )


def _save(fig: plt.Figure, out: str | Path | None) -> plt.Figure:
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig


def scatter_panel(
    sample: pd.DataFrame,
    pixel: pd.DataFrame,
    x: str = "cot_atlid",
    y: str = "cot_orac",
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """Side-by-side density scatter: sample-level vs pixel-aggregate.

    Both panels show log10(y) vs log10(x) over τ ∈ [0.05, 100]. The
    inset reports N, bias, RMSE, R_log.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
    for ax, d, label in [
        (axes[0], sample, "sample-level (nearest ATLID)"),
        (axes[1], pixel, "pixel-aggregate (mean cloudy ATLID)"),
    ]:
        d2 = d[[x, y]].dropna()
        n, bias, rmse, _, r_log = _stats(d2, x, y)
        _setup_log_axes(ax)
        if n >= 2:
            im = _density_image(ax, _logclip(d2[x]), _logclip(d2[y]))
            _attach_colorbar(fig, ax, im, label="count")
        _stat_text(ax, n, bias, rmse, r_log,
                   median_bias=(float((d2[y] - d2[x]).median()) if n >= 2 else None),
                   median_ci=(median_ci95(d2[y] - d2[x]) if n >= 2 else None))
        ax.set_title(label, pad=6)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, out)


def diagnostic_panel(
    sample: pd.DataFrame,
    x: str = "cot_atlid",
    y: str = "cot_orac",
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """2×2 scatter coloured by lat / dist / time-diff / attenuated flag."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    d = sample[[x, y, "ec_lat", "distance_km", "time_diff_s", "attenuated"]].dropna(
        subset=[x, y]
    )
    xv = _logclip(d[x]); yv = _logclip(d[y])

    panels = [
        (axes[0, 0], d["ec_lat"].values,        "viridis", 20, 70,
         "latitude [deg]",          "(a) latitude"),
        (axes[0, 1], d["distance_km"].values,   "plasma",   0,  6,
         "match distance [km]",     "(b) match distance"),
        (axes[1, 0], d["time_diff_s"].values,   "cividis",  0, 450,
         r"$|\Delta t|$ [s]",       "(c) time offset"),
    ]
    for ax, c, cmap, vmin, vmax, cb_label, title in panels:
        _setup_log_axes(ax)
        sc = ax.scatter(xv, yv, c=c, cmap=cmap, s=3, alpha=0.55,
                        vmin=vmin, vmax=vmax, zorder=2,
                        edgecolors="none", rasterized=True)
        _attach_colorbar(fig, ax, sc, label=cb_label)
        ax.set_title(title, pad=6)

    ax = axes[1, 1]
    _setup_log_axes(ax)
    not_att = d[~d["attenuated"]]
    att = d[d["attenuated"]]
    ax.scatter(_logclip(not_att[x]), _logclip(not_att[y]),
               c="0.7", s=3, alpha=0.5, edgecolors="none",
               label=f"normal ($N$={len(not_att):,})", zorder=2, rasterized=True)
    ax.scatter(_logclip(att[x]), _logclip(att[y]),
               c="tab:red", s=6, alpha=0.7, edgecolors="none",
               label=f"attenuated ($N$={len(att):,})", zorder=2, rasterized=True)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9)
    ax.set_title("(d) attenuated profiles ($\\tau$ is lower bound)", pad=6)
    # Reserve right margin so this panel aligns with the colourbarred ones.
    divider = make_axes_locatable(ax)
    spacer = divider.append_axes("right", size="4.5%", pad=0.08)
    spacer.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.0)
    fig.tight_layout()
    return _save(fig, out)


def bias_by_stratum(
    stats: pd.DataFrame,
    *,
    metric: str = "bias",
    title: str = "Bias by stratum",
    out: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of one metric across strata, labelling N per bar."""
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
