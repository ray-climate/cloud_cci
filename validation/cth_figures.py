"""Plotting helpers for cth validation reports.

Linear-axis variants of :mod:`validation.figures`. CTH is approximately
linear on [0, 18] km, so log-clipping (cot convention) would compress
the informative high-cirrus regime.

The scatter panels use a 2D-histogram density plot (`np.histogram2d` +
`imshow` with `LogNorm`) rather than hexbin, for a cleaner
publication-quality look. Colourbars are sized via
`make_axes_locatable` so they exactly match the data-axes height.

Reusable panels:

- :func:`scatter_panel`            : sample + pixel side-by-side density.
- :func:`scatter_compare`          : R10 vs R11 density, single view.
- :func:`scatter_compare_by_surface`: R10 vs R11 × ocean vs land, 2×2.
- :func:`diagnostic_panel`         : 2×2 scatter coloured by lat / dist /
                                      Δt / ATLID cloud_class.
- :func:`bias_by_stratum`          : single-retrieval stratum bar chart.
- :func:`bias_bar_compare`         : R10 vs R11 stratum bar chart.
- :func:`qc_sensitivity_panel`     : bias / RMSE / R / N across QC modes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

CTH_LIM = (0, 18)
CTH_TICKS = (0, 3, 6, 9, 12, 15, 18)
DENSITY_BINS = 60
DENSITY_CMAP = "viridis"


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
    ax.set_xlim(CTH_LIM); ax.set_ylim(CTH_LIM)
    ax.set_xticks(CTH_TICKS); ax.set_yticks(CTH_TICKS)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="in", top=True, right=True, length=4)
    ax.set_xlabel(r"ATLID cloud top height [km]")
    ax.set_ylabel(r"ORAC $\mathrm{cth_{corrected}}$ [km]")
    # 1:1 line drawn on top of the density / scatter layers (zorder 2) so
    # the reference is visible across the dense ridge.
    ax.plot(CTH_LIM, CTH_LIM, color="0.2", lw=0.9, ls="--", zorder=3)


def _density_image(ax, x: np.ndarray, y: np.ndarray,
                   lim: tuple[float, float] = CTH_LIM,
                   bins: int = DENSITY_BINS, cmap: str = DENSITY_CMAP):
    """2D-histogram density on ``ax`` with `LogNorm` colour scale.

    ``lim`` defines the axis range used both for the histogram domain
    and the imshow extent.

    Returns the AxesImage handle so the caller can attach a colourbar.
    Counts of zero are masked → transparent (background shows through).
    """
    H, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[lim, lim]
    )
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
    """Attach a colourbar on the right with the same height as ``ax``."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label)
    cb.ax.tick_params(direction="in", length=3)
    return cb


def _stat_text(ax, n: int, bias: float, rmse: float, r: float, *,
               unit: str = "km", median_bias: float | None = None,
               loc: tuple[float, float] = (0.04, 0.96)) -> None:
    """Stats annotation in the upper-left of ``ax``. ``unit`` is appended to
    bias/RMSE when truthy; pass ``unit=""`` for dimensionless variables. If
    ``median_bias`` is given (heavy-tailed cot/cer), it is shown as the headline
    with the mean flagged as skew-sensitive."""
    suffix = f" {unit}" if unit else ""
    if median_bias is not None:
        bias_lines = (f"median bias = {median_bias:+.2f}{suffix}\n"
                      f"mean bias = {bias:+.2f}{suffix} (skewed)\n")
    else:
        bias_lines = f"bias = {bias:+.2f}{suffix}\n"
    txt = (
        f"$N$ = {n:,}\n"
        f"{bias_lines}"
        f"RMSE = {rmse:.2f}{suffix}\n"
        f"$R$ = {r:.2f}"
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
    x: str = "cth_atlid_thick_km",
    y: str = "cth_orac_corrected_km",
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """Side-by-side density: sample-level vs pixel-aggregate."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
    for ax, d, label in [
        (axes[0], sample, "sample-level (nearest ATLID)"),
        (axes[1], pixel, "pixel-aggregate (mean cloudy ATLID)"),
    ]:
        d2 = d[[x, y]].dropna()
        n, bias, rmse, r = _stats(d2, x, y)
        _setup_axes(ax)
        if n >= 2:
            im = _density_image(ax, d2[x].values, d2[y].values)
            _attach_colorbar(fig, ax, im, label="count")
        _stat_text(ax, n, bias, rmse, r)
        ax.set_title(label, pad=6)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, out)


def diagnostic_panel(
    sample: pd.DataFrame,
    x: str = "cth_atlid_thick_km",
    y: str = "cth_orac_corrected_km",
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """2×2 scatter coloured by lat / dist / Δt / ATLID cloud class."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5))
    cols = [x, y, "ec_lat", "distance_km", "time_diff_s", "cloud_class_atlid"]
    d = sample[cols].dropna(subset=[x, y])
    xv, yv = d[x].values, d[y].values

    # Time-offset colour scale adapts to the collocation window: SEVIRI uses a
    # 7.5-min (450 s) tolerance, but the polar-orbiter SLSTR crossings run to
    # tens of minutes. Cap at the 99th percentile (>=450 s so SEVIRI is unchanged).
    tdiff = d["time_diff_s"].values
    tdiff_vmax = 450.0
    if np.isfinite(tdiff).any():
        tdiff_vmax = max(450.0, float(np.nanpercentile(tdiff, 99)))
    panels = [
        (axes[0, 0], d["ec_lat"].values,        "viridis", -60, 60,
         "latitude [deg]",          "(a) latitude"),
        (axes[0, 1], d["distance_km"].values,    "plasma",   0,  6,
         "match distance [km]",     "(b) match distance"),
        (axes[1, 0], d["time_diff_s"].values,    "cividis",  0, tdiff_vmax,
         r"$|\Delta t|$ [s]",       "(c) time offset"),
    ]
    for ax, c, cmap, vmin, vmax, cb_label, title in panels:
        _setup_axes(ax)
        sc = ax.scatter(xv, yv, c=c, cmap=cmap, s=3, alpha=0.55,
                        vmin=vmin, vmax=vmax, zorder=2,
                        edgecolors="none", rasterized=True)
        _attach_colorbar(fig, ax, sc, label=cb_label)
        ax.set_title(title, pad=6)

    # Cloud-class panel: categorical legend, no colourbar — but match height.
    ax = axes[1, 1]
    _setup_axes(ax)
    class_names = {1: "thick", 2: "thin", 3: "thin/thick",
                   4: "thick/thick", 5: "thin/thin"}
    palette = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c",
               4: "#d62728", 5: "#9467bd"}
    for cls, name in class_names.items():
        sub = d[d["cloud_class_atlid"] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(sub[x], sub[y], s=4, alpha=0.6,
                   c=palette[cls], label=f"{name} ($N$={len(sub):,})",
                   edgecolors="none", zorder=2, rasterized=True)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9)
    ax.set_title("(d) ATLID cloud class", pad=6)
    # Reserve right margin to keep alignment with the colourbarred panels.
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
    """1×2 density scatter, R10 left vs R11 right."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
    for ax, d, label in [(axes[0], d_r10, "R10"), (axes[1], d_r11, "R11")]:
        d2 = d[[x, y]].dropna()
        n, bias, rmse, r = _stats(d2, x, y)
        _setup_axes(ax)
        if n >= 2:
            im = _density_image(ax, d2[x].values, d2[y].values)
            _attach_colorbar(fig, ax, im, label="count")
        _stat_text(ax, n, bias, rmse, r)
        ax.set_title(label, pad=6)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, out)


def scatter_compare_by_surface(
    d_r10: pd.DataFrame,
    d_r11: pd.DataFrame,
    x: str = "cth_atlid_thick_km",
    y: str = "cth_orac_corrected_km",
    lsflag_col: str = "lsflag_orac",
    suptitle: str = "",
    out: str | Path | None = None,
) -> plt.Figure:
    """2×2 density scatter — rows: ocean / land, cols: R10 / R11.

    The land/ocean split uses ORAC ``lsflag_orac`` (0 = sea, 1 = land).
    Each panel reports its own N, bias, RMSE, R in the upper-left box.
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
            d2 = d.loc[mask_fn(d).fillna(False), [x, y]].dropna()
            n, bias, rmse, r = _stats(d2, x, y)
            _setup_axes(ax)
            if n >= 2:
                im = _density_image(ax, d2[x].values, d2[y].values)
                _attach_colorbar(fig, ax, im, label="count")
            _stat_text(ax, n, bias, rmse, r)
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
