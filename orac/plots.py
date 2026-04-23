"""Publication-quality full-disk plots of ORAC SEVIRI cloud products.

All plots are drawn in the SEVIRI geostationary projection via Cartopy, so the
native 3712×3712 grid can be rendered directly with ``imshow`` (fast) while
coastlines land in the right place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except ImportError:  # pragma: no cover
    _HAS_CARTOPY = False

from .flags import qc_pass_mask

# SEVIRI L1.5 full-disk geostationary geometry.
SEVIRI_SAT_HEIGHT = 35785831.0  # m above Earth surface
SEVIRI_EXTENT = (-5568000.0, 5568000.0, -5568000.0, 5568000.0)  # ±5568 km at nadir


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def set_publication_style() -> None:
    """Apply a compact publication-quality rc block (call once per session)."""
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["DejaVu Serif", "Nimbus Roman", "Times New Roman"],
        "mathtext.fontset":   "dejavuserif",
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "axes.labelsize":     10,
        "axes.linewidth":     0.8,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "legend.fontsize":     9,
        "legend.frameon":     False,
        "figure.titlesize":    13,
        "figure.titleweight": "bold",
        "figure.dpi":          110,
        "savefig.dpi":         300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches":  0.08,
        "pdf.fonttype":        42,
        "ps.fonttype":         42,
    })


# ---------------------------------------------------------------------------
# Per-product display specs
# ---------------------------------------------------------------------------
# Each entry carries the colormap, normalisation, and colorbar label used when
# plotting that retrieved quantity. Ranges follow community-typical SEVIRI
# quicklook conventions.
PRODUCT_SPECS: dict[str, dict] = {
    "cot":          dict(cmap="viridis",    norm=LogNorm(vmin=0.5, vmax=100),
                         label=r"Cloud optical thickness",                    short="COT"),
    "cer":          dict(cmap="magma",      norm=Normalize(vmin=2, vmax=35),
                         label=r"Cloud effective radius [$\mu$m]",            short="CER"),
    "ctp":          dict(cmap="plasma_r",   norm=Normalize(vmin=100, vmax=1000),
                         label=r"Cloud top pressure [hPa]",                   short="CTP"),
    "ctp_corrected":dict(cmap="plasma_r",   norm=Normalize(vmin=100, vmax=1000),
                         label=r"Corrected cloud top pressure [hPa]",         short="CTP$_{corr}$"),
    "cth":          dict(cmap="cividis",    norm=Normalize(vmin=0, vmax=16),
                         label=r"Cloud top height [km]",                      short="CTH"),
    "cth_corrected":dict(cmap="cividis",    norm=Normalize(vmin=0, vmax=16),
                         label=r"Corrected cloud top height [km]",            short="CTH$_{corr}$"),
    "ctt":          dict(cmap="RdYlBu_r",   norm=Normalize(vmin=200, vmax=300),
                         label=r"Cloud top temperature [K]",                  short="CTT"),
    "cwp":          dict(cmap="inferno",    norm=LogNorm(vmin=5, vmax=2000),
                         label=r"Cloud liquid water path [g m$^{-2}$]",       short="CWP"),
    "cc_total":     dict(cmap="Greys",      norm=Normalize(vmin=0, vmax=1),
                         label=r"Cloud fraction",                             short="CF"),
}

# Discrete/categorical products
PHASE_COLORS  = {1: "#1f77b4", 2: "#d62728"}                                 # liquid, ice
PHASE_LABELS  = {1: "Liquid water", 2: "Ice / water-ice-agg"}

CLDTYPE_PALETTE = {  # Pavolonis categories
    0:  ("#eeeeee", "Clear"),
    2:  ("#b5d7ff", "Fog"),
    3:  ("#1f77b4", "Water cloud"),
    4:  ("#6baed6", "Supercooled"),
    5:  ("#9e9ac8", "Mixed"),
    6:  ("#d62728", "Opaque ice"),
    7:  ("#ff7f0e", "Cirrus"),
    8:  ("#8c564b", "Overlap"),
    9:  ("#e377c2", "Prob opaque ice"),
    11: ("#bcbd22", "Dust (clear)"),
    12: ("#7f7f7f", "Dust (switched)"),
}


# ---------------------------------------------------------------------------
# Axes / base plot
# ---------------------------------------------------------------------------
def _make_axes(fig: plt.Figure, pos=(1, 1, 1)) -> plt.Axes:
    """Create an axis in SEVIRI geostationary projection (or plain if no cartopy)."""
    if _HAS_CARTOPY:
        proj = ccrs.Geostationary(
            central_longitude=0.0,
            satellite_height=SEVIRI_SAT_HEIGHT,
            sweep_axis="y",
        )
        ax = fig.add_subplot(*pos, projection=proj)
        ax.set_global()
        ax.coastlines(resolution="50m", linewidth=0.35, color="0.25")
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                       linewidth=0.25, edgecolor="0.45")
        ax.gridlines(draw_labels=False, linewidth=0.25, color="0.55", alpha=0.35)
    else:  # pragma: no cover
        ax = fig.add_subplot(*pos)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    return ax


def _select_and_mask(
    ds: xr.Dataset,
    var: str,
    cloud_only: bool = True,
    qc_rules: str | None = "default",
) -> np.ndarray:
    """Extract a 2-D array for plotting, with QC / cloud-only masking applied."""
    arr = np.asarray(ds[var].squeeze(drop=True).values, dtype="float32")
    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    on_disk = np.isfinite(lat) & np.isfinite(lon)

    mask = on_disk
    if cloud_only and "cldmask" in ds.variables:
        cm = np.asarray(ds["cldmask"].squeeze(drop=True).values)
        mask = mask & (cm == 1)
    if qc_rules is not None and "qcflag" in ds.variables:
        mask = mask & np.asarray(qc_pass_mask(ds["qcflag"], rules=qc_rules).values)

    out = np.full_like(arr, np.nan, dtype="float32")
    out[mask] = arr[mask]
    return out


def plot_full_disk(
    ds: xr.Dataset,
    var: str,
    ax: plt.Axes | None = None,
    *,
    cloud_only: bool = True,
    qc_rules: str | None = "default",
    title: str | None = None,
    show_cbar: bool = True,
    spec: Mapping | None = None,
) -> plt.Axes:
    """Draw a single full-disk map of ``var`` onto ``ax`` (created if None)."""
    if ax is None:
        fig = plt.figure(figsize=(7.0, 7.2))
        ax = _make_axes(fig)
    spec = dict(spec or PRODUCT_SPECS[var])

    data = _select_and_mask(ds, var, cloud_only=cloud_only, qc_rules=qc_rules)

    im_kw: dict = dict(
        extent=SEVIRI_EXTENT,
        origin="upper",
        interpolation="nearest",
        cmap=spec["cmap"],
        norm=spec["norm"],
    )
    if _HAS_CARTOPY:
        im_kw["transform"] = ccrs.Geostationary(
            central_longitude=0.0,
            satellite_height=SEVIRI_SAT_HEIGHT,
            sweep_axis="y",
        )
    im = ax.imshow(data, **im_kw)

    if title is None:
        ts = ds.attrs.get("scan_time", "")
        retrieval = ds.attrs.get("retrieval", "")
        title = f"{spec['short']} — {retrieval}  {ts[:16].replace('T', ' ')} UTC"
    ax.set_title(title, loc="left")

    if show_cbar:
        cbar = plt.colorbar(im, ax=ax, orientation="vertical",
                            shrink=0.75, pad=0.03, extend="both")
        cbar.set_label(spec["label"])
        cbar.ax.tick_params(labelsize=9, length=3, width=0.6)
        cbar.outline.set_linewidth(0.6)

    return ax


# ---------------------------------------------------------------------------
# Categorical plots
# ---------------------------------------------------------------------------
def plot_phase(
    ds: xr.Dataset,
    ax: plt.Axes | None = None,
    qc_rules: str | None = "default",
    title: str | None = None,
) -> plt.Axes:
    """Discrete liquid/ice phase map with legend."""
    if ax is None:
        fig = plt.figure(figsize=(7.0, 7.2))
        ax = _make_axes(fig)
    arr = np.asarray(ds["phase"].squeeze(drop=True).values)
    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    on_disk = np.isfinite(lat) & np.isfinite(lon)
    mask = on_disk
    if qc_rules is not None and "qcflag" in ds.variables:
        mask = mask & np.asarray(qc_pass_mask(ds["qcflag"], rules=qc_rules).values)

    display = np.full_like(arr, -1, dtype="int8")
    display[mask] = arr[mask].astype("int8")
    rgb = np.zeros((*display.shape, 4), dtype="float32")
    for val, col in PHASE_COLORS.items():
        hex_rgb = tuple(int(col[i:i+2], 16) / 255 for i in (1, 3, 5))
        rgb[display == val] = (*hex_rgb, 1.0)

    if _HAS_CARTOPY:
        ax.imshow(rgb, extent=SEVIRI_EXTENT, origin="upper",
                  transform=ccrs.Geostationary(central_longitude=0.0,
                                               satellite_height=SEVIRI_SAT_HEIGHT,
                                               sweep_axis="y"),
                  interpolation="nearest")
    else:
        ax.imshow(rgb, origin="upper", interpolation="nearest")

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[v], label=PHASE_LABELS[v])
               for v in PHASE_COLORS]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=9,
              facecolor="white", edgecolor="0.4", framealpha=0.9)

    if title is None:
        ts = ds.attrs.get("scan_time", "")
        retrieval = ds.attrs.get("retrieval", "")
        title = f"Cloud phase — {retrieval}  {ts[:16].replace('T', ' ')} UTC"
    ax.set_title(title, loc="left")
    return ax


def plot_cldtype(
    ds: xr.Dataset,
    ax: plt.Axes | None = None,
    qc_rules: str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Discrete Pavolonis cloud-type map with legend."""
    if ax is None:
        fig = plt.figure(figsize=(7.5, 7.2))
        ax = _make_axes(fig)
    arr = np.asarray(ds["cldtype"].squeeze(drop=True).values)
    lat = np.asarray(ds["lat"].values)
    lon = np.asarray(ds["lon"].values)
    on_disk = np.isfinite(lat) & np.isfinite(lon)

    rgb = np.zeros((*arr.shape, 4), dtype="float32")
    for code, (col, _) in CLDTYPE_PALETTE.items():
        hex_rgb = tuple(int(col[i:i+2], 16) / 255 for i in (1, 3, 5))
        m = on_disk & (arr.astype(int) == code)
        rgb[m] = (*hex_rgb, 1.0)

    if _HAS_CARTOPY:
        ax.imshow(rgb, extent=SEVIRI_EXTENT, origin="upper",
                  transform=ccrs.Geostationary(central_longitude=0.0,
                                               satellite_height=SEVIRI_SAT_HEIGHT,
                                               sweep_axis="y"),
                  interpolation="nearest")
    else:
        ax.imshow(rgb, origin="upper", interpolation="nearest")

    handles = [plt.Rectangle((0, 0), 1, 1, color=col, label=name)
               for col, name in CLDTYPE_PALETTE.values()]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=8, title="Pavolonis type",
              title_fontsize=9)

    if title is None:
        ts = ds.attrs.get("scan_time", "")
        retrieval = ds.attrs.get("retrieval", "")
        title = f"Cloud type — {retrieval}  {ts[:16].replace('T', ' ')} UTC"
    ax.set_title(title, loc="left")
    return ax


# ---------------------------------------------------------------------------
# Multi-panel layout
# ---------------------------------------------------------------------------
def plot_product_suite(
    ds: xr.Dataset,
    variables: tuple[str, ...] = ("cot", "cer", "ctp", "cth", "ctt", "cwp"),
    ncols: int = 3,
    qc_rules: str | None = "default",
    cloud_only: bool = True,
    suptitle: str | None = None,
) -> plt.Figure:
    """3×2 panel of the six headline cloud products."""
    nrows = int(np.ceil(len(variables) / ncols))
    fig = plt.figure(figsize=(5.6 * ncols, 5.7 * nrows))
    for i, var in enumerate(variables):
        ax = _make_axes(fig, pos=(nrows, ncols, i + 1))
        plot_full_disk(ds, var, ax=ax,
                       cloud_only=cloud_only, qc_rules=qc_rules)
    if suptitle is None:
        ts = ds.attrs.get("scan_time", "")
        retrieval = ds.attrs.get("retrieval", "")
        suptitle = f"ORAC SEVIRI MSG-3  —  {retrieval}  —  {ts[:16].replace('T', ' ')} UTC"
    fig.suptitle(suptitle, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


# ---------------------------------------------------------------------------
# Convenience saver
# ---------------------------------------------------------------------------
def save_figure(fig: plt.Figure, path: str | Path, also_pdf: bool = True) -> list[Path]:
    """Write PNG (always) and PDF (optional). Returns list of written paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [path.with_suffix(".png")]
    fig.savefig(out[-1])
    if also_pdf:
        out.append(path.with_suffix(".pdf"))
        fig.savefig(out[-1])
    return out
