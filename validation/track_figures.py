"""Per-orbit case-study figure: SEVIRI cot map + ATLID extinction curtain
+ paired cot_ATLID vs cot_ORAC along the orbit track.

One figure per ATLID frame. Used to verify that the monthly aggregate stats
in :func:`validation.statistics.cot_report` rest on a sensible per-orbit
comparison — at this scale you can see the track land in cloud features and
follow the line-by-line agreement with ORAC.
"""
from __future__ import annotations

from datetime import timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from orac.io import open_slot
from orac.metadata import discover_slots

from .readers import read_accap_track, read_aebd_track
from .reference import cot_cer_water_from_accap

ATLID_VERTICAL_KM = (0.0, 16.0)        # extinction curtain height range
EXT_NORM = LogNorm(vmin=1e-5, vmax=5e-3)  # particle extinction at 355 nm [m^-1]
COT_NORM = LogNorm(vmin=0.5, vmax=100.0)
COT_CMAP = "viridis"
EXT_CMAP = "magma"
MAP_PAD_DEG = 6.0                      # extra degrees around track bbox


def _haversine_km(lat1, lon1, lat2, lon2):
    phi1 = np.deg2rad(lat1); phi2 = np.deg2rad(lat2)
    a = (np.sin((phi2 - phi1) / 2) ** 2
         + np.cos(phi1) * np.cos(phi2)
         * np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _along_track_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cumulative great-circle distance from the first profile."""
    seg = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return np.concatenate([[0.0], np.cumsum(seg)])


def _find_frame_h5(frame_id: str, root: Path = Path("earthcare_data/ATL_EBD_2A")) -> Path:
    matches = list(root.rglob(f"*{frame_id}.h5"))
    if not matches:
        raise FileNotFoundError(f"no A-EBD .h5 found for frame {frame_id} under {root}")
    return matches[0]


def _find_accap_h5(frame_id: str, root: Path = Path("earthcare_data/ACM_CAP_2B")) -> Path:
    matches = list(root.rglob(f"*{frame_id}.h5"))
    if not matches:
        raise FileNotFoundError(f"no ACM-CAP .h5 found for frame {frame_id} under {root}")
    return matches[0]


def _open_seviri_at(scan_time: pd.Timestamp,
                    seviri_root: str | Path,
                    retrieval: str = "R11"):
    """Return (slot, dataset) for the SEVIRI slot at ``scan_time``."""
    slot_dt = pd.Timestamp(scan_time).to_pydatetime().replace(tzinfo=timezone.utc)
    slots = discover_slots(seviri_root,
                           slot_dt - timedelta(minutes=1),
                           slot_dt + timedelta(minutes=1),
                           retrievals=(retrieval,))
    if not slots:
        raise RuntimeError(f"no {retrieval} slot found for {scan_time}")
    return slots[0], open_slot(slots[0], retrieval,
                               variables=("lat", "lon", "cot", "cldmask"))


def track_panel(
    frame_id: str,
    matches: pd.DataFrame,
    seviri_root: str | Path,
    retrieval: str = "R11",
    out: str | Path | None = None,
) -> plt.Figure:
    """Three-panel track figure for one A-EBD frame.

    Panels (top→bottom):
      1. SEVIRI cot map zoomed to the orbit footprint, with the ATLID track
         coloured by per-profile cot_ATLID. Matched SEVIRI pixels overlaid
         in matched-pixel colour for visual consistency.
      2. ATLID particle extinction curtain (height vs along-track distance).
      3. Line plot of cot_ATLID vs cot_ORAC along the track. Attenuated
         columns shaded red, ORAC-saturated cells shaded grey.
    """
    fr = matches[matches["frame_id"] == frame_id].sort_values("ec_time").reset_index(drop=True)
    if fr.empty:
        raise ValueError(f"no matches for frame {frame_id}")
    on = fr[fr["valid_match"]]
    if on.empty:
        raise ValueError(f"no on-disk matches for frame {frame_id}")

    h5 = _find_frame_h5(frame_id)
    track = read_aebd_track(h5)

    # Time-ordered ATLID arrays.
    order = np.argsort(track["time"])
    lat = track["lat"][order]
    lon = track["lon"][order]
    ext = track["extinction"][order]
    height = track["height"][order]
    qs = track["quality_status"][order]
    along = _along_track_km(lat, lon)

    # SEVIRI slot — use the modal scan_time among on-disk matches.
    scan_time = pd.Timestamp(on["sev_scan_time"].mode().iloc[0])
    slot, ds = _open_seviri_at(scan_time, seviri_root, retrieval)

    # Bbox of track + pad.
    lat_min = float(np.nanmin(lat)) - MAP_PAD_DEG
    lat_max = float(np.nanmax(lat)) + MAP_PAD_DEG
    lon_min = float(np.nanmin(lon)) - MAP_PAD_DEG
    lon_max = float(np.nanmax(lon)) + MAP_PAD_DEG

    sl_lat = np.asarray(ds["lat"].values)
    sl_lon = np.asarray(ds["lon"].values)
    sl_cot = np.asarray(ds["cot"].squeeze(drop=True).values)
    sl_cm  = np.asarray(ds["cldmask"].squeeze(drop=True).values)

    in_box = (np.isfinite(sl_lat) & np.isfinite(sl_lon)
              & (sl_lat >= lat_min) & (sl_lat <= lat_max)
              & (sl_lon >= lon_min) & (sl_lon <= lon_max))
    cloudy = in_box & np.isfinite(sl_cot) & (sl_cm == 1)

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=(1.4, 1.0, 0.9), hspace=0.32)

    # ── Row 1: SEVIRI cot map + track ───────────────────────────────────
    # ORAC's lat/lon grid is NaN off-disk, so pcolormesh refuses; scatter
    # the in-bbox pixels instead. ~10⁵–10⁶ points for typical orbit widths.
    ax_map = fig.add_subplot(gs[0])
    pcm = ax_map.scatter(sl_lon[cloudy], sl_lat[cloudy], c=sl_cot[cloudy],
                         cmap=COT_CMAP, norm=COT_NORM, s=4, marker="s",
                         linewidths=0, alpha=0.7)
    fig.colorbar(pcm, ax=ax_map, shrink=0.85, label="ORAC cot (cloudy only)")
    # Track laid on top: black halo + cot-coloured dot for visibility on busy backgrounds.
    ax_map.scatter(fr["ec_lon"], fr["ec_lat"], c="black", s=22, linewidths=0)
    sc = ax_map.scatter(fr["ec_lon"], fr["ec_lat"], c=fr["cot_atlid"],
                        cmap=COT_CMAP, norm=COT_NORM, s=8, linewidths=0)
    ax_map.set_xlim(lon_min, lon_max); ax_map.set_ylim(lat_min, lat_max)
    ax_map.set_xlabel("lon [deg]"); ax_map.set_ylabel("lat [deg]")
    # Per-orbit headline stats over base-filtered subset.
    base = fr[fr["valid_match"] & (fr["cldmask_orac"] == 1) & (fr["cot_atlid"] > 0)
              & (~fr["cot_orac_saturated"]) & fr["cot_atlid"].notna()
              & fr["cot_orac"].notna() & (~fr["attenuated"])]
    if len(base) >= 2:
        diff = base["cot_orac"] - base["cot_atlid"]
        bias = float(diff.mean())
        rmse = float(np.sqrt((diff ** 2).mean()))
        r = float(np.corrcoef(base["cot_atlid"], base["cot_orac"])[0, 1])
        stat_str = f"  bias={bias:+.2f}  RMSE={rmse:.2f}  R={r:.2f}"
    else:
        stat_str = ""
    ax_map.set_title(
        f"frame {frame_id}  scan={scan_time.strftime('%Y-%m-%d %H:%M UTC')}  "
        f"retrieval={retrieval}  N={len(base)}{stat_str}", fontsize=10)
    ax_map.grid(alpha=0.3)

    # ── Row 2: ATLID extinction curtain ─────────────────────────────────
    ax_curt = fig.add_subplot(gs[1])
    ext_show = np.where(qs <= 1, ext, np.nan)  # show only good / likely good
    ALT_GRID = np.arange(ATLID_VERTICAL_KM[0], ATLID_VERTICAL_KM[1] + 0.05, 0.1)
    grid = np.full((len(ALT_GRID) - 1, len(along)), np.nan)
    h_km = height / 1000.0
    for i in range(len(along)):
        valid = np.isfinite(h_km[i]) & np.isfinite(ext_show[i])
        if not valid.any():
            continue
        h_i, e_i = h_km[i, valid], ext_show[i, valid]
        sort = np.argsort(h_i)
        h_i, e_i = h_i[sort], e_i[sort]
        for j in range(len(ALT_GRID) - 1):
            mask = (h_i >= ALT_GRID[j]) & (h_i < ALT_GRID[j + 1])
            if mask.any():
                grid[j, i] = e_i[mask].mean()

    pcm2 = ax_curt.pcolormesh(along, ALT_GRID[:-1], grid, cmap=EXT_CMAP, norm=EXT_NORM,
                              shading="auto", rasterized=True)
    fig.colorbar(pcm2, ax=ax_curt, shrink=0.85,
                 label=r"particle extinction at 355 nm [m$^{-1}$]")
    ax_curt.set_ylim(ATLID_VERTICAL_KM)
    ax_curt.set_xlabel("along-track distance [km]")
    ax_curt.set_ylabel("altitude [km]")
    ax_curt.set_title("ATLID A-EBD curtain (good + likely-good bins)", fontsize=10)

    # ── Row 3: cot_atlid vs cot_orac line ──────────────────────────────
    ax_cmp = fig.add_subplot(gs[2])
    fr_t = fr.copy()
    fr_t["along"] = np.interp(
        np.searchsorted(track["time"][order], pd.to_datetime(fr_t["ec_time"]).values),
        np.arange(len(along)), along, left=along[0], right=along[-1])
    ax_cmp.plot(fr_t["along"], fr_t["cot_atlid"], "k-", lw=0.6, label="ATLID τ₃₅₅")
    ax_cmp.plot(fr_t["along"], fr_t["cot_orac"], color="tab:orange", lw=0.6,
                label="ORAC cot")
    att = fr_t[fr_t["attenuated"]]
    sat = fr_t[fr_t["cot_orac_saturated"] == True]
    if len(att):
        ax_cmp.scatter(att["along"], att["cot_atlid"], s=8, c="tab:red",
                       label=f"attenuated (N={len(att)})", zorder=4)
    if len(sat):
        ax_cmp.scatter(sat["along"], sat["cot_orac"], s=10, marker="x",
                       c="0.4", label=f"ORAC saturated (N={len(sat)})", zorder=4)
    ax_cmp.set_yscale("log")
    ax_cmp.set_ylim(0.05, 200)
    ax_cmp.set_xlabel("along-track distance [km]")
    ax_cmp.set_ylabel("column τ")
    ax_cmp.legend(loc="upper right", fontsize=8, ncol=2)
    ax_cmp.set_title("paired column optical depth along the orbit", fontsize=10)
    ax_cmp.grid(alpha=0.3)

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Synergy (ACM-CAP) case-study panel
# ---------------------------------------------------------------------------

# Phase classification colours
_PHASE_COLORS = {
    "clear":       "#cccccc",
    "liquid_only": "#1f77b4",   # blue
    "mixed":       "#9467bd",   # purple
    "ice_only":    "#d62728",   # red
}

LIQ_EXT_NORM = LogNorm(vmin=1e-5, vmax=1e-2)  # m^-1
IWC_NORM = LogNorm(vmin=1e-7, vmax=1e-3)      # kg m^-3
WATER_COT_NORM = LogNorm(vmin=0.3, vmax=100.0)


def _phase_class_per_profile(liq_p: np.ndarray, ice_p: np.ndarray) -> np.ndarray:
    """Return a 1-D array of phase-class strings per profile."""
    out = np.full(liq_p.shape, "clear", dtype=object)
    out[liq_p & ~ice_p] = "liquid_only"
    out[liq_p & ice_p]  = "mixed"
    out[~liq_p & ice_p] = "ice_only"
    return out


def track_panel_synergy(
    frame_id: str,
    matches: pd.DataFrame,
    seviri_root: str | Path,
    retrieval: str = "R11",
    out: str | Path | None = None,
) -> plt.Figure:
    """Three-panel case-study figure for one ACM-CAP frame.

    Panels (top → bottom):
      1. SEVIRI cot map zoomed to the ACM-CAP nadir track. Track points
         coloured by per-profile phase classification (liquid_only / mixed
         / ice_only / clear) — shows where ACM-CAP says the cloud is liquid.
      2. ACM-CAP liquid extinction curtain + ice-water-content overlay.
         Reveals how much of the column has lidar+radar signal vs not.
      3. Along-track line plot: ACM-CAP liquid_optical_depth vs ORAC cot
         at the matched SEVIRI pixel. Phase classification shaded along
         the bottom. Where bias is large, you can see the lines diverge
         and read off whether it's a phase, footprint, or detection issue.
    """
    fr = matches[matches["frame_id"] == frame_id].sort_values("ec_time").reset_index(drop=True)
    if fr.empty:
        raise ValueError(f"no matches for frame {frame_id}")
    on = fr[fr["valid_match"]]
    if on.empty:
        raise ValueError(f"no on-disk matches for frame {frame_id}")

    h5 = _find_accap_h5(frame_id)
    track = read_accap_track(h5)
    cot_w, cer_w, liq_p, ice_p = cot_cer_water_from_accap(
        track["liquid_optical_depth"], track["liquid_extinction"],
        track["liquid_eff_radius"], track["liquid_classification"],
        track["ice_water_content"], track["height"],
    )
    phase = _phase_class_per_profile(liq_p, ice_p)

    order = np.argsort(track["time"])
    lat = track["lat"][order]
    lon = track["lon"][order]
    liq_ext = track["liquid_extinction"][order]
    iwc = track["ice_water_content"][order]
    height = track["height"][order]
    cot_w_o = cot_w[order]
    phase_o = phase[order]
    along = _along_track_km(lat, lon)

    scan_time = pd.Timestamp(on["sev_scan_time"].mode().iloc[0])
    slot, ds = _open_seviri_at(scan_time, seviri_root, retrieval)

    lat_min = float(np.nanmin(lat)) - MAP_PAD_DEG
    lat_max = float(np.nanmax(lat)) + MAP_PAD_DEG
    lon_min = float(np.nanmin(lon)) - MAP_PAD_DEG
    lon_max = float(np.nanmax(lon)) + MAP_PAD_DEG

    sl_lat = np.asarray(ds["lat"].values)
    sl_lon = np.asarray(ds["lon"].values)
    sl_cot = np.asarray(ds["cot"].squeeze(drop=True).values)
    sl_cm  = np.asarray(ds["cldmask"].squeeze(drop=True).values)
    in_box = (np.isfinite(sl_lat) & np.isfinite(sl_lon)
              & (sl_lat >= lat_min) & (sl_lat <= lat_max)
              & (sl_lon >= lon_min) & (sl_lon <= lon_max))
    cloudy = in_box & np.isfinite(sl_cot) & (sl_cm == 1)

    fig = plt.figure(figsize=(12, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=(1.4, 1.0, 0.95), hspace=0.35)

    # ── Row 1: SEVIRI cot map + ATLID nadir track coloured by phase ─────
    ax_map = fig.add_subplot(gs[0])
    pcm = ax_map.scatter(sl_lon[cloudy], sl_lat[cloudy], c=sl_cot[cloudy],
                         cmap=COT_CMAP, norm=COT_NORM, s=4, marker="s",
                         linewidths=0, alpha=0.65)
    fig.colorbar(pcm, ax=ax_map, shrink=0.85, label="ORAC cot (cloudy only)")
    # Black halo for visibility, then phase-coloured dots on top.
    ax_map.scatter(lon, lat, c="black", s=18, linewidths=0)
    for cls, col in _PHASE_COLORS.items():
        m = phase_o == cls
        if m.any():
            ax_map.scatter(lon[m], lat[m], c=col, s=6, linewidths=0,
                           label=f"{cls} (N={int(m.sum()):,})")
    ax_map.legend(loc="lower left", fontsize=8, framealpha=0.85,
                  markerscale=2)
    ax_map.set_xlim(lon_min, lon_max); ax_map.set_ylim(lat_min, lat_max)
    ax_map.set_xlabel("lon [deg]"); ax_map.set_ylabel("lat [deg]")
    base = fr[fr["valid_match"] & (fr["cldmask_orac"] == 1)
              & fr["cot_water_atlid"].notna() & fr["cot_orac"].notna()].copy()
    if "quality_status_atlid" in base.columns:
        headline = base[base["liquid_only_atlid"].astype(bool)
                        & (base["quality_status_atlid"] == 0)]
    else:
        headline = base
    if len(headline) >= 2:
        diff = headline["cot_orac"] - headline["cot_water_atlid"]
        bias = float(diff.mean())
        rmse = float(np.sqrt((diff ** 2).mean()))
        r = float(np.corrcoef(headline["cot_water_atlid"], headline["cot_orac"])[0, 1])
        stat_str = f"  bias={bias:+.2f}  RMSE={rmse:.2f}  R={r:.2f}"
    else:
        stat_str = ""
    ax_map.set_title(
        f"frame {frame_id}  scan={scan_time.strftime('%Y-%m-%d %H:%M UTC')}  "
        f"retrieval={retrieval}  N_liquid_only={len(headline)}{stat_str}",
        fontsize=10)
    ax_map.grid(alpha=0.3)

    # ── Row 2: ACM-CAP liquid extinction curtain + IWC overlay ──────────
    ax_curt = fig.add_subplot(gs[1])
    ALT_GRID = np.arange(ATLID_VERTICAL_KM[0], ATLID_VERTICAL_KM[1] + 0.05, 0.1)
    h_km = height / 1000.0
    grid_liq = np.full((len(ALT_GRID) - 1, len(along)), np.nan)
    grid_ice = np.full((len(ALT_GRID) - 1, len(along)), np.nan)
    for i in range(len(along)):
        valid_h = np.isfinite(h_km[i])
        if not valid_h.any():
            continue
        h_i = h_km[i][valid_h]
        e_i = liq_ext[i][valid_h]
        iwc_i = iwc[i][valid_h]
        sort = np.argsort(h_i)
        h_i, e_i, iwc_i = h_i[sort], e_i[sort], iwc_i[sort]
        for j in range(len(ALT_GRID) - 1):
            mask = (h_i >= ALT_GRID[j]) & (h_i < ALT_GRID[j + 1])
            if mask.any():
                vals_e = e_i[mask][np.isfinite(e_i[mask]) & (e_i[mask] > 0)]
                if len(vals_e):
                    grid_liq[j, i] = vals_e.mean()
                vals_iwc = iwc_i[mask][np.isfinite(iwc_i[mask]) & (iwc_i[mask] > 0)]
                if len(vals_iwc):
                    grid_ice[j, i] = vals_iwc.mean()

    pcm2 = ax_curt.pcolormesh(along, ALT_GRID[:-1], grid_liq, cmap="Blues",
                              norm=LIQ_EXT_NORM, shading="auto", rasterized=True)
    fig.colorbar(pcm2, ax=ax_curt, shrink=0.85,
                 label=r"liquid extinction [m$^{-1}$]")
    # IWC contour overlay — outline of ice regions in red.
    has_ice = np.isfinite(grid_ice) & (grid_ice > 1e-7)
    if has_ice.any():
        ax_curt.contour(along, ALT_GRID[:-1], has_ice.astype(int),
                        levels=[0.5], colors=["#d62728"], linewidths=0.8,
                        linestyles="-")
    ax_curt.set_ylim(ATLID_VERTICAL_KM)
    ax_curt.set_xlabel("along-track distance [km]")
    ax_curt.set_ylabel("altitude [km]")
    ax_curt.set_title(
        "ACM-CAP liquid extinction (blue fill); ice-present mask (red outline)",
        fontsize=10)

    # ── Row 3: along-track ACM-CAP τ vs ORAC cot ────────────────────────
    ax_cmp = fig.add_subplot(gs[2])
    fr_t = fr.copy()
    fr_t["along"] = np.interp(
        np.searchsorted(track["time"][order], pd.to_datetime(fr_t["ec_time"]).values),
        np.arange(len(along)), along, left=along[0], right=along[-1])
    ax_cmp.plot(fr_t["along"], fr_t["cot_water_atlid"], "k-", lw=0.7,
                label="ACM-CAP liquid τ")
    ax_cmp.plot(fr_t["along"], fr_t["cot_orac"], color="tab:orange", lw=0.7,
                label="ORAC cot")
    # Phase classification shaded along the bottom.
    cpr_used = (fr_t["cpr_assim_status"] == 0) if "cpr_assim_status" in fr_t.columns else pd.Series(False, index=fr_t.index)
    if cpr_used.any():
        ax_cmp.scatter(fr_t.loc[cpr_used, "along"],
                       np.full(cpr_used.sum(), 0.07),
                       s=4, c="#2ca02c", marker="^",
                       label=f"CPR-assimilated (N={int(cpr_used.sum()):,})")
    ax_cmp.set_yscale("log")
    ax_cmp.set_ylim(0.05, 200)
    ax_cmp.set_xlabel("along-track distance [km]")
    ax_cmp.set_ylabel(r"column $\tau$")
    ax_cmp.legend(loc="upper right", fontsize=8, ncol=2)
    ax_cmp.set_title("paired column τ along the orbit (water-cloud)", fontsize=10)
    ax_cmp.grid(alpha=0.3)

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig
