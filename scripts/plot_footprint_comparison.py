"""EarthCARE ATLID x SEVIRI footprint-scale demonstration.

Motivates the sample-level (one-row-per-ATLID-profile) collocation protocol
used in the ORAC x EarthCARE validation module by contrasting the native
sampling scales of the two instruments on a real overpass.

Three panels:

  (a) Full SEVIRI disk with the ATLID nadir track overlaid and the zoom box
      marked -- orbit-scale context.
  (b) 40 km x 40 km zoom near 45 deg N, 30 deg W showing SEVIRI pixel
      footprints (grey boxes) and ATLID nadir samples (red dots) -- the scale
      mismatch.
  (c) Histogram of the number of ATLID samples falling into each SEVIRI pixel
      traversed by the track -- quantifies the ratio that drives the
      collocation design.

Uses A-CTH for the ATLID geometry (the nadir track is identical across all
ATLID L2 products, so this figure is representative for cot/A-EBD too).
SEVIRI geometry is geostationary and time-invariant, so a 2026-02 SEVIRI
grid is used here even though the ATLID frame is from 2024-12-01.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

ATLID_FILE = Path(
    "earthcare_data/ATL_CTH_2A/2026/02/01/"
    "ECA_EXBC_ATL_CTH_2A_20260201T030940Z_20260201T035115Z_09541B.h5"
)
FRAME_ID = "09541B"
FRAME_DATE = "2026-02-01"
SEVIRI_FILE = Path(
    "/gws/ssde/j25a/cloud_ecv/data_out/seviri/2026/02/01/0300/"
    "ESACCI-L2-CLOUD-CLD-SEVIRI_ORAC_MSG3_202602010312_R11.primary.nc"
)
OUT = Path("figures/validation/footprint_scale_comparison.png")

# Zoom target latitude — actual centre is picked on the ATLID track at this lat
# so the track passes through the centre of the panel. SEVIRI pixels there are
# ~5-6 km along-track, a good contrast with ATLID's ~1 km sampling.
ZOOM_TARGET_LAT = 45.0
ZOOM_HALF_KM = 20.0  # 40 km square

# Representative SEVIRI pixel size at the zoom latitude (measured from the
# actual neighbour spacing in the SEVIRI file).
SEVIRI_PIXEL_AT_KM = 5.9   # along-track (MSG "north-south")
SEVIRI_PIXEL_AC_KM = 4.0   # across-track


def load_atlid():
    with h5py.File(ATLID_FILE, "r") as f:
        sd = f["ScienceData"]
        return sd["latitude"][:], sd["longitude"][:]


def load_seviri():
    ds = xr.open_dataset(SEVIRI_FILE, decode_times=False, mask_and_scale=True)
    return ds["lat"].values, ds["lon"].values


def count_atlid_per_seviri(atlid_lat, atlid_lon, sev_lat, sev_lon):
    """For each on-disk ATLID sample, find its nearest SEVIRI pixel.
    Returns an array of counts (one entry per unique SEVIRI pixel hit)."""
    finite = np.isfinite(sev_lat) & np.isfinite(sev_lon)
    lat_flat = sev_lat[finite]
    lon_flat = sev_lon[finite]

    # Cheap approx-Cartesian metric; good enough within <75 deg lat for
    # nearest-neighbour queries because we only need relative distances locally.
    mean_lat = np.nanmean(atlid_lat)
    scale = np.cos(np.radians(mean_lat))
    tree = cKDTree(np.column_stack([lat_flat, lon_flat * scale]))

    on_disk = (np.abs(atlid_lat) < 75) & (np.abs(atlid_lon) < 75)
    pts = np.column_stack([atlid_lat[on_disk], atlid_lon[on_disk] * scale])
    _, nearest = tree.query(pts, k=1)
    return np.asarray(list(Counter(nearest.tolist()).values()))


def zoom_window(centre_lat, centre_lon):
    dlat = ZOOM_HALF_KM / 111.0
    dlon = ZOOM_HALF_KM / (111.0 * np.cos(np.radians(centre_lat)))
    return (centre_lat - dlat, centre_lat + dlat,
            centre_lon - dlon, centre_lon + dlon)


def pick_zoom_centre(atlid_lat, atlid_lon):
    """Pick the ATLID sample whose latitude is closest to ZOOM_TARGET_LAT so the
    zoom window is centred exactly on the track."""
    idx = int(np.nanargmin(np.abs(atlid_lat - ZOOM_TARGET_LAT)))
    return float(atlid_lat[idx]), float(atlid_lon[idx])


def main():
    atlid_lat, atlid_lon = load_atlid()
    sev_lat, sev_lon = load_seviri()

    counts = count_atlid_per_seviri(atlid_lat, atlid_lon, sev_lat, sev_lon)

    zoom_lat, zoom_lon = pick_zoom_centre(atlid_lat, atlid_lon)
    lat_lo, lat_hi, lon_lo, lon_hi = zoom_window(zoom_lat, zoom_lon)
    in_zoom_sev = np.isfinite(sev_lat) & np.isfinite(sev_lon) \
        & (sev_lat >= lat_lo) & (sev_lat <= lat_hi) \
        & (sev_lon >= lon_lo) & (sev_lon <= lon_hi)
    sev_lat_z = sev_lat[in_zoom_sev]
    sev_lon_z = sev_lon[in_zoom_sev]

    in_zoom_at = (atlid_lat >= lat_lo) & (atlid_lat <= lat_hi) \
        & (atlid_lon >= lon_lo) & (atlid_lon <= lon_hi)
    at_lat_z = atlid_lat[in_zoom_at]
    at_lon_z = atlid_lon[in_zoom_at]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 6.0), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 0.9],
                          left=0.05, right=0.98, top=0.91, bottom=0.16,
                          wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # --- (a) context ---
    theta = np.linspace(0, 2 * np.pi, 400)
    ax_a.fill(66 * np.cos(theta), 66 * np.sin(theta),
              color="#eef2f6", zorder=0)
    ax_a.plot(66 * np.cos(theta), 66 * np.sin(theta),
              color="0.5", lw=0.8, zorder=1)
    # Tiny subsampled SEVIRI pixel hint (every 50th pixel, on-disk only)
    step = 50
    subsmp = np.isfinite(sev_lat[::step, ::step])
    ax_a.scatter(sev_lon[::step, ::step][subsmp],
                 sev_lat[::step, ::step][subsmp],
                 s=0.3, color="0.75", zorder=1)
    ax_a.plot(atlid_lon, atlid_lat, color="tab:red", lw=1.4,
              label="ATLID nadir track", zorder=4)
    ax_a.add_patch(mpatches.Rectangle(
        (lon_lo, lat_lo), lon_hi - lon_lo, lat_hi - lat_lo,
        fill=False, edgecolor="black", lw=1.4, zorder=5
    ))
    ax_a.annotate("zoom (b)", xy=(lon_hi + 2, lat_hi),
                  fontsize=9, zorder=6)
    ax_a.set_xlabel("longitude [deg]")
    ax_a.set_ylabel("latitude [deg]")
    ax_a.set_xlim(-82, 82)
    ax_a.set_ylim(-82, 82)
    ax_a.set_aspect("equal")
    ax_a.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax_a.set_title(f"(a) SEVIRI disk + ATLID frame {FRAME_ID} ({FRAME_DATE})")

    # --- (b) zoom ---
    for la, lo in zip(sev_lat_z, sev_lon_z):
        hx = (SEVIRI_PIXEL_AC_KM / 2) / (111.0 * np.cos(np.radians(la)))
        hy = (SEVIRI_PIXEL_AT_KM / 2) / 111.0
        ax_b.add_patch(mpatches.Rectangle(
            (lo - hx, la - hy), 2 * hx, 2 * hy,
            facecolor="#d8dee6", edgecolor="0.35", lw=0.4, zorder=1
        ))
    ax_b.scatter(sev_lon_z, sev_lat_z, s=6, color="0.3", zorder=2,
                 label="SEVIRI pixel centres")
    ax_b.plot(at_lon_z, at_lat_z, "-", color="tab:red", lw=0.7,
              alpha=0.6, zorder=3)
    ax_b.scatter(at_lon_z, at_lat_z, s=9, color="tab:red", zorder=4,
                 label="ATLID nadir samples")
    ax_b.set_xlim(lon_lo, lon_hi)
    ax_b.set_ylim(lat_lo, lat_hi)
    ax_b.set_xlabel("longitude [deg]")
    ax_b.set_ylabel("latitude [deg]")
    ax_b.set_aspect("equal")
    ax_b.set_title(
        f"(b) 40 km zoom centred on track at {zoom_lat:.2f}°N, "
        f"{abs(zoom_lon):.2f}°W"
    )
    ax_b.legend(loc="upper right", fontsize=8, framealpha=0.95)

    # scale bar: 10 km
    bar_km = 10.0
    bar_dlon = bar_km / (111.0 * np.cos(np.radians(zoom_lat)))
    bar_y = lat_lo + 0.08 * (lat_hi - lat_lo)
    bar_x0 = lon_lo + 0.08 * (lon_hi - lon_lo)
    ax_b.plot([bar_x0, bar_x0 + bar_dlon], [bar_y, bar_y], "k-", lw=2.5)
    ax_b.text(bar_x0 + bar_dlon / 2, bar_y + 0.015 * (lat_hi - lat_lo),
              "10 km", ha="center", fontsize=9)

    # Footprint info: placed below panel (b), outside the axes, so the plot
    # area itself stays uncluttered.
    ax_b.text(
        0.5, -0.22,
        "SEVIRI pixel ≈ 5.9 × 4.0 km (at 45°N)      "
        "ATLID sample ≈ 1 km along-track,  ~30 m wide",
        transform=ax_b.transAxes, fontsize=9, ha="center", va="top",
        bbox=dict(facecolor="#f4f4f4", edgecolor="0.6", pad=4)
    )

    # --- (c) histogram ---
    bins = np.arange(0.5, counts.max() + 1.5, 1)
    ax_c.hist(counts, bins=bins, color="tab:red", alpha=0.8,
              edgecolor="white")
    median = int(np.median(counts))
    ymax = ax_c.get_ylim()[1]
    ax_c.axvline(median, color="k", ls="--", lw=1.0)
    ax_c.text(median + 0.4, ymax * 0.96, f"median = {median}",
              fontsize=9, va="top",
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax_c.set_xlabel("ATLID samples per SEVIRI pixel")
    ax_c.set_ylabel("SEVIRI pixels hit")
    ax_c.set_title(
        f"(c) {len(counts)} SEVIRI pixels hit along track\n"
        f"mean {counts.mean():.1f} samples/pixel"
    )
    ax_c.set_xlim(0, max(12, counts.max() + 1))

    fig.suptitle(
        "ATLID × SEVIRI footprint-scale comparison — "
        "motivation for sample-level collocation",
        fontsize=12, y=0.97,
    )
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"track: {len(atlid_lat)} ATLID samples, "
          f"{len(counts)} SEVIRI pixels hit "
          f"(median {median}, max {counts.max()})")


if __name__ == "__main__":
    main()
