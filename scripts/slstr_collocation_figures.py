"""Presentation figures for the SLSTR x ATLID (EarthCARE) collocation strategy.

Produces three PNGs under figures/slstr_collocation/:

1. collocation_map_polar.png  - where the matches happen (N & S polar stereo).
   The single figure that shows SLSTR x ATLID is a polar comparison.
2. match_quality.png          - nearest-pixel distance and |Dt| distributions,
   proving the crossings are spatially tight.
3. crossing_case_study.png    - one SLSTR swath (cloud-top) with the ATLID nadir
   track threading across it: the mechanics of a single collocation.

Reads the CTH matches CSVs (small) plus, for the case study, one SLSTR granule
and one A-CTH frame.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MATCH_GLOB = "validation_data/slstr_cth_2025-12/matches_cth_*.csv"
OUT = Path("figures/slstr_collocation")
OUT.mkdir(parents=True, exist_ok=True)
CASE_FRAME = "08642G"   # 2130 matches, ~ -78 deg, Antarctic; strong crossing


def _load_all_matches(cols):
    paths = sorted(glob.glob(MATCH_GLOB))
    parts = []
    for p in paths:
        try:
            d = pd.read_csv(p, usecols=lambda c: c in cols)
            parts.append(d)
        except Exception:
            continue
    return pd.concat(parts, ignore_index=True), len(paths)


# ---------------------------------------------------------------------------
# 1. Polar collocation map
# ---------------------------------------------------------------------------
def collocation_map():
    df, nfr = _load_all_matches(["valid_match", "ec_lat", "ec_lon"])
    v = df[df["valid_match"] == True]
    lat, lon = v["ec_lat"].values, v["ec_lon"].values
    # subsample for plotting
    rng = np.random.RandomState(0)
    if lat.size > 120_000:
        sel = rng.choice(lat.size, 120_000, replace=False)
        lat_s, lon_s = lat[sel], lon[sel]
    else:
        lat_s, lon_s = lat, lon

    fig = plt.figure(figsize=(13, 6.6))
    for i, (proj, extent, name, mask) in enumerate([
        (ccrs.NorthPolarStereo(), [-180, 180, 60, 90], "Northern hemisphere", lat_s > 0),
        (ccrs.SouthPolarStereo(), [-180, 180, -90, -60], "Southern hemisphere", lat_s < 0),
    ]):
        ax = fig.add_subplot(1, 2, i + 1, projection=proj)
        ax.set_extent(extent, ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#e8e4dc", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#d7e7f2", zorder=0)
        ax.coastlines(resolution="110m", linewidth=0.5, color="#555")
        gl = ax.gridlines(draw_labels=False, linewidth=0.4, color="grey", alpha=0.5)
        ax.scatter(lon_s[mask], lat_s[mask], s=1.5, c="#c0392b", alpha=0.25,
                   transform=ccrs.PlateCarree(), zorder=3, edgecolors="none")
        nn = int((lat > 0).sum()) if i == 0 else int((lat < 0).sum())
        ax.set_title(f"{name}\nN = {nn:,} matched ATLID profiles", fontsize=11)

    med = float(np.median(np.abs(lat)))
    fig.suptitle(
        "Where ORAC-SLSTR x EarthCARE-ATLID collocations occur (December 2025)\n"
        f"{len(v):,} matched profiles from {nfr} A-CTH frames  |  "
        f"median |latitude| = {med:.1f} deg  |  100% poleward of 60 deg",
        fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = OUT / "collocation_map_polar.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# 2. Match-quality distributions
# ---------------------------------------------------------------------------
def match_quality():
    df, _ = _load_all_matches(["valid_match", "distance_km", "time_diff_s"])
    v = df[df["valid_match"] == True]
    dist = v["distance_km"].values
    dt = v["time_diff_s"].values / 60.0

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].hist(dist, bins=np.arange(0, 3.05, 0.1), color="#1565c0", edgecolor="white")
    ax[0].axvline(np.median(dist), color="#c0392b", lw=1.5, ls="--",
                  label=f"median = {np.median(dist):.2f} km")
    ax[0].set_xlabel("ATLID -> nearest SLSTR pixel distance [km]")
    ax[0].set_ylabel("matched profiles")
    ax[0].set_title("Spatial match is tight (SLSTR ~1 km nadir pixels)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].hist(dt, bins=np.arange(0, 61, 3), color="#2e7d32", edgecolor="white")
    ax[1].axvline(np.median(dt), color="#c0392b", lw=1.5, ls="--",
                  label=f"median = {np.median(dt):.1f} min")
    ax[1].set_xlabel("|time offset|  ATLID vs SLSTR [min]")
    ax[1].set_ylabel("matched profiles")
    ax[1].set_title("Temporal offset (60-min window; agreement flat vs Dt)")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle("Collocation match quality - December 2025 (all matched profiles)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUT / "match_quality.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------------------
# 3. Single-crossing case study
# ---------------------------------------------------------------------------
def case_study():
    import sys
    sys.path.insert(0, ".")
    from datetime import timedelta
    from orac.io import julian_to_datetime
    from orac.slstr import discover_granules, open_granule
    from validation.readers import read_acth_track

    mpath = f"validation_data/slstr_cth_2025-12/matches_cth_{CASE_FRAME}.csv"
    m = pd.read_csv(mpath, parse_dates=["sev_scan_time"])
    mv = m[m["valid_match"] == True].copy()
    # dominant granule time
    gt = pd.to_datetime(mv["sev_scan_time"]).dropna()
    t0 = gt.mode().iloc[0]
    root = "/gws/ssde/j25a/cloud_ecv/data_out/slstr/v5.1_new_snowice/slstra/l2b"
    cand = discover_granules(root, (t0 - timedelta(minutes=6)).to_pydatetime(),
                             (t0 + timedelta(minutes=6)).to_pydatetime())
    # pick the granule whose footprint best contains the matched pixels
    tgt_lat, tgt_lon = mv["sev_lat"].median(), mv["sev_lon"].median()
    best, bestd = None, 1e9
    for g in cand:
        ds = open_granule(g, variables=("lat", "lon"))
        la, lo = ds["lat"].values, ds["lon"].values
        d = np.nanmin((la - tgt_lat) ** 2 + (np.cos(np.deg2rad(tgt_lat)) * (lo - tgt_lon)) ** 2)
        ds.close()
        if d < bestd:
            bestd, best = d, g
    ds = open_granule(best, variables=("lat", "lon", "cth", "cldmask"))
    slat = ds["lat"].values; slon = ds["lon"].values
    scth = np.ma.masked_invalid(ds["cth"].values)
    ds.close()

    # full ATLID track for this frame
    acth = glob.glob(f"earthcare_data/ATL_CTH_2A/2025/12/*/*{CASE_FRAME}*.h5")
    tr = read_acth_track(acth[0]) if acth else None

    proj = ccrs.SouthPolarStereo() if tgt_lat < 0 else ccrs.NorthPolarStereo()
    fig = plt.figure(figsize=(9, 8.5))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    latc, lonc = float(tgt_lat), float(tgt_lon)
    ax.set_extent([lonc - 22, lonc + 22, latc - 6, latc + 6], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#efece6", zorder=0)
    ax.coastlines(resolution="50m", linewidth=0.6, color="#555")
    ax.gridlines(draw_labels=True, linewidth=0.4, color="grey", alpha=0.5)

    # SLSTR swath cloud-top (subsample grid for speed)
    st = 3
    pm = ax.pcolormesh(slon[::st, ::st], slat[::st, ::st], scth[::st, ::st],
                       transform=ccrs.PlateCarree(), cmap="viridis", vmin=0, vmax=12,
                       shading="auto", alpha=0.85, zorder=1)
    cb = fig.colorbar(pm, ax=ax, shrink=0.6, pad=0.08)
    cb.set_label("SLSTR / ATLID cloud-top height [km]")

    # ATLID nadir track (full frame, grey) + matched segment coloured by ATLID CTH
    if tr is not None:
        ax.plot(tr["lon"], tr["lat"], transform=ccrs.PlateCarree(),
                color="0.35", lw=0.8, zorder=3, label="ATLID nadir track (full frame)")
    ax.scatter(mv["ec_lon"], mv["ec_lat"], c=np.clip(mv["cth_atlid_thick_km"], 0, 12),
               cmap="viridis", vmin=0, vmax=12, s=7, transform=ccrs.PlateCarree(),
               zorder=4, edgecolors="k", linewidths=0.15,
               label=f"matched ATLID profiles (N={len(mv):,})")

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"Single collocation crossing - frame {CASE_FRAME}, {t0:%Y-%m-%d %H:%M} UTC\n"
        f"ATLID nadir lidar threading the SLSTR swath  |  median match {mv['distance_km'].median():.2f} km",
        fontsize=11)
    fig.tight_layout()
    p = OUT / "crossing_case_study.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    collocation_map()
    match_quality()
    case_study()
