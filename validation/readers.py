"""EarthCARE L2 product readers — minimal, return canonical NumPy arrays."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

# EarthCARE HDF5 sentinel fill (h5py does not auto-decode it).
_AEBD_FILL = 9.96921e36
_TIME_EPOCH = np.datetime64("2000-01-01T00:00:00", "ns")


def _mask_fill(arr: np.ndarray, fill: float = _AEBD_FILL) -> np.ndarray:
    """Replace sentinel-fill values with NaN. Operates in float64."""
    out = arr.astype(np.float64)
    out[np.abs(out) >= 1e30] = np.nan
    return out


def read_aebd_track(path: str | Path) -> dict:
    """Read an ATL_EBD_2A frame and return per-profile + per-bin fields.

    Returns a dict with:
        lat, lon          : (n_profile,) float64 [deg]
        time              : (n_profile,) datetime64[ns]
        extinction        : (n_profile, n_bin) float64 [m^-1] — NaN where fill
        height            : (n_profile, n_bin) float64 [m]
        quality_status    : (n_profile, n_bin) int8 — 0 good, 3 saturated, etc.
        frame_id          : str
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        sd = f["ScienceData"]
        lat = np.asarray(sd["latitude"][:], dtype=np.float64)
        lon = np.asarray(sd["longitude"][:], dtype=np.float64)
        t_sec = np.asarray(sd["time"][:], dtype=np.float64)
        ext = _mask_fill(np.asarray(sd["particle_extinction_coefficient_355nm"][:]))
        height = _mask_fill(np.asarray(sd["height"][:]))
        qs = np.asarray(sd["quality_status"][:], dtype=np.int8)

    t_sec_clean = np.where(np.isfinite(t_sec) & (t_sec < 1e30), t_sec, 0.0)
    time = _TIME_EPOCH + (t_sec_clean * 1e9).astype("timedelta64[ns]")

    return {
        "lat": lat,
        "lon": lon,
        "time": time,
        "extinction": ext,
        "height": height,
        "quality_status": qs,
        "frame_id": path.stem.split("_")[-1],
    }


def read_acth_track(path: str | Path) -> dict:
    """Read an ATL_CTH_2A frame and return per-profile fields.

    Returns a dict with one value per ATLID profile (n_profile,):
        lat, lon                  : float64 [deg]
        time                      : datetime64[ns]
        cth_thick                 : float64 [m, AMSL] — `ATLID_thick_cloud_top_height`
                                    (uppermost optically-thick top, no horizontal
                                    averaging). Headline reference for passive CTH.
        cth_raw                   : float64 [m, AMSL] — `ATLID_cloud_top_height`
                                    (lidar-detected top, 11-profile averaged).
                                    Diagnostic; sees thin cirrus passive misses.
        cth_confidence            : int8 — `ATLID_cloud_top_height_confidence`
                                    (file says 0–10; observed up to ~20). −127 fill.
        quality_status            : int8 — −1 no cloud, 0 good, 1 valid/low-conf,
                                    2/3 warnings, 4 bad.
        cloud_class               : int8 — `simplified_uppermost_cloud_classification`
                                    (1 thick / 2 thin / 3 thin-over-thick /
                                     4 thick-over-thick / 5 thin-over-thin /
                                     6 cloud-influenced no-detection / 0 none).
        geoid_offset              : float64 [m] — geoid above WGS84; carried for
                                    audit only, not applied (both CTH refs are AMSL).
        tropopause_height_wmo     : float64 [m, AMSL] — for sanity caps.
        frame_id                  : str
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        sd = f["ScienceData"]
        lat = np.asarray(sd["latitude"][:], dtype=np.float64)
        lon = np.asarray(sd["longitude"][:], dtype=np.float64)
        t_sec = np.asarray(sd["time"][:], dtype=np.float64)
        cth_thick = _mask_fill(np.asarray(sd["ATLID_thick_cloud_top_height"][:]))
        cth_raw = _mask_fill(np.asarray(sd["ATLID_cloud_top_height"][:]))
        conf = np.asarray(sd["ATLID_cloud_top_height_confidence"][:], dtype=np.int8)
        qs = np.asarray(sd["quality_status"][:], dtype=np.int8)
        cls = np.asarray(sd["simplified_uppermost_cloud_classification"][:], dtype=np.int8)
        geoid = _mask_fill(np.asarray(sd["geoid_offset"][:]))
        trop = _mask_fill(np.asarray(sd["tropopause_height_wmo"][:]))

    t_sec_clean = np.where(np.isfinite(t_sec) & (t_sec < 1e30), t_sec, 0.0)
    time = _TIME_EPOCH + (t_sec_clean * 1e9).astype("timedelta64[ns]")

    return {
        "lat": lat,
        "lon": lon,
        "time": time,
        "cth_thick": cth_thick,
        "cth_raw": cth_raw,
        "cth_confidence": conf,
        "quality_status": qs,
        "cloud_class": cls,
        "geoid_offset": geoid,
        "tropopause_height_wmo": trop,
        "frame_id": path.stem.split("_")[-1],
    }
