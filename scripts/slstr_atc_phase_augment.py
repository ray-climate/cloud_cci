"""Augment the SLSTR x EarthCARE day matches with EarthCARE A-TC (ATLID Target
Classification) phase and cloud detection, to validate ORAC's cloud MASK and cloud
PHASE against a purpose-built categorical reference (the two-way contingency that
ACM-CAP's liquid-centric flags cannot provide — POD_ice in particular).

A-TC classifies every ATLID bin:
   0 clear | 1 warm-liquid | 2 supercooled-liquid | 3 ice | <0 missing/surface/atten
A passive imager (ORAC) sees the CLOUD-TOP phase, so each profile is reduced to
the phase of its highest cloud bin. Cloud mask = any cloud bin in the column.

Augment is by frame_id (aligned across EarthCARE products) + exact ec_time match.
ALL valid matches are augmented (clear + cloudy) so the cloud-mask contingency has
both hit and false-alarm cells.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import pandas as pd

ATC_ROOT = "earthcare_data/ATL_TC__2A"
GLOB = "validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv"
OUT = ("/tmp/claude-7051641/-gws-pw-j07-nceo-aerosolfire-rsong-project-cloud-cci/"
       "3a1e8f12-6f9c-4529-9d79-b8ab9052e120/scratchpad/atc_phase.parquet")
COLS = ["valid_match", "phase_orac", "cldmask_orac", "cot_orac", "lsflag_orac",
        "sza_orac", "ec_lat", "ec_time", "frame_id"]

CLEAR, LIQ_W, LIQ_SC, ICE = 0, 1, 2, 3


def _frame_map() -> dict:
    return {p.stem.split("_")[-1]: p for p in Path(ATC_ROOT).rglob("*.h5")}


def reduce_frame(cls: np.ndarray, height: np.ndarray):
    """Per-profile reduction of the A-TC column.

    Returns arrays (n_profile,):
      cloud       : bool  — any cloud bin (1/2/3) in column
      top_class   : int8  — class of the highest-altitude cloud bin (0 if none)
      any_ice     : bool  — any ice bin (3)
      any_liquid  : bool  — any liquid bin (1/2)
      usable      : bool  — column has at least one classified (>=0) bin
    """
    n = cls.shape[0]
    is_cloud = (cls == LIQ_W) | (cls == LIQ_SC) | (cls == ICE)
    any_ice = (cls == ICE).any(axis=1)
    any_liq = ((cls == LIQ_W) | (cls == LIQ_SC)).any(axis=1)
    cloud = is_cloud.any(axis=1)
    usable = (cls >= 0).any(axis=1)
    # top cloud bin = cloud bin with the greatest height
    h = np.where(is_cloud, height, -np.inf)
    top_idx = np.argmax(h, axis=1)
    top_class = cls[np.arange(n), top_idx].astype(np.int8)
    top_class = np.where(cloud, top_class, np.int8(0))
    return cloud, top_class, any_ice, any_liq, usable


def main() -> int:
    fmap = _frame_map()
    print(f"A-TC frames available: {len(fmap)}")
    paths = sorted(glob.glob(GLOB))
    d = pd.concat([pd.read_csv(p, usecols=lambda c: c in COLS) for p in paths],
                  ignore_index=True)
    d = d[d["valid_match"] == True].reset_index(drop=True)
    print(f"valid matches: {len(d)}")

    for col in ["atc_cloud", "atc_any_ice", "atc_any_liquid", "atc_usable"]:
        d[col] = False
    d["atc_top_class"] = np.int8(-1)  # -1 = not augmented

    ec_ns = pd.to_datetime(d["ec_time"]).astype("int64").values
    hit_frames = 0
    for fid, grp in d.groupby("frame_id"):
        path = fmap.get(str(fid))
        if path is None:
            continue
        try:
            with h5py.File(path, "r") as f:
                sd = f["ScienceData"]
                t = np.asarray(sd["time"][:], dtype=np.float64)
                cls = np.asarray(sd["classification"][:], dtype=np.int16)
                height = np.asarray(sd["height"][:], dtype=np.float64)
        except Exception:
            continue
        cloud, top_class, any_ice, any_liq, usable = reduce_frame(cls, height)
        t = np.where(np.isfinite(t) & (t < 1e30), t, 0.0)
        prof_ns = (np.datetime64("2000-01-01T00:00:00") +
                   (t * 1e9).astype("timedelta64[ns]")).astype("int64")
        order = np.argsort(prof_ns)
        ps = prof_ns[order]
        q = ec_ns[grp.index]
        idx = np.clip(np.searchsorted(ps, q), 0, len(ps) - 1)
        left = np.clip(idx - 1, 0, len(ps) - 1)
        pick = np.where(np.abs(ps[idx] - q) <= np.abs(ps[left] - q), idx, left)
        pj = order[pick]
        d.loc[grp.index, "atc_cloud"] = cloud[pj]
        d.loc[grp.index, "atc_any_ice"] = any_ice[pj]
        d.loc[grp.index, "atc_any_liquid"] = any_liq[pj]
        d.loc[grp.index, "atc_usable"] = usable[pj]
        d.loc[grp.index, "atc_top_class"] = top_class[pj]
        d.loc[grp.index, "_dt_s"] = np.abs(prof_ns[pj] - q) / 1e9
        hit_frames += 1

    aug = d["atc_top_class"] >= 0
    print(f"augmented {hit_frames} frames; {aug.sum()} rows ({100*aug.mean():.0f}%) "
          f"have an A-TC match")
    if aug.any():
        print(f"profile-match |dt| median {d.loc[aug,'_dt_s'].median():.3f} s, "
              f"max {d.loc[aug,'_dt_s'].max():.3f} s")
    d.to_parquet(OUT)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
