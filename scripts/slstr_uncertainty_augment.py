"""Augment the water-COT day matches with the reported uncertainties needed for
the error-consistency (z-score) test:

  sigma_orac = ORAC `cot_uncertainty`            (linear tau, absolute)     [granule]
  sigma_ref  = ACM-CAP liquid_optical_depth_error x tau  (fractional->abs) [A-TC frame]

delta = (cot_orac - cot_water_atlid) / sqrt(sigma_orac^2 + sigma_ref^2)

ORAC side sampled at the matched pixel (granule groupby); reference side by
frame_id + exact ec_time. Output parquet -> .uncertainty_cache/ (GWS; /tmp is a
small shared tmpfs that fills up).
"""
from __future__ import annotations

import datetime as dt
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import pandas as pd

from orac.slstr import discover_granules, open_granule

SLSTR_ROOT = "/gws/ssde/j25a/cloud_ecv/data_out/slstr/v5.1_new_snowice/slstra/l2b"
ACM_ROOT = "earthcare_data/ACM_CAP_2B"
GLOB = "validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv"
OUT = Path(".uncertainty_cache") / "cot_unc_pairs.parquet"
PIX_PER_GRAN = 1200 * 1500
COLS = ["valid_match", "quality_status_atlid", "cot_orac", "cot_water_atlid",
        "phase_orac", "liquid_only_atlid", "cldmask_orac", "lsflag_orac",
        "sza_orac", "ec_lat", "ec_time", "frame_id",
        "sev_along_track", "sev_across_track", "sev_pixel_id"]


def _frame_map():
    return {p.stem.split("_")[-1]: p for p in Path(ACM_ROOT).rglob("*.h5")}


def ref_augment(d, fmap):
    d["cot_fracerr_atlid"] = np.nan
    ec_ns = pd.to_datetime(d["ec_time"]).astype("int64").values
    for fid, grp in d.groupby("frame_id"):
        path = fmap.get(str(fid))
        if path is None:
            continue
        try:
            with h5py.File(path, "r") as f:
                sd = f["ScienceData"]
                t = np.asarray(sd["time"][:], dtype=np.float64)
                fe = np.asarray(sd["liquid_optical_depth_error"][:], dtype=np.float64)
        except Exception:
            continue
        t = np.where(np.isfinite(t) & (t < 1e30), t, 0.0)
        prof_ns = (np.datetime64("2000-01-01T00:00:00") +
                   (t * 1e9).astype("timedelta64[ns]")).astype("int64")
        fe = np.where(np.isfinite(fe) & (fe < 1e30), fe, np.nan)
        order = np.argsort(prof_ns); ps = prof_ns[order]
        q = ec_ns[grp.index]
        idx = np.clip(np.searchsorted(ps, q), 0, len(ps) - 1)
        left = np.clip(idx - 1, 0, len(ps) - 1)
        pick = np.where(np.abs(ps[idx] - q) <= np.abs(ps[left] - q), idx, left)
        d.loc[grp.index, "cot_fracerr_atlid"] = fe[order[pick]]
    print(f"ref: cot_fracerr finite {100*np.isfinite(d['cot_fracerr_atlid']).mean():.0f}%")
    return d


def orac_augment(d):
    grans = discover_granules(SLSTR_ROOT, dt.datetime(2025, 12, 1), dt.datetime(2026, 1, 1))
    gmap = {int(g.start_time.strftime("%Y%m%d%H%M")): g for g in grans}
    d["pixkey"] = (d["sev_pixel_id"] // PIX_PER_GRAN).astype(np.int64)
    d["cot_unc_orac"] = np.nan
    for pk, grp in d[d["sev_pixel_id"] >= 0].groupby("pixkey"):
        g = gmap.get(int(pk))
        if g is None:
            continue
        try:
            ds = open_granule(g, variables=("cot_uncertainty",))
            cu = np.asarray(ds["cot_uncertainty"].squeeze(drop=True).values)
            ds.close()
        except Exception:
            continue
        at = grp["sev_along_track"].astype(int).values
        ac = grp["sev_across_track"].astype(int).values
        inb = (at >= 0) & (at < cu.shape[0]) & (ac >= 0) & (ac < cu.shape[1])
        vals = np.full(len(grp), np.nan); vals[inb] = cu[at[inb], ac[inb]]
        d.loc[grp.index, "cot_unc_orac"] = vals
    print(f"orac: cot_unc finite {100*np.isfinite(d['cot_unc_orac']).mean():.0f}%")
    return d


def main():
    paths = sorted(glob.glob(GLOB))
    d = pd.concat([pd.read_csv(p, usecols=lambda c: c in COLS) for p in paths],
                  ignore_index=True)
    d = d[(d["valid_match"] == True) & (d["sev_pixel_id"] >= 0)].reset_index(drop=True)
    print(f"valid matches: {len(d)}")
    d = ref_augment(d, _frame_map())
    d = orac_augment(d)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT)
    print("wrote", OUT)

    # quick delta on the phase-agree liquid, qc, cloudy subset
    m = ((d.quality_status_atlid == 0) & (d.liquid_only_atlid == True)
         & (d.cldmask_orac == 1) & (d.phase_orac == 1)
         & np.isfinite(d.cot_orac) & np.isfinite(d.cot_water_atlid)
         & np.isfinite(d.cot_unc_orac) & np.isfinite(d.cot_fracerr_atlid)
         & (d.cot_orac > 0) & (d.cot_water_atlid > 0))
    v = d[m].copy()
    sref = v.cot_fracerr_atlid * v.cot_water_atlid
    denom = np.sqrt(v.cot_unc_orac**2 + sref**2)
    dl = ((v.cot_orac - v.cot_water_atlid) / denom).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\nCOT error-consistency  N={len(dl)}")
    print(f"  sigma_orac median {v.cot_unc_orac.median():.2f}   sigma_ref median {sref.median():.2f}")
    print(f"  delta: mean {dl.mean():+.2f}  median {dl.median():+.2f}  std {dl.std():.2f}  "
          f"robust {(np.percentile(dl,75)-np.percentile(dl,25))/1.349:.2f}")
    print(f"  within +/-1 {100*(dl.abs()<1).mean():.0f}%  +/-2 {100*(dl.abs()<2).mean():.0f}%")


if __name__ == "__main__":
    raise SystemExit(main())
