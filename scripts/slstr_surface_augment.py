"""Augment the SLSTR x EarthCARE day matches with ORAC surface temperature
(stemp) and cloud water path (cwp), sampled at each matched pixel, so we can
stratify the retrieval skill by surface type (sea-ice / open water / snow &
ice-sheet / snow-free land) and validate CWP.

Two match sets are augmented and cached as parquet:
  * synergy day  -> water-COT and CER (ACM-CAP reference)
  * cot day      -> ice-COT           (A-EBD reference)

Groups matched rows by their source granule (pixel_key encoded in
sev_pixel_id), opens each granule once, samples stemp + cwp at the stored
(along_track, across_track) indices.
"""
from __future__ import annotations

import datetime as dt
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from orac.slstr import discover_granules, open_granule

SLSTR_ROOT = "/gws/ssde/j25a/cloud_ecv/data_out/slstr/v5.1_new_snowice/slstra/l2b"
SCRATCH = ("/tmp/claude-7051641/-gws-pw-j07-nceo-aerosolfire-rsong-project-cloud-cci/"
           "3a1e8f12-6f9c-4529-9d79-b8ab9052e120/scratchpad")
PIX_PER_GRAN = 1200 * 1500

SETS = {
    "synergy": dict(
        glob="validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv",
        cols=["valid_match", "quality_status_atlid", "cot_orac", "cer_orac",
              "cot_water_atlid", "cer_water_atlid", "phase_orac",
              "liquid_only_atlid", "cldmask_orac", "lsflag_orac", "sza_orac",
              "ec_lat", "sev_along_track", "sev_across_track", "sev_pixel_id"],
        out=f"{SCRATCH}/surface_synergy.parquet",
    ),
    "cot": dict(
        glob="validation_data/slstr_cot_2025-12_day/matches_*.csv",
        cols=["valid_match", "quality_status_atlid", "cot_orac", "cot_atlid",
              "phase_orac", "cldmask_orac", "lsflag_orac", "sza_orac",
              "ec_lat", "sev_along_track", "sev_across_track", "sev_pixel_id",
              "cot_orac_saturated"],
        out=f"{SCRATCH}/surface_cot.parquet",
    ),
}


def _gmap():
    grans = discover_granules(SLSTR_ROOT, dt.datetime(2025, 12, 1),
                              dt.datetime(2026, 1, 1))
    return {int(g.start_time.strftime("%Y%m%d%H%M")): g for g in grans}


def augment(name: str, cfg: dict, gmap: dict) -> pd.DataFrame:
    paths = sorted(glob.glob(cfg["glob"]))
    d = pd.concat([pd.read_csv(p, usecols=lambda c: c in cfg["cols"]) for p in paths],
                  ignore_index=True)
    d = d[(d["valid_match"] == True) & (d["sev_pixel_id"] >= 0)].copy()
    d["pixkey"] = (d["sev_pixel_id"] // PIX_PER_GRAN).astype(np.int64)
    print(f"[{name}] valid matches: {len(d)}, unique granules: {d['pixkey'].nunique()}")

    d["stemp_orac"] = np.nan
    d["cwp_orac"] = np.nan
    ok = 0
    for pk, grp in d.groupby("pixkey"):
        g = gmap.get(int(pk))
        if g is None:
            continue
        try:
            ds = open_granule(g, variables=("stemp", "cwp"))
            st = np.asarray(ds["stemp"].squeeze(drop=True).values)
            cw = np.asarray(ds["cwp"].squeeze(drop=True).values)
            ds.close()
        except Exception as e:
            continue
        at = grp["sev_along_track"].astype(int).values
        ac = grp["sev_across_track"].astype(int).values
        inb = (at >= 0) & (at < st.shape[0]) & (ac >= 0) & (ac < st.shape[1])
        for var, col in ((st, "stemp_orac"), (cw, "cwp_orac")):
            vals = np.full(len(grp), np.nan)
            vals[inb] = var[at[inb], ac[inb]]
            d.loc[grp.index, col] = vals
        ok += 1
    print(f"[{name}] sampled from {ok} granules; stemp finite: "
          f"{np.isfinite(d['stemp_orac']).mean()*100:.0f}%")
    d.to_parquet(cfg["out"])
    print(f"[{name}] wrote {cfg['out']}")
    return d


def describe(name: str, d: pd.DataFrame) -> None:
    print(f"\n===== {name}: lsflag / stemp distribution (finite stemp) =====")
    dd = d[np.isfinite(d["stemp_orac"])].copy()
    print("lsflag_orac value counts:")
    print(dd["lsflag_orac"].value_counts(dropna=False).to_string())
    for lf, g in dd.groupby("lsflag_orac"):
        st = g["stemp_orac"].values
        print(f"  lsflag={lf}: N={len(g)}  stemp K "
              f"min/median/max = {np.nanmin(st):.1f}/{np.nanmedian(st):.1f}/{np.nanmax(st):.1f}")
    for thr in (271.35, 273.15):
        print(f"  stemp < {thr} K: {(dd['stemp_orac'] < thr).mean()*100:.0f}% of pixels")


def main() -> int:
    gmap = _gmap()
    for name, cfg in SETS.items():
        d = augment(name, cfg, gmap)
        describe(name, d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
