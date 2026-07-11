"""Augment the water-COT day matches with ORAC cloud-top temperature (ctt),
sampled at each matched SLSTR pixel, to test whether phase misclassification is
driven by cloud-top temperature (cold tops read as ice).

Groups matched rows by their source granule (via the pixel_key encoded in
sev_pixel_id), opens each granule once, and samples ctt at the stored
(along_track, across_track) indices. Writes the filtered + augmented frame to the
scratchpad for the figure step.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from orac.slstr import discover_granules, open_granule

SLSTR_ROOT = "/gws/ssde/j25a/cloud_ecv/data_out/slstr/v5.1_new_snowice/slstra/l2b"
OUT = ("/tmp/claude-7051641/-gws-pw-j07-nceo-aerosolfire-rsong-project-cloud-cci/"
       "3a1e8f12-6f9c-4529-9d79-b8ab9052e120/scratchpad/cotw_phase_ctt.parquet")
NCOLS = 1500
PIX_PER_GRAN = 1200 * 1500


def main() -> int:
    cols = ["valid_match", "quality_status_atlid", "cot_orac", "cot_water_atlid",
            "phase_orac", "liquid_only_atlid", "cldmask_orac", "lsflag_orac",
            "sza_orac", "ec_lat", "sev_along_track", "sev_across_track", "sev_pixel_id"]
    paths = sorted(glob.glob("validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv"))
    d = pd.concat([pd.read_csv(p, usecols=lambda c: c in cols) for p in paths],
                  ignore_index=True)
    m = ((d["valid_match"] == True) & (d["quality_status_atlid"] == 0)
         & (d["liquid_only_atlid"] == True) & (d["cldmask_orac"] == 1)
         & np.isfinite(d["cot_orac"]) & np.isfinite(d["cot_water_atlid"])
         & (d["cot_orac"] > 0) & (d["cot_water_atlid"] > 0)
         & (d["sev_pixel_id"] >= 0))
    d = d[m].copy()
    d["pixkey"] = (d["sev_pixel_id"] // PIX_PER_GRAN).astype(np.int64)
    print(f"filtered liquid-only profiles: {len(d)}, unique granules: {d['pixkey'].nunique()}")

    # map pixel_key (YYYYMMDDHHMM int) -> granule
    grans = discover_granules(SLSTR_ROOT,
                              __import__("datetime").datetime(2025, 12, 1),
                              __import__("datetime").datetime(2026, 1, 1))
    gmap = {int(g.start_time.strftime("%Y%m%d%H%M")): g for g in grans}

    d["ctt_orac"] = np.nan
    ok = 0
    for pk, grp in d.groupby("pixkey"):
        g = gmap.get(int(pk))
        if g is None:
            continue
        try:
            ds = open_granule(g, variables=("ctt",))
            ctt = np.asarray(ds["ctt"].squeeze(drop=True).values)
            ds.close()
        except Exception:
            continue
        at = grp["sev_along_track"].astype(int).values
        ac = grp["sev_across_track"].astype(int).values
        inb = (at >= 0) & (at < ctt.shape[0]) & (ac >= 0) & (ac < ctt.shape[1])
        vals = np.full(len(grp), np.nan)
        vals[inb] = ctt[at[inb], ac[inb]]
        d.loc[grp.index, "ctt_orac"] = vals
        ok += 1
    print(f"sampled ctt from {ok} granules; ctt finite: {np.isfinite(d['ctt_orac']).mean()*100:.0f}%")
    d.to_parquet(OUT)
    # quick look: misclassification vs ctt
    dd = d[np.isfinite(d["ctt_orac"])].copy()
    dd["ice"] = dd["phase_orac"] == 2
    dd["ctb"] = pd.cut(dd["ctt_orac"] - 273.15,
                       [-60, -30, -20, -10, 0, 40],
                       labels=["<−30", "−30..−20", "−20..−10", "−10..0", ">0 °C"])
    print("\nmisclassification (% ORAC=ice) vs cloud-top temperature:")
    for b, x in dd.groupby("ctb", observed=True):
        print(f"  {str(b):>10} °C: {100*x['ice'].mean():5.1f}%  (N={len(x)})")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
