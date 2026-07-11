"""Build a cloud water-path (CWP) validation pair for SLSTR ORAC vs EarthCARE
ACM-CAP, without re-running the collocation.

Reference side (ACM-CAP): open each matched frame once, integrate the radar+lidar
liquid_water_content profile to a liquid water path (LWP) and read the direct
ice_water_path (IWP); match each collocated row to its ACM-CAP profile by ec_time.
ORAC side: sample the ORAC `cwp` (total cloud water path, g/m2) at the matched
SLSTR pixel, reusing the granule-groupby pattern.

Both LWP and cwp are converted to g/m2. Output parquet feeds the CWP figure.
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
OUT = ("/tmp/claude-7051641/-gws-pw-j07-nceo-aerosolfire-rsong-project-cloud-cci/"
       "3a1e8f12-6f9c-4529-9d79-b8ab9052e120/scratchpad/cwp_pairs.parquet")
PIX_PER_GRAN = 1200 * 1500
COLS = ["valid_match", "quality_status_atlid", "cot_orac", "cer_orac", "phase_orac",
        "liquid_only_atlid", "liquid_present_atlid", "ice_present_atlid",
        "cldmask_orac", "lsflag_orac", "ec_lat", "ec_time", "frame_id",
        "sev_along_track", "sev_across_track", "sev_pixel_id"]


def _frame_file_map() -> dict:
    m = {}
    for p in Path(ACM_ROOT).rglob("*.h5"):
        m[p.stem.split("_")[-1]] = p
    return m


def ref_augment(d: pd.DataFrame, fmap: dict) -> pd.DataFrame:
    d["lwp_atlid"] = np.nan  # g/m2
    d["iwp_atlid"] = np.nan  # g/m2
    ec_ns = pd.to_datetime(d["ec_time"]).astype("int64").values
    for fid, grp in d.groupby("frame_id"):
        path = fmap.get(str(fid))
        if path is None:
            continue
        try:
            with h5py.File(path, "r") as f:
                sd = f["ScienceData"]
                t = np.asarray(sd["time"][:], dtype=np.float64)
                lwc = np.asarray(sd["liquid_water_content"][:], dtype=np.float64)
                h = np.asarray(sd["height"][:], dtype=np.float64)
                iwp = np.asarray(sd["ice_water_path"][:], dtype=np.float64)
        except Exception:
            continue
        # profile time (s since 2000-01-01) -> ns epoch for nearest match
        t = np.where(np.isfinite(t) & (t < 1e30), t, 0.0)
        prof_ns = (np.datetime64("2000-01-01T00:00:00") +
                   (t * 1e9).astype("timedelta64[ns]")).astype("int64")
        # integrate LWC (kg/m3) over height (m) -> kg/m2 -> g/m2
        lwc = np.where(np.isfinite(lwc) & (lwc < 1e30) & (lwc > 0), lwc, 0.0)
        h = np.where(np.isfinite(h) & (np.abs(h) < 1e30), h, np.nan)
        dz = np.abs(np.gradient(h, axis=1))
        dz = np.where(np.isfinite(dz), dz, 0.0)
        lwp = np.nansum(lwc * dz, axis=1) * 1000.0        # g/m2
        iwp_g = np.where(np.isfinite(iwp) & (iwp < 1e30), iwp, np.nan) * 1000.0

        order = np.argsort(prof_ns)
        ps = prof_ns[order]
        idx = np.searchsorted(ps, ec_ns[grp.index])
        idx = np.clip(idx, 0, len(ps) - 1)
        left = np.clip(idx - 1, 0, len(ps) - 1)
        pick = np.where(np.abs(ps[idx] - ec_ns[grp.index]) <=
                        np.abs(ps[left] - ec_ns[grp.index]), idx, left)
        pj = order[pick]
        d.loc[grp.index, "lwp_atlid"] = lwp[pj]
        d.loc[grp.index, "iwp_atlid"] = iwp_g[pj]
    print(f"ref: lwp finite {np.isfinite(d['lwp_atlid']).mean()*100:.0f}%")
    return d


def orac_cwp(d: pd.DataFrame) -> pd.DataFrame:
    grans = discover_granules(SLSTR_ROOT, dt.datetime(2025, 12, 1), dt.datetime(2026, 1, 1))
    gmap = {int(g.start_time.strftime("%Y%m%d%H%M")): g for g in grans}
    d["pixkey"] = (d["sev_pixel_id"] // PIX_PER_GRAN).astype(np.int64)
    d["cwp_orac"] = np.nan
    for pk, grp in d.groupby("pixkey"):
        g = gmap.get(int(pk))
        if g is None:
            continue
        try:
            ds = open_granule(g, variables=("cwp",))
            cw = np.asarray(ds["cwp"].squeeze(drop=True).values)
            ds.close()
        except Exception:
            continue
        at = grp["sev_along_track"].astype(int).values
        ac = grp["sev_across_track"].astype(int).values
        inb = (at >= 0) & (at < cw.shape[0]) & (ac >= 0) & (ac < cw.shape[1])
        vals = np.full(len(grp), np.nan)
        vals[inb] = cw[at[inb], ac[inb]]
        d.loc[grp.index, "cwp_orac"] = vals
    print(f"orac: cwp finite {np.isfinite(d['cwp_orac']).mean()*100:.0f}%")
    return d


def main() -> int:
    paths = sorted(glob.glob(GLOB))
    d = pd.concat([pd.read_csv(p, usecols=lambda c: c in COLS) for p in paths],
                  ignore_index=True)
    d = d[(d["valid_match"] == True) & (d["sev_pixel_id"] >= 0)].reset_index(drop=True)
    print(f"valid matches: {len(d)}")
    d = ref_augment(d, _frame_file_map())
    d = orac_cwp(d)
    d.to_parquet(OUT)
    print("wrote", OUT)

    # quick look: liquid CWP pair (phase-agree liquid, qc, cloudy)
    m = ((d["quality_status_atlid"] == 0) & (d["liquid_only_atlid"] == True)
         & (d["cldmask_orac"] == 1) & (d["phase_orac"] == 1)
         & np.isfinite(d["cwp_orac"]) & np.isfinite(d["lwp_atlid"])
         & (d["cwp_orac"] > 0) & (d["lwp_atlid"] > 0))
    v = d[m]
    diff = (v["cwp_orac"] - v["lwp_atlid"]).values
    print(f"\nLIQUID CWP pair N={len(v)}")
    print(f"  ORAC cwp median {np.median(v['cwp_orac']):.0f}  ACM-CAP LWP median {np.median(v['lwp_atlid']):.0f} g/m2")
    print(f"  median bias {np.median(diff):+.0f}  mean {np.mean(diff):+.0f}  g/m2")
    if len(v) > 3:
        print(f"  r(log) {np.corrcoef(np.log(v['cwp_orac']), np.log(v['lwp_atlid']))[0,1]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
