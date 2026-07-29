"""Error-consistency (z-score) test for ORAC SLSTR cloud-top height vs EarthCARE
A-CTH — diagnostic pass.

delta = (cth_orac - cth_atlid) / sqrt(sigma_orac^2 + sigma_ref^2)
ATLID CTH is treated as truth (sigma_ref ~ one range bin, small vs ORAC's km sigma;
tested both = 0 and a nominal 0.1 km). If ORAC's stated sigma is calibrated and
errors Gaussian, delta ~ N(0,1): std ~ 1, ~68% within +/-1.

Caches the CTH matches to parquet so downstream iteration is fast.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# /tmp is a small shared tmpfs that fills up; cache on the GWS project fs instead.
CACHE = Path(".uncertainty_cache") / "cth_unc.parquet"
COLS = ["valid_match", "cth_orac_km", "cth_orac_uncertainty_km",
        "cth_orac_corrected_km", "cth_orac_corrected_uncertainty_km",
        "cth_atlid_thick_km", "cldmask_orac", "quality_status_atlid", "ec_lat"]


def load() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    ps = sorted(glob.glob("validation_data/slstr_cth_2025-12/matches_*.csv"))
    d = pd.concat([pd.read_csv(p, usecols=lambda c: c in COLS) for p in ps],
                  ignore_index=True)
    d = d[(d.valid_match == True) & (d.cldmask_orac == 1)
          & (d.quality_status_atlid == 0)].reset_index(drop=True)
    d.to_parquet(CACHE)
    return d


def diag_sigma(name, s):
    s = s[np.isfinite(s)]
    cap = np.nanmax(s)
    print(f"\n[{name}] N={len(s)}  min/med/max = {s.min():.3f}/{np.median(s):.3f}/{s.max():.3f}")
    print(f"   at cap (>= {0.975*cap:.1f}): {100*(s >= 0.975*cap).mean():.0f}%   "
          f"near 0 (< 0.05): {100*(s < 0.05).mean():.0f}%   "
          f"informative (0.05..{0.975*cap:.1f}): {100*((s>=0.05)&(s<0.975*cap)).mean():.0f}%")


def delta_stats(tag, d, xo, so, sref_km=0.0, sig_lo=0.05, sig_hi=None):
    x = d[xo].values; xr = d["cth_atlid_thick_km"].values; sg = d[so].values
    m = np.isfinite(x) & np.isfinite(xr) & np.isfinite(sg) & (sg > sig_lo)
    if sig_hi is not None:
        m &= sg < sig_hi
    x, xr, sg = x[m], xr[m], sg[m]
    denom = np.sqrt(sg**2 + sref_km**2)
    dl = (x - xr) / denom
    dl = dl[np.isfinite(dl)]
    rstd = (np.percentile(dl, 75) - np.percentile(dl, 25)) / 1.349
    print(f"\n=== delta [{tag}]  N={len(dl)} ===")
    print(f"  mean {np.mean(dl):+.2f}  median {np.median(dl):+.2f}  "
          f"std {np.std(dl):.2f}  ROBUST std(IQR/1.349) {rstd:.2f}")
    print(f"  within +/-1 {100*(np.abs(dl)<1).mean():.0f}% (68)  "
          f"+/-2 {100*(np.abs(dl)<2).mean():.0f}% (95)  "
          f"+/-3 {100*(np.abs(dl)<3).mean():.0f}% (99.7)")
    print(f"  skew {stats.skew(dl):+.2f}  excess-kurt {stats.kurtosis(dl):+.2f}")
    return dl


def main():
    d = load()
    print(f"CTH QC matches: {len(d)}")
    diag_sigma("cth_uncertainty (raw)", d["cth_orac_uncertainty_km"])
    diag_sigma("cth_corrected_uncertainty", d["cth_orac_corrected_uncertainty_km"])

    cap = np.nanmax(d["cth_orac_corrected_uncertainty_km"])
    # (1) all informative-sigma pixels, sigma_ref=0
    delta_stats("corrected, informative sigma", d,
                "cth_orac_corrected_km", "cth_orac_corrected_uncertainty_km",
                sref_km=0.0, sig_lo=0.05, sig_hi=0.975*cap)
    # (2) same but nominal ATLID sigma 0.1 km
    delta_stats("corrected, informative, sigma_ref=0.1km", d,
                "cth_orac_corrected_km", "cth_orac_corrected_uncertainty_km",
                sref_km=0.1, sig_lo=0.05, sig_hi=0.975*cap)
    # (3) raw (uncorrected) uncertainty for comparison
    capr = np.nanmax(d["cth_orac_uncertainty_km"])
    delta_stats("raw uncertainty, informative", d,
                "cth_orac_km", "cth_orac_uncertainty_km",
                sref_km=0.0, sig_lo=0.05, sig_hi=0.975*capr)


if __name__ == "__main__":
    raise SystemExit(main())
