"""Per-variable statistics over a matches DataFrame.

Two views per variable:

- **Sample-level** : one row per ATLID profile, joined to its nearest
  SEVIRI pixel. Preserves thin-cirrus signal and lets τ-stratified
  reporting filter before aggregation.
- **Pixel-aggregate** : groupby ``(sev_scan_time, sev_pixel_id)`` with
  per-variable rule (mean for cot/cer/cwp, max for cth, mode for
  phase/cldtype, any-cloudy for cldmask). Matches the spatial scale
  SEVIRI's retrieval represents.

Both views go through the same stratification engine so the resulting
tables are directly comparable.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

# Filter type: takes a DataFrame, returns a boolean mask.
StratumFn = Callable[[pd.DataFrame], pd.Series]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_to_pixel(
    matches: pd.DataFrame,
    var_atlid: str,
    var_orac: str,
    extra_any: Sequence[str] = ("attenuated", "cot_orac_saturated"),
) -> pd.DataFrame:
    """Group ``matches`` by ``(sev_scan_time, sev_pixel_id)`` and aggregate.

    For continuous variables (``var_atlid``, ``var_orac``) the rule is
    arithmetic mean — appropriate for additive quantities like cot, cer, cwp.
    Other variables (cth = max, phase = mode, cldmask = any) need their own
    rules and are not handled here.

    Each pixel-aggregate row also carries:

    - ``n_atlid``         : number of ATLID profiles contributing to the pixel
    - ``ec_lat``          : mean of ATLID-profile latitudes in the pixel
    - ``distance_km``     : mean haversine distance to pixel centre
    - ``time_diff_s``     : mean time offset to SEVIRI scan_time
    - per-flag ``any()``  : True if any contributing profile carries the flag
                            (e.g. attenuated, cot_orac_saturated)
    """
    extras = [c for c in extra_any if c in matches.columns]
    agg = {
        var_atlid: (var_atlid, "mean"),
        var_orac: (var_orac, "mean"),
        "n_atlid": (var_atlid, "size"),
        "ec_lat": ("ec_lat", "mean"),
        "distance_km": ("distance_km", "mean"),
        "time_diff_s": ("time_diff_s", "mean"),
        **{c: (c, "any") for c in extras},
    }
    return matches.groupby(["sev_scan_time", "sev_pixel_id"], as_index=False).agg(**agg)


# ---------------------------------------------------------------------------
# Continuous-variable stats
# ---------------------------------------------------------------------------

def continuous_stats(d: pd.DataFrame, x: str, y: str) -> dict:
    """Bias, RMSE, MAE, R, slope, intercept, N for ``y - x``."""
    d = d[[x, y]].dropna()
    n = len(d)
    if n < 2:
        return dict(n=n, bias=np.nan, rmse=np.nan, mae=np.nan,
                    r=np.nan, slope=np.nan, intercept=np.nan)
    diff = d[y].values - d[x].values
    bias = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    mae = float(np.abs(diff).mean())
    if d[x].std() > 0 and d[y].std() > 0:
        r = float(np.corrcoef(d[x].values, d[y].values)[0, 1])
        slope, intercept = np.polyfit(d[x].values, d[y].values, 1)
    else:
        r = np.nan
        slope = intercept = np.nan
    return dict(n=n, bias=bias, rmse=rmse, mae=mae, r=r,
                slope=float(slope), intercept=float(intercept))


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------

def cot_strata(
    cot_atlid_col: str = "cot_atlid",
    lat_col: str = "ec_lat",
    dist_col: str = "distance_km",
    tdiff_col: str = "time_diff_s",
) -> dict[str, StratumFn | None]:
    """Default stratification for cot validation.

    ``None`` for the "all" stratum means no filter (the full sample).
    """
    return {
        "all": None,
        "lat_tropics": lambda d: np.abs(d[lat_col]) < 30,
        "lat_midlat":  lambda d: (np.abs(d[lat_col]) >= 30) & (np.abs(d[lat_col]) < 60),
        "lat_polar":   lambda d: np.abs(d[lat_col]) >= 60,
        "dist_lt2km":  lambda d: d[dist_col] < 2,
        "dist_2_5km":  lambda d: (d[dist_col] >= 2) & (d[dist_col] < 5),
        "dist_ge5km":  lambda d: d[dist_col] >= 5,
        "tdiff_lt3min": lambda d: d[tdiff_col] < 180,
        "tdiff_ge3min": lambda d: d[tdiff_col] >= 180,
        "not_attenuated": lambda d: ~d.get("attenuated", pd.Series(False, index=d.index)).fillna(False),
        "attenuated":     lambda d:  d.get("attenuated", pd.Series(False, index=d.index)).fillna(False),
        "tau_thin":       lambda d: (d[cot_atlid_col] >= 0.15) & (d[cot_atlid_col] < 1),
        "tau_mid":        lambda d: (d[cot_atlid_col] >= 1) & (d[cot_atlid_col] < 3),
        "tau_thick":      lambda d: (d[cot_atlid_col] >= 3) & (d[cot_atlid_col] < 10),
        "tau_very_thick": lambda d: d[cot_atlid_col] >= 10,
    }


def stratified_stats(
    matches: pd.DataFrame,
    var_atlid: str,
    var_orac: str,
    strata: Mapping[str, StratumFn | None] | None = None,
) -> pd.DataFrame:
    """Apply :func:`continuous_stats` per stratum.

    Returns a tidy table with one row per stratum and columns
    ``(stratum, n, bias, rmse, mae, r, slope, intercept)``.
    """
    if strata is None:
        strata = cot_strata()
    rows = []
    for name, sel in strata.items():
        d = matches if sel is None else matches[sel(matches).fillna(False)]
        rows.append({"stratum": name, **continuous_stats(d, var_atlid, var_orac)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience: full cot report
# ---------------------------------------------------------------------------

def cot_report(
    matches: pd.DataFrame,
    base_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> dict[str, pd.DataFrame]:
    """End-to-end cot statistics on a matches DataFrame.

    ``base_filter`` selects the rows used as input to both views (e.g.
    cloudy in ORAC + ATLID > 0 + not saturated). If ``None``, defaults to
    ``cldmask_orac == 1 & cot_atlid > 0 & ~cot_orac_saturated``.

    Returns a dict with keys ``sample`` and ``pixel`` mapping to the
    stratified-stats tables for each view.
    """
    if base_filter is None:
        def base_filter(d):
            mask = pd.Series(True, index=d.index)
            if "cldmask_orac" in d.columns:
                mask &= d["cldmask_orac"] == 1
            if "cot_atlid" in d.columns:
                mask &= d["cot_atlid"] > 0
            if "cot_orac_saturated" in d.columns:
                mask &= ~d["cot_orac_saturated"]
            if "valid_match" in d.columns:
                mask &= d["valid_match"]
            mask &= d["cot_atlid"].notna() & d["cot_orac"].notna()
            return mask

    base = matches[base_filter(matches)]
    sample_stats = stratified_stats(base, "cot_atlid", "cot_orac")
    pix = aggregate_to_pixel(base, "cot_atlid", "cot_orac")
    pixel_stats = stratified_stats(pix, "cot_atlid", "cot_orac")
    return {"sample": sample_stats, "pixel": pixel_stats}
