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
    extra_first: Sequence[str] = ("lsflag_orac",),
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
    - per-pixel attrs ``first()`` : value taken from the first contributing
                            profile (e.g. lsflag_orac is constant within a pixel)
    """
    extras_any = [c for c in extra_any if c in matches.columns]
    extras_first = [c for c in extra_first if c in matches.columns]
    agg = {
        var_atlid: (var_atlid, "mean"),
        var_orac: (var_orac, "mean"),
        "n_atlid": (var_atlid, "size"),
        "ec_lat": ("ec_lat", "mean"),
        "distance_km": ("distance_km", "mean"),
        "time_diff_s": ("time_diff_s", "mean"),
        **{c: (c, "any") for c in extras_any},
        **{c: (c, "first") for c in extras_first},
    }
    return matches.groupby(["sev_scan_time", "sev_pixel_id"], as_index=False).agg(**agg)


# ---------------------------------------------------------------------------
# Continuous-variable stats
# ---------------------------------------------------------------------------

def continuous_stats(d: pd.DataFrame, x: str, y: str,
                     log_floor: float = 0.05) -> dict:
    """Bias, RMSE, MAE, R, R_log, slope, intercept, N for ``y - x``.

    ``r_log`` is Pearson R on ``log10(clip(x, log_floor)) vs log10(clip(y, ...))``.
    For heavy-tailed quantities like cot it's the metric the literature
    uses; raw-space R is dominated by a few extreme points.
    """
    d = d[[x, y]].dropna()
    n = len(d)
    if n < 2:
        return dict(n=n, bias=np.nan, bias_lo=np.nan, bias_hi=np.nan,
                    median_bias=np.nan, median_bias_lo=np.nan, median_bias_hi=np.nan,
                    rmse=np.nan, mae=np.nan, r=np.nan, r_log=np.nan,
                    slope=np.nan, intercept=np.nan)
    xv = d[x].values
    yv = d[y].values
    diff = yv - xv
    bias = float(diff.mean())
    # median bias is the robust headline for heavy-tailed quantities (cot/cer):
    # the mean is dominated by a small high-value tail, so it can even flip sign
    # relative to the typical (median) difference.
    median_bias = float(np.median(diff))
    # 95% confidence intervals (analytical, fast):
    #  - mean: +/- 1.96 * standard error
    #  - median: distribution-free order-statistic interval
    _Z = 1.959964
    se = float(diff.std(ddof=1) / np.sqrt(n))
    bias_lo, bias_hi = bias - _Z * se, bias + _Z * se
    ds = np.sort(diff)
    lo = int(np.clip(np.floor(n / 2 - _Z * np.sqrt(n) / 2) - 1, 0, n - 1))
    hi = int(np.clip(np.ceil(n / 2 + _Z * np.sqrt(n) / 2) - 1, 0, n - 1))
    median_bias_lo, median_bias_hi = float(ds[lo]), float(ds[hi])
    rmse = float(np.sqrt((diff ** 2).mean()))
    mae = float(np.abs(diff).mean())
    if d[x].std() > 0 and d[y].std() > 0:
        r = float(np.corrcoef(xv, yv)[0, 1])
        slope, intercept = np.polyfit(xv, yv, 1)
        lx = np.log10(np.clip(xv, log_floor, None))
        ly = np.log10(np.clip(yv, log_floor, None))
        r_log = float(np.corrcoef(lx, ly)[0, 1]) if lx.std() > 0 and ly.std() > 0 else np.nan
    else:
        r = r_log = np.nan
        slope = intercept = np.nan
    return dict(n=n, bias=bias, bias_lo=bias_lo, bias_hi=bias_hi,
                median_bias=median_bias, median_bias_lo=median_bias_lo,
                median_bias_hi=median_bias_hi, rmse=rmse, mae=mae,
                r=r, r_log=r_log, slope=float(slope), intercept=float(intercept))


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------

def cot_strata(
    cot_atlid_col: str = "cot_atlid",
    lat_col: str = "ec_lat",
    dist_col: str = "distance_km",
    tdiff_col: str = "time_diff_s",
    lsflag_col: str = "lsflag_orac",
) -> dict[str, StratumFn | None]:
    """Default stratification for cot validation.

    ``None`` for the "all" stratum means no filter (the full sample).
    """
    def _bool_or_false(d: pd.DataFrame, col: str) -> pd.Series:
        if col not in d.columns:
            return pd.Series(False, index=d.index)
        return d[col].fillna(False).astype(bool)

    return {
        "all": None,
        "lat_tropics": lambda d: np.abs(d[lat_col]) < 30,
        "lat_midlat":  lambda d: (np.abs(d[lat_col]) >= 30) & (np.abs(d[lat_col]) < 60),
        "lat_polar":   lambda d: np.abs(d[lat_col]) >= 60,
        **_polar_strata(lat_col),
        # Surface type: ORAC `lsflag` (0 = ocean, 1 = land). Pixel-aggregate
        # rows take the modal value via mean>0.5 (one row per pixel; equivalent).
        "ocean": lambda d: (d[lsflag_col] < 0.5) if lsflag_col in d.columns else pd.Series(False, index=d.index),
        "land":  lambda d: (d[lsflag_col] >= 0.5) if lsflag_col in d.columns else pd.Series(False, index=d.index),
        "dist_lt2km":  lambda d: d[dist_col] < 2,
        "dist_2_5km":  lambda d: (d[dist_col] >= 2) & (d[dist_col] < 5),
        "dist_ge5km":  lambda d: d[dist_col] >= 5,
        "tdiff_lt3min": lambda d: d[tdiff_col] < 180,
        "tdiff_ge3min": lambda d: d[tdiff_col] >= 180,
        "not_attenuated": lambda d: ~_bool_or_false(d, "attenuated"),
        "attenuated":     lambda d:  _bool_or_false(d, "attenuated"),
        # Passive-equivalent subset: ATLID τ > 0.3 (Karlsson 2013, PVIR v6).
        # SEVIRI's nominal cloud-detection threshold; the honest passive
        # comparison stratum.
        "tau_passive":    lambda d: d[cot_atlid_col] > 0.3,
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
    drop_attenuated: bool = True,
    ice_only: bool = True,
) -> dict[str, pd.DataFrame]:
    """End-to-end cot statistics on a matches DataFrame.

    ``base_filter`` selects the rows used as input to both views (e.g.
    cloudy in ORAC + ATLID > 0 + not saturated). If ``None``, defaults to
    ``cldmask_orac == 1 & cot_atlid > 0 & ~cot_orac_saturated``.

    ``drop_attenuated`` removes profiles flagged as lidar-saturated from
    the headline stats — they are lower bounds on τ and a fair point-by-
    point comparison with ORAC isn't possible. The dropped rows are still
    reported separately as ``sample_attenuated`` / ``pixel_attenuated``.

    ``ice_only`` filters to ORAC ice-phase pixels (``phase_orac == 2``).
    ATLID at 355 nm extincts in the first ~100 m of liquid cloud and the
    inversion's lidar-ratio assumption is unreliable in liquid; ice clouds
    are the only regime where ATLID column τ is comparable to ORAC.

    Returns a dict with keys ``sample`` and ``pixel`` mapping to the
    stratified-stats tables for each view, plus optional
    ``sample_attenuated`` / ``pixel_attenuated`` if attenuated rows exist.
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
            if ice_only and "phase_orac" in d.columns:
                mask &= d["phase_orac"] == 2
            mask &= d["cot_atlid"].notna() & d["cot_orac"].notna()
            return mask

    base = matches[base_filter(matches)]
    out: dict[str, pd.DataFrame] = {}

    if drop_attenuated and "attenuated" in base.columns:
        att_mask = base["attenuated"].fillna(False).astype(bool)
        base_main = base[~att_mask]
        att_rows = base[att_mask]
    else:
        base_main = base
        att_rows = base.iloc[0:0]

    out["sample"] = stratified_stats(base_main, "cot_atlid", "cot_orac")
    out["pixel"] = stratified_stats(
        aggregate_to_pixel(base_main, "cot_atlid", "cot_orac"),
        "cot_atlid", "cot_orac",
    )
    if len(att_rows) > 0:
        out["sample_attenuated"] = stratified_stats(att_rows, "cot_atlid", "cot_orac")
        out["pixel_attenuated"] = stratified_stats(
            aggregate_to_pixel(att_rows, "cot_atlid", "cot_orac"),
            "cot_atlid", "cot_orac",
        )
    return out


# ---------------------------------------------------------------------------
# CTH report
# ---------------------------------------------------------------------------

def _polar_strata(lat_col: str) -> dict[str, StratumFn]:
    """Polar sub-bands and hemisphere splits. All empty away from the poles, so
    they only populate for the SLSTR × EarthCARE (crossing-limited) collocation,
    where the whole sample lives in |lat| 70–83°. Additive: they add rows to the
    stats table and are ignored (N=0) for the geostationary SEVIRI collocation."""
    return {
        "lat_70_75": lambda d: (np.abs(d[lat_col]) >= 70) & (np.abs(d[lat_col]) < 75),
        "lat_75_80": lambda d: (np.abs(d[lat_col]) >= 75) & (np.abs(d[lat_col]) < 80),
        "lat_80_85": lambda d: (np.abs(d[lat_col]) >= 80) & (np.abs(d[lat_col]) < 85),
        "nh": lambda d: d[lat_col] > 0,
        "sh": lambda d: d[lat_col] < 0,
    }


def _surface_strata(lsflag_col: str, lat_col: str, dist_col: str,
                    tdiff_col: str) -> dict[str, StratumFn | None]:
    return {
        "all": None,
        "ocean": lambda d: (d[lsflag_col] < 0.5) if lsflag_col in d.columns else pd.Series(False, index=d.index),
        "land":  lambda d: (d[lsflag_col] >= 0.5) if lsflag_col in d.columns else pd.Series(False, index=d.index),
        "lat_tropics": lambda d: np.abs(d[lat_col]) < 30,
        "lat_midlat":  lambda d: (np.abs(d[lat_col]) >= 30) & (np.abs(d[lat_col]) < 60),
        "lat_polar":   lambda d: np.abs(d[lat_col]) >= 60,
        **_polar_strata(lat_col),
        "dist_lt2km":   lambda d: d[dist_col] < 2,
        "dist_2_5km":   lambda d: (d[dist_col] >= 2) & (d[dist_col] < 5),
        "dist_ge5km":   lambda d: d[dist_col] >= 5,
        "tdiff_lt3min": lambda d: d[tdiff_col] < 180,
        "tdiff_ge3min": lambda d: d[tdiff_col] >= 180,
    }


def _composition_strata(phase_col: str = "phase_orac") -> dict[str, StratumFn]:
    """Pixel-composition strata derived from the phase-fraction columns
    emitted by :func:`aggregate_to_pixel_water`. Apples-to-apples filters
    that lift phase composition from per-profile (where ATLID is nadir-
    narrow) to per-SEVIRI-pixel (the actual ORAC footprint).
    """
    def _frac(col: str) -> StratumFn:
        return lambda d: (d[col] if col in d.columns else pd.Series(0.0, index=d.index))

    return {
        "water_pure_pixel":     lambda d: _frac("frac_liquid_only")(d) >= 0.8,
        "water_dominant_pixel": lambda d: (_frac("frac_liquid_only")(d) >= 0.5) & (_frac("frac_liquid_only")(d) < 0.8),
        "mixed_pixel":          lambda d: _frac("frac_mixed")(d) >= 0.3,
        "not_water_pixel":      lambda d: _frac("frac_liquid_only")(d) < 0.3,
        # Phase-agreement subsets, restricted to water-pure pixels.
        "phase_agree_liquid": lambda d: (
            (_frac("frac_liquid_only")(d) >= 0.8)
            & ((d[phase_col] == 1) if phase_col in d.columns else pd.Series(False, index=d.index))
        ),
        "phase_disagree_atlid_liquid": lambda d: (
            (_frac("frac_liquid_only")(d) >= 0.8)
            & ((d[phase_col] == 2) if phase_col in d.columns else pd.Series(False, index=d.index))
        ),
        # CPR contribution (pixel-level: max==0 → CPR assimilated for every
        # profile in the pixel).
        "atlid_radar_synergy": lambda d: (d["cpr_assim_status"] == 0) if "cpr_assim_status" in d.columns else pd.Series(False, index=d.index),
        "atlid_only":          lambda d: (d["cpr_assim_status"] != 0) if "cpr_assim_status" in d.columns else pd.Series(False, index=d.index),
    }


def _sampling_strata() -> dict[str, StratumFn]:
    """Sub-pixel sampling / heterogeneity strata for pixel-aware experiments."""
    return {
        "n_liq_eq1":   lambda d: (d["n_liquid_only"] == 1) if "n_liquid_only" in d.columns else pd.Series(False, index=d.index),
        "n_liq_eq2":   lambda d: (d["n_liquid_only"] == 2) if "n_liquid_only" in d.columns else pd.Series(False, index=d.index),
        "n_liq_3_4":   lambda d: ((d["n_liquid_only"] >= 3) & (d["n_liquid_only"] <= 4)) if "n_liquid_only" in d.columns else pd.Series(False, index=d.index),
        "n_liq_ge5":   lambda d: (d["n_liquid_only"] >= 5) if "n_liquid_only" in d.columns else pd.Series(False, index=d.index),
        "het_cv_lt0p25":      lambda d: (d["ref_cv_atlid"] < 0.25) if "ref_cv_atlid" in d.columns else pd.Series(False, index=d.index),
        "het_cv_0p25_0p75":   lambda d: ((d["ref_cv_atlid"] >= 0.25) & (d["ref_cv_atlid"] < 0.75)) if "ref_cv_atlid" in d.columns else pd.Series(False, index=d.index),
        "het_cv_ge0p75":      lambda d: (d["ref_cv_atlid"] >= 0.75) if "ref_cv_atlid" in d.columns else pd.Series(False, index=d.index),
    }


def cot_water_strata(
    cot_atlid_col: str = "cot_water_atlid",
    lat_col: str = "ec_lat",
    dist_col: str = "distance_km",
    tdiff_col: str = "time_diff_s",
    lsflag_col: str = "lsflag_orac",
    phase_col: str = "phase_orac",
) -> dict[str, StratumFn | None]:
    """Stratification for water-cloud cot validation.

    Standard surface / lat / dist / Δt strata, plus pixel-composition
    strata (water_pure_pixel = headline; phase_agree_liquid / phase_disagree;
    mixed_pixel / not_water_pixel), τ bands on the pixel ATLID value, and
    CPR-contribution strata.
    """
    strata = _surface_strata(lsflag_col, lat_col, dist_col, tdiff_col)
    strata.update(_composition_strata(phase_col))
    strata.update(_sampling_strata())
    strata.update({
        "tau_thin":       lambda d: (d[cot_atlid_col] >= 0.15) & (d[cot_atlid_col] < 1),
        "tau_mid":        lambda d: (d[cot_atlid_col] >= 1) & (d[cot_atlid_col] < 3),
        "tau_thick":      lambda d: (d[cot_atlid_col] >= 3) & (d[cot_atlid_col] < 10),
        "tau_very_thick": lambda d: d[cot_atlid_col] >= 10,
    })
    return strata


def cer_water_strata(
    cer_atlid_col: str = "cer_water_atlid",
    lat_col: str = "ec_lat",
    dist_col: str = "distance_km",
    tdiff_col: str = "time_diff_s",
    lsflag_col: str = "lsflag_orac",
    phase_col: str = "phase_orac",
) -> dict[str, StratumFn | None]:
    """Stratification for water-cloud cer validation. CER bands instead of τ-bands;
    same composition strata as :func:`cot_water_strata`.
    """
    strata = _surface_strata(lsflag_col, lat_col, dist_col, tdiff_col)
    strata.update(_composition_strata(phase_col))
    strata.update(_sampling_strata())
    strata.update({
        "cer_small":  lambda d: d[cer_atlid_col] < 8,
        "cer_mid":    lambda d: (d[cer_atlid_col] >= 8) & (d[cer_atlid_col] < 16),
        "cer_large":  lambda d: d[cer_atlid_col] >= 16,
    })
    return strata


def aggregate_to_pixel_water(
    matches: pd.DataFrame,
    var_atlid: str,
    var_orac: str,
) -> pd.DataFrame:
    """Pixel-aggregate for water-cloud variables — apples-to-apples comparison.

    ATLID-side value (``var_atlid``) is the mean over **only the
    strict-liquid profiles** in the pixel (per-profile ``liquid_only_atlid``
    True). Mixed-phase, ice-only and clear profiles are excluded from the
    mean — pixels with no strict-liquid profile come out as NaN and drop
    out of the comparison naturally.

    Phase composition is emitted as profile counts and fractions
    (``frac_liquid_only``, ``frac_mixed``, ``frac_ice_only``,
    ``frac_clear``) so the pixel-composition strata in
    :func:`cot_water_strata` can pick out water-pure pixels for the
    headline comparison.

    Required columns in ``matches``: ``liquid_only_atlid``,
    ``liquid_present_atlid``, ``ice_present_atlid`` (all bool/0-1).
    """
    df = matches.copy()

    liquid_only = df["liquid_only_atlid"].astype(bool)
    liquid_present = df["liquid_present_atlid"].astype(bool)
    ice_present = df["ice_present_atlid"].astype(bool)

    # Mask the ATLID reference so the pixel mean is over liquid_only profiles.
    df[var_atlid] = df[var_atlid].where(liquid_only)

    # Phase-class indicators for partition counting (mutually exclusive).
    df["_n_liquid_only"] = liquid_only.astype(np.int8)
    df["_n_mixed"] = (liquid_present & ice_present).astype(np.int8)
    df["_n_ice_only"] = (~liquid_present & ice_present).astype(np.int8)
    df["_n_clear"] = (~liquid_present & ~ice_present).astype(np.int8)

    extras_first = [c for c in ("lsflag_orac", "phase_orac") if c in df.columns]
    agg = {
        var_atlid: (var_atlid, "mean"),
        var_orac: (var_orac, "mean"),
        "ref_std_atlid": (var_atlid, "std"),
        "ref_min_atlid": (var_atlid, "min"),
        "ref_max_atlid": (var_atlid, "max"),
        "n_total": (var_orac, "size"),
        "n_liquid_only": ("_n_liquid_only", "sum"),
        "n_mixed":       ("_n_mixed", "sum"),
        "n_ice_only":    ("_n_ice_only", "sum"),
        "n_clear":       ("_n_clear", "sum"),
        "ec_lat": ("ec_lat", "mean"),
        "distance_km": ("distance_km", "mean"),
        "time_diff_s": ("time_diff_s", "mean"),
        "quality_status_atlid": ("quality_status_atlid", "max"),
        "convergence_status_atlid": ("convergence_status_atlid", "max"),
        "cpr_assim_status": ("cpr_assim_status", "max"),
        "atlid_assim_status": ("atlid_assim_status", "max"),
        **{c: (c, "first") for c in extras_first},
    }
    out = df.groupby(["sev_scan_time", "sev_pixel_id"], as_index=False).agg(**agg)

    n_total = out["n_total"].astype(np.float64).clip(lower=1)
    out["frac_liquid_only"] = out["n_liquid_only"] / n_total
    out["frac_mixed"]       = out["n_mixed"] / n_total
    out["frac_ice_only"]    = out["n_ice_only"] / n_total
    out["frac_clear"]       = out["n_clear"] / n_total
    out["ref_range_atlid"] = out["ref_max_atlid"] - out["ref_min_atlid"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ref_cv_atlid"] = np.where(
            np.isfinite(out[var_atlid]) & (np.abs(out[var_atlid]) > 0),
            out["ref_std_atlid"] / np.abs(out[var_atlid]),
            np.nan,
        )
    return out


def dedupe_to_sample_water(matches: pd.DataFrame) -> pd.DataFrame:
    """Sample-level view: one row per SEVIRI pixel, nearest ATLID profile.
    Mirrors :func:`dedupe_to_sample` — kept separate so the cth path is
    untouched if rules diverge later.
    """
    idx = matches.groupby("sev_pixel_id")["distance_km"].idxmin()
    return matches.loc[idx].reset_index(drop=True)


def filter_water_sampling(
    matches: pd.DataFrame,
    var_atlid: str,
    var_orac: str,
    *,
    min_n_liquid_only: int = 1,
    min_n_total: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate raw matches with per-pixel sampling metadata and apply a pixel filter.

    The filtered raw rows are suitable for ``dedupe_to_sample_water`` while the
    second return value is the already-aggregated pixel table for the same set
    of retained ORAC pixels.
    """
    pixel = aggregate_to_pixel_water(matches, var_atlid, var_orac)
    if pixel.empty:
        return matches.iloc[0:0].copy(), pixel

    keep = (
        (pixel["n_liquid_only"] >= min_n_liquid_only)
        & (pixel["n_total"] >= min_n_total)
    )
    pixel = pixel[keep].reset_index(drop=True)
    if pixel.empty:
        return matches.iloc[0:0].copy(), pixel

    meta_cols = [
        "sev_scan_time", "sev_pixel_id",
        "n_total", "n_liquid_only", "n_mixed", "n_ice_only", "n_clear",
        "frac_liquid_only", "frac_mixed", "frac_ice_only", "frac_clear",
        "ref_std_atlid", "ref_min_atlid", "ref_max_atlid",
        "ref_range_atlid", "ref_cv_atlid",
    ]
    annotated = matches.merge(pixel[meta_cols], on=["sev_scan_time", "sev_pixel_id"], how="inner")
    return annotated, pixel


def homogeneity_sweep_stats(
    pixel: pd.DataFrame,
    var_atlid: str,
    var_orac: str,
    *,
    n_cuts: Sequence[int] = (1, 3, 5),
    cv_edges: Sequence[float] = (0.0, 0.25, 0.75, float("inf")),
) -> pd.DataFrame:
    """Stratify a pixel-aggregate table by (n_liquid_only cut, homogeneity bin).

    For each ``min_n_liquid_only`` threshold in ``n_cuts`` and each
    ``ref_cv_atlid`` bin in ``cv_edges``, compute bias / RMSE / MAE / R / N
    using :func:`continuous_stats`. Pixels with non-finite ``ref_cv_atlid``
    are dropped (single-profile pixels with zero std).
    """
    rows: list[dict] = []
    base = pixel.dropna(subset=[var_atlid, var_orac, "ref_cv_atlid", "n_liquid_only"]).copy()
    edges = list(cv_edges)
    for n_cut in n_cuts:
        sub = base[base["n_liquid_only"] >= n_cut]
        for lo, hi in zip(edges[:-1], edges[1:]):
            bin_label = _format_cv_bin(lo, hi)
            cell = sub[(sub["ref_cv_atlid"] >= lo) & (sub["ref_cv_atlid"] < hi)]
            stats = continuous_stats(cell, var_atlid, var_orac)
            rows.append({
                "n_cut": int(n_cut),
                "cv_bin": bin_label,
                "cv_lo": float(lo),
                "cv_hi": float(hi),
                **stats,
            })
    return pd.DataFrame(rows)


def _format_cv_bin(lo: float, hi: float) -> str:
    def _fmt(x: float) -> str:
        return "inf" if not np.isfinite(x) else (f"{x:g}")
    return f"[{_fmt(lo)},{_fmt(hi)})"


def _water_report(
    matches: pd.DataFrame,
    var_atlid: str,
    var_orac: str,
    strata_factory: Callable[[], dict[str, StratumFn | None]],
    qc_modes: tuple[str, ...] = ("qc_off", "qc_strict", "qc_relaxed"),
    min_n_liquid_only: int = 1,
    min_n_total: int = 1,
) -> pd.DataFrame:
    """Generic engine for cot_water / cer_water reports."""
    base_mask = (
        matches["valid_match"]
        & (matches["cldmask_orac"] == 1)
        & matches[var_atlid].notna()
        & matches[var_orac].notna()
    )
    base = matches[base_mask].copy()

    rows: list[pd.DataFrame] = []
    for qc in qc_modes:
        if qc not in SYNERGY_QC_MODES:
            raise ValueError(f"Unknown qc_mode {qc!r}; known: {list(SYNERGY_QC_MODES)}")
        qc_mask = SYNERGY_QC_MODES[qc](base).fillna(False)
        sub = base[qc_mask]
        if sub.empty:
            continue
        annotated, pixel = filter_water_sampling(
            sub, var_atlid, var_orac,
            min_n_liquid_only=min_n_liquid_only,
            min_n_total=min_n_total,
        )
        if annotated.empty or pixel.empty:
            continue
        sample = dedupe_to_sample_water(annotated)
        s_stats = stratified_stats(sample, var_atlid, var_orac, strata=strata_factory())
        p_stats = stratified_stats(pixel, var_atlid, var_orac, strata=strata_factory())
        rows.append(s_stats.assign(qc_mode=qc, view="sample"))
        rows.append(p_stats.assign(qc_mode=qc, view="pixel"))
    if not rows:
        return pd.DataFrame(columns=[
            "qc_mode", "view", "stratum", "n", "bias", "rmse", "mae",
            "r", "r_log", "slope", "intercept",
        ])
    return pd.concat(rows, ignore_index=True)


def cot_water_report(matches: pd.DataFrame, **kw) -> pd.DataFrame:
    """End-to-end water-cloud cot stats across QC modes × views × strata."""
    return _water_report(matches, "cot_water_atlid", "cot_orac",
                         cot_water_strata, **kw)


def cer_water_report(matches: pd.DataFrame, **kw) -> pd.DataFrame:
    """End-to-end water-cloud cer stats across QC modes × views × strata."""
    return _water_report(matches, "cer_water_atlid", "cer_orac",
                         cer_water_strata, **kw)


def cth_strata(
    cth_atlid_col: str = "cth_atlid_thick_km",
    lat_col: str = "ec_lat",
    dist_col: str = "distance_km",
    tdiff_col: str = "time_diff_s",
    lsflag_col: str = "lsflag_orac",
    class_col: str = "cloud_class_atlid",
) -> dict[str, StratumFn | None]:
    """Default stratification for cth validation. ``None`` = no filter (full sample)."""
    return {
        "all": None,
        "ocean": lambda d: (d[lsflag_col] < 0.5) if lsflag_col in d.columns else pd.Series(False, index=d.index),
        "land":  lambda d: (d[lsflag_col] >= 0.5) if lsflag_col in d.columns else pd.Series(False, index=d.index),
        "lat_tropics": lambda d: np.abs(d[lat_col]) < 30,
        "lat_midlat":  lambda d: (np.abs(d[lat_col]) >= 30) & (np.abs(d[lat_col]) < 60),
        "lat_polar":   lambda d: np.abs(d[lat_col]) >= 60,
        **_polar_strata(lat_col),
        "cth_low":  lambda d: d[cth_atlid_col] < 3,
        "cth_mid":  lambda d: (d[cth_atlid_col] >= 3) & (d[cth_atlid_col] < 7),
        "cth_high": lambda d: d[cth_atlid_col] >= 7,
        "dist_lt2km":   lambda d: d[dist_col] < 2,
        "dist_2_5km":   lambda d: (d[dist_col] >= 2) & (d[dist_col] < 5),
        "dist_ge5km":   lambda d: d[dist_col] >= 5,
        "tdiff_lt3min": lambda d: d[tdiff_col] < 180,
        "tdiff_ge3min": lambda d: d[tdiff_col] >= 180,
        # Per-profile ATLID class. cloud_class -127 / 0 / 6 are filtered by the
        # base mask (no useful CTH); 1=thick, 2=thin, 3=thin-over-thick,
        # 4=thick-over-thick, 5=thin-over-thin.
        "class_thick":            lambda d: d[class_col] == 1 if class_col in d.columns else pd.Series(False, index=d.index),
        "class_thin":             lambda d: d[class_col] == 2 if class_col in d.columns else pd.Series(False, index=d.index),
        "class_thin_over_thick":  lambda d: d[class_col] == 3 if class_col in d.columns else pd.Series(False, index=d.index),
        "class_thick_over_thick": lambda d: d[class_col] == 4 if class_col in d.columns else pd.Series(False, index=d.index),
    }


# QC base filters — applied before sample-level dedupe / pixel aggregation. Each
# returns a boolean mask over a per-profile matches DataFrame.
def _qc_off(d: pd.DataFrame) -> pd.Series:
    """All cloudy-and-paired rows; no ATLID QC restriction."""
    return pd.Series(True, index=d.index)


def _qc_strict(d: pd.DataFrame) -> pd.Series:
    """``quality_status == 0 & confidence >= 5 & cth_thick ≤ trop + 2 km``."""
    return (
        (d["quality_status_atlid"] == 0)
        & (d["confidence_atlid"] >= 5)
        & (d["cth_atlid_thick_km"] <= d["tropopause_km_atlid"] + 2)
    )


def _qc_relaxed(d: pd.DataFrame) -> pd.Series:
    """``quality_status ∈ {0,1} & confidence >= 3 & cth_thick ≤ trop + 2 km``."""
    return (
        (d["quality_status_atlid"].isin([0, 1]))
        & (d["confidence_atlid"] >= 3)
        & (d["cth_atlid_thick_km"] <= d["tropopause_km_atlid"] + 2)
    )


def _qc_no_trop_cap(d: pd.DataFrame) -> pd.Series:
    """Strict QS+confidence, no tropopause cap (exposes stratospheric tail)."""
    return (d["quality_status_atlid"] == 0) & (d["confidence_atlid"] >= 5)


# ---------------------------------------------------------------------------
# Synergy water-cloud QC modes (shared by cot_water and cer_water reports)
# ---------------------------------------------------------------------------

def _synergy_qc_off(d: pd.DataFrame) -> pd.Series:
    """All cloudy-and-paired rows; no ACM-CAP QC restriction."""
    return pd.Series(True, index=d.index)


def _synergy_qc_strict(d: pd.DataFrame) -> pd.Series:
    """``quality_status == 0`` (variational converged with all observations).

    No phase filter — phase composition is now applied as a stratum at
    pixel level (see :func:`cot_water_strata`).
    """
    return d["quality_status_atlid"] == 0


def _synergy_qc_relaxed(d: pd.DataFrame) -> pd.Series:
    """``quality_status ∈ {0, 1}`` — accept "unconverged but usable" too."""
    return d["quality_status_atlid"].isin([0, 1])


SYNERGY_QC_MODES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "qc_off":     _synergy_qc_off,
    "qc_strict":  _synergy_qc_strict,
    "qc_relaxed": _synergy_qc_relaxed,
}


CTH_QC_MODES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "qc_off":         _qc_off,
    "qc_strict":      _qc_strict,
    "qc_relaxed":     _qc_relaxed,
    "qc_no_trop_cap": _qc_no_trop_cap,
}


def aggregate_to_pixel_cth(matches: pd.DataFrame) -> pd.DataFrame:
    """Pixel-aggregate for CTH. ATLID side = mean over cloudy profiles, ORAC
    side = ``first`` (constant within pixel by construction).

    Clear-sky ATLID profiles (``quality_status == -1``) carry NaN
    ``cth_atlid_thick_km`` from :func:`cth_from_acth` — pandas `mean`
    skips NaN, so the aggregation is automatically over cloudy profiles
    only. ``n_atlid`` is total profiles in the pixel; ``n_atlid_cloudy``
    is the count of cloudy ones (the cloud fraction = the ratio).

    Independent of :func:`aggregate_to_pixel`, which uses ``mean`` for
    both var_atlid and var_orac and is shaped around cot.
    """
    extras_first = [c for c in ("lsflag_orac", "phase_orac",
                                "cth_orac_uncertainty_km",
                                "cth_orac_corrected_uncertainty_km") if c in matches.columns]
    agg = {
        "cth_atlid_thick_km": ("cth_atlid_thick_km", "mean"),
        "cth_atlid_raw_km":   ("cth_atlid_raw_km", "mean"),
        "cth_orac_corrected_km": ("cth_orac_corrected_km", "first"),
        "cth_orac_km":           ("cth_orac_km", "first"),
        "n_atlid": ("cth_atlid_thick_km", "size"),
        "n_atlid_cloudy": ("cth_atlid_thick_km", "count"),
        "ec_lat": ("ec_lat", "mean"),
        "distance_km": ("distance_km", "mean"),
        "time_diff_s": ("time_diff_s", "mean"),
        # Class within a SEVIRI pixel is rarely mixed; first is good enough.
        "cloud_class_atlid": ("cloud_class_atlid", "first"),
        "quality_status_atlid": ("quality_status_atlid", "max"),  # worst-of
        "confidence_atlid": ("confidence_atlid", "min"),          # worst-of
        "tropopause_km_atlid": ("tropopause_km_atlid", "mean"),
        **{c: (c, "first") for c in extras_first},
    }
    return matches.groupby(["sev_scan_time", "sev_pixel_id"], as_index=False).agg(**agg)


def dedupe_to_sample(matches: pd.DataFrame) -> pd.DataFrame:
    """Sample-level view: one row per SEVIRI pixel, nearest ATLID profile.

    Selection: per ``sev_pixel_id`` pick the row with the smallest
    ``distance_km`` (haversine to pixel centre). All columns from that row
    are kept as-is — no aggregation.
    """
    idx = matches.groupby("sev_pixel_id")["distance_km"].idxmin()
    return matches.loc[idx].reset_index(drop=True)


def cth_report(
    matches: pd.DataFrame,
    qc_modes: tuple[str, ...] = ("qc_off", "qc_strict", "qc_relaxed", "qc_no_trop_cap"),
    var_atlid: str = "cth_atlid_thick_km",
    var_orac: str = "cth_orac_corrected_km",
) -> pd.DataFrame:
    """End-to-end CTH stratified stats across QC modes and views.

    For each ``qc_mode`` the matches DataFrame is filtered (base = cloudy in
    ORAC, finite ATLID/ORAC CTH, valid_match, plus the QC predicate), then
    two views are built:

    - ``sample`` — dedupe to one row per SEVIRI pixel, nearest ATLID profile.
    - ``pixel``  — groupby ``sev_pixel_id`` with ATLID = ``max``.

    Each view is then stratified with :func:`cth_strata`. Returns a single
    tidy table with columns ``(qc_mode, view, stratum, n, bias, rmse, mae,
    r, slope, intercept)``.
    """
    base_mask = (
        matches["valid_match"]
        & (matches["cldmask_orac"] == 1)
        & matches[var_atlid].notna()
        & matches[var_orac].notna()
    )
    base = matches[base_mask].copy()

    rows: list[pd.DataFrame] = []
    for qc in qc_modes:
        if qc not in CTH_QC_MODES:
            raise ValueError(f"Unknown qc_mode {qc!r}; known: {list(CTH_QC_MODES)}")
        qc_mask = CTH_QC_MODES[qc](base).fillna(False)
        sub = base[qc_mask]
        if sub.empty:
            continue

        sample = dedupe_to_sample(sub)
        pixel = aggregate_to_pixel_cth(sub)

        s_stats = stratified_stats(sample, var_atlid, var_orac, strata=cth_strata())
        p_stats = stratified_stats(pixel, var_atlid, var_orac, strata=cth_strata())
        rows.append(s_stats.assign(qc_mode=qc, view="sample"))
        rows.append(p_stats.assign(qc_mode=qc, view="pixel"))

    if not rows:
        return pd.DataFrame(columns=[
            "qc_mode", "view", "stratum", "n", "bias", "rmse", "mae",
            "r", "r_log", "slope", "intercept",
        ])
    return pd.concat(rows, ignore_index=True)
