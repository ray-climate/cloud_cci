"""Per-profile ORAC-equivalent reference values from EarthCARE L2 products.

Each function takes the raw arrays returned by `validation.readers` and
returns a 1-D array of length n_profile, ready to join row-for-row with
the matches CSV.
"""
from __future__ import annotations

import numpy as np


def cot_from_aebd(
    extinction: np.ndarray,
    height: np.ndarray,
    quality_status: np.ndarray | None = None,
    qc_max: int = 1,
    surface_layer_top_m: float = 5000.0,
    surface_qs3_fraction: float = 0.5,
    saturation_min_tau: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-profile column optical depth from A-EBD particle extinction.

    Integrates ``extinction × dz`` over the column with quality masking.

    Parameters
    ----------
    extinction
        ``(n_profile, n_bin)`` particle extinction coefficient at 355 nm
        in m\ :sup:`-1`. NaN-filled bins are treated as zero contribution.
    height
        ``(n_profile, n_bin)`` per-profile altitude grid in metres
        (top-down, decreasing along axis=1).
    quality_status
        Optional A-EBD quality flag. Bins with ``qs > qc_max`` are
        zeroed before integration.
    qc_max
        Largest quality_status value treated as usable (default 1 keeps
        "Good" and "Likely good"; 2 and 3 are dropped).
    surface_layer_top_m
        Top of the "lower atmosphere" used by the attenuation detector
        (default 5 km — captures stratus, cumulus, nimbostratus).
    surface_qs3_fraction
        Fraction of bins in the lower-atmosphere layer that must be QS=3
        for a profile to be flagged attenuated (default 0.5).
    saturation_min_tau
        Minimum measured τ for the attenuation flag to fire (default 1.0).
        Excludes clear-sky profiles where the lidar simply has weak return
        but isn't actually saturated by an opaque cloud.

    Returns
    -------
    cot
        ``(n_profile,)`` column τ at 355 nm. For cloud particles (size
        parameter ≫ 1) Q\ :sub:`ext` → 2 in the geometric optics limit, so
        this is a fair reference for ORAC's visible-band cot.
    attenuated
        ``(n_profile,)`` boolean — True if the lidar was saturated by an
        opaque cloud above (the lower atmosphere is dominated by QS=3 AND
        the column has measurable τ). On these profiles the returned τ is
        a lower bound; downstream stats should stratify on this flag.

    Notes
    -----
    The attenuation detector uses a surface-layer-fraction-of-QS=3
    criterion rather than "any QS=3 in column" — the latter fires on
    isolated noise bins anywhere in the profile and over-flags ~99% of
    profiles. The surface-layer rule asks the physically meaningful
    question: did the lidar reach the surface? If the lower 5 km is
    mostly QS=3 ("no signal"), it didn't.
    """
    ext = np.where(np.isfinite(extinction), extinction, 0.0).astype(np.float64)
    h = np.where(np.isfinite(height), height, 0.0).astype(np.float64)

    if quality_status is not None:
        bad = quality_status > qc_max
        ext = np.where(bad, 0.0, ext)

    # Trapezoid over decreasing height → negative; take abs.
    cot = np.abs(np.trapz(ext, h, axis=1))

    if quality_status is not None:
        in_lower = (h >= 0) & (h <= surface_layer_top_m)
        n_lower = in_lower.sum(axis=1)
        n_qs3 = ((quality_status == 3) & in_lower).sum(axis=1)
        frac_qs3 = np.where(n_lower > 0, n_qs3 / n_lower, 0.0)
        attenuated = (frac_qs3 >= surface_qs3_fraction) & (cot >= saturation_min_tau)
    else:
        attenuated = np.zeros(ext.shape[0], dtype=bool)

    return cot, attenuated


def cot_cer_water_from_accap(
    liquid_optical_depth: np.ndarray,
    liquid_extinction: np.ndarray,
    liquid_eff_radius: np.ndarray,
    liquid_classification: np.ndarray,
    ice_water_content: np.ndarray,
    height: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-profile water-cloud COT and CER from ACM-CAP, plus phase-presence flags.

    No QC is applied — the matches CSV carries ``quality_status``,
    ``convergence_status``, ``synergy_status``, ``cost_function``, and the
    per-instrument assimilation statuses so QC choices are made as strata
    at evaluate time.

    The phase-presence flags are returned independently rather than as a
    derived ``liquid_only`` flag — apples-to-apples validation lifts phase
    composition to the SEVIRI-pixel level (counts/fractions across
    profiles in a pixel), so we want raw per-profile flags. ``liquid_only``
    is just ``liquid_present & ~ice_present`` and can be derived downstream
    if needed.

    Parameters
    ----------
    liquid_optical_depth
        ``(n_profile,)`` ACM-CAP ``liquid_optical_depth`` (dimensionless,
        NaN where fill). Used directly as the COT reference — no integral.
    liquid_extinction
        ``(n_profile, n_bin)`` per-bin ``liquid_extinction`` [m⁻¹], NaN fill.
    liquid_eff_radius
        ``(n_profile, n_bin)`` per-bin ``liquid_effective_radius`` [m].
    liquid_classification
        ``(n_profile, n_bin)`` int8: 0 none, 1 detected, 2 in-rain, 3 in-ice.
        Only ``== 1`` bins (clean liquid) contribute to the τ-weighted CER.
    ice_water_content
        ``(n_profile, n_bin)`` per-bin ice [kg m⁻³], NaN fill. Carried
        for completeness; ``ice_present`` is derived from
        ``liquid_classification`` instead because ACM-CAP reports IWC > 0
        in non-cloud bins (drizzle / precip / virga falling out of the
        cloud bottom) ~80% of the time, which would over-flag ice.
    height
        ``(n_profile, n_bin)`` per-profile altitude grid in m (top-down).

    Returns
    -------
    cot_water_atlid
        ``(n_profile,)`` water-cloud τ (the ACM-CAP scalar).
    cer_water_atlid
        ``(n_profile,)`` τ-weighted mean effective radius in **µm**:
        ``<r_eff> = Σ ext·r·dz / Σ ext·dz`` over clean-liquid bins.
        NaN where the denominator is zero (no clean-liquid bins).
    liquid_present
        ``(n_profile,)`` bool — True when ``liquid_optical_depth > 0``.
    ice_present
        ``(n_profile,)`` bool — True when any bin in the column has
        ``liquid_classification == 3`` ("in ice"). This is ACM-CAP's
        bin-level "detected as cloud and inside an ice region" flag.
    """
    cot = np.asarray(liquid_optical_depth, dtype=np.float64)

    ext = np.where(np.isfinite(liquid_extinction), liquid_extinction, 0.0).astype(np.float64)
    re_m = np.where(np.isfinite(liquid_eff_radius), liquid_eff_radius, 0.0).astype(np.float64)
    h = np.where(np.isfinite(height), height, 0.0).astype(np.float64)
    cls = np.asarray(liquid_classification)

    # Per-bin |dz| from a top-down grid. Edge bins use forward difference.
    dz = np.abs(np.diff(h, axis=1))
    dz = np.concatenate([dz[:, :1], dz], axis=1)

    clean_liquid = (cls == 1)
    weight = np.where(clean_liquid, ext * dz, 0.0)
    num = (weight * re_m).sum(axis=1)
    den = weight.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        re_um = np.where(den > 0, (num / den) * 1.0e6, np.nan)

    liquid_present = np.isfinite(cot) & (cot > 0)
    # ice_water_content is unused for the ice-presence flag because ACM-CAP
    # reports IWC > 0 in many non-cloud bins (drizzle / precip below cloud
    # base). Use the bin-level liquid_classification == 3 ("in ice") flag
    # instead — it's strict and physically defensible.
    ice_present = (np.asarray(liquid_classification) == 3).any(axis=1)

    return cot, re_um, liquid_present, ice_present


def cth_from_acth(
    cth_thick: np.ndarray,
    cth_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-profile ORAC-equivalent CTH from A-CTH, in **kilometres** AMSL.

    ATLID stores heights in metres above the geoid; ORAC SEVIRI
    ``cth_corrected`` is ``cloud_top_altitude`` in km AMSL. Conversion is
    m → km, no geoid shift. No QC is applied — the matches CSV carries
    ``quality_status``, ``cth_confidence``, ``cloud_class``, and
    ``tropopause_km_atlid`` so QC choices can be made as strata at
    evaluate time. Fill values arrive as NaN from :func:`read_acth_track`.

    Parameters
    ----------
    cth_thick, cth_raw
        ``(n_profile,)`` heights from :func:`read_acth_track`
        (metres AMSL, NaN where fill).

    Returns
    -------
    cth_thick_km, cth_raw_km
        ``(n_profile,)`` heights in km AMSL.
    """
    thick_km = np.asarray(cth_thick, dtype=np.float64) / 1000.0
    raw_km = np.asarray(cth_raw, dtype=np.float64) / 1000.0
    return thick_km, raw_km
