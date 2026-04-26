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
