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

    Returns
    -------
    cot
        ``(n_profile,)`` column τ at 355 nm. For cloud particles (size
        parameter ≫ 1) Q\ :sub:`ext` → 2 in the geometric optics limit, so
        this is a fair reference for ORAC's visible-band cot.
    attenuated
        ``(n_profile,)`` boolean — True if any bin has QS=3 (lidar
        fully attenuated). On those profiles the returned τ is a lower
        bound; downstream stats should stratify on this flag.
    """
    ext = np.where(np.isfinite(extinction), extinction, 0.0).astype(np.float64)
    h = np.where(np.isfinite(height), height, 0.0).astype(np.float64)

    if quality_status is not None:
        bad = quality_status > qc_max
        ext = np.where(bad, 0.0, ext)
        attenuated = np.any(quality_status == 3, axis=1)
    else:
        attenuated = np.zeros(ext.shape[0], dtype=bool)

    # Trapezoid over decreasing height → negative; take abs.
    cot = np.abs(np.trapz(ext, h, axis=1))
    return cot, attenuated
