"""Decoders for ORAC bitmask / categorical flag variables.

All bit positions and category names are taken directly from the NetCDF
``flag_masks`` / ``flag_meanings`` attributes on the R10/R11 primary files.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# qcflag — 8 independent bits
# ---------------------------------------------------------------------------
QCFLAG_BITS: dict[str, int] = {
    "not_converged":             1,   # bit 0
    "cost_gt_100":               2,   # bit 1
    "snow_ice_surface":          4,   # bit 2
    "particle_type_inconsistent": 8,   # bit 3
    "dfs_gt_1":                 16,   # bit 4
    "elevation_gt_1p5km":       32,   # bit 5
    "sun_glint":                64,   # bit 6
    "hit_limit":               128,   # bit 7
}


def decode_qcflag(qc: xr.DataArray) -> xr.Dataset:
    """Return an xr.Dataset with one boolean layer per qcflag bit."""
    qc_i = qc.fillna(-1).astype("int32")
    layers = {name: ((qc_i & mask) != 0) for name, mask in QCFLAG_BITS.items()}
    out = xr.Dataset(layers)
    out.attrs["description"] = "qcflag decomposed into independent boolean layers"
    return out


def qc_pass_mask(
    qc: xr.DataArray,
    rules: Literal["default", "strict", "permissive"] = "default",
) -> xr.DataArray:
    """Return a boolean mask of pixels that pass the chosen QC rule set.

    - ``default``: exclude non-converged, cost>100, and particle-type-inconsistent
    - ``strict``: default + exclude hit_limit and sun_glint
    - ``permissive``: only exclude non-converged
    """
    qc_i = qc.fillna(-1).astype("int32")
    if rules == "default":
        bad = QCFLAG_BITS["not_converged"] | QCFLAG_BITS["cost_gt_100"] | QCFLAG_BITS["particle_type_inconsistent"]
    elif rules == "strict":
        bad = (QCFLAG_BITS["not_converged"] | QCFLAG_BITS["cost_gt_100"]
               | QCFLAG_BITS["particle_type_inconsistent"]
               | QCFLAG_BITS["hit_limit"] | QCFLAG_BITS["sun_glint"])
    elif rules == "permissive":
        bad = QCFLAG_BITS["not_converged"]
    else:
        raise ValueError(f"Unknown rules: {rules!r}")
    passed = (qc_i & bad) == 0
    passed.attrs["rules"] = rules
    return passed


# ---------------------------------------------------------------------------
# cldtype — Pavolonis cloud type (0..12)
# ---------------------------------------------------------------------------
CLDTYPE_NAMES: list[str] = [
    "clear",                    # 0
    "N/A",                      # 1
    "fog",                      # 2
    "water",                    # 3
    "supercooled",              # 4
    "mixed",                    # 5
    "opaque_ice",               # 6
    "cirrus",                   # 7
    "overlap",                  # 8
    "prob_opaque_ice",          # 9
    "N/A",                      # 10
    "dust_clear",               # 11
    "dust_switched_from_cloud", # 12
]


def cldtype_names(cldtype: xr.DataArray) -> np.ndarray:
    """Return an ndarray of category strings for each pixel (keeps shape)."""
    arr = np.asarray(cldtype)
    out = np.empty(arr.shape, dtype=object)
    for i, name in enumerate(CLDTYPE_NAMES):
        out[arr == i] = name
    out[(arr < 0) | (arr >= len(CLDTYPE_NAMES))] = "unknown"
    return out


# ---------------------------------------------------------------------------
# ann_phase, phase_pavolonis
# ---------------------------------------------------------------------------
ANN_PHASE_NAMES: dict[int, str] = {0: "clear", 1: "liquid", 2: "ice"}
PHASE_NAMES: dict[int, str] = {1: "liquid", 2: "water-ice-agg"}
PHASE_PAVOLONIS_NAMES: dict[int, str] = {0: "clear_or_unknown", 1: "liquid", 2: "ice"}


# ---------------------------------------------------------------------------
# variables_retrieved — mixed "approach" (low bits) + per-variable flags
# ---------------------------------------------------------------------------
# Bits 0..7 encode a categorical "approach"; bits 7+ are independent variable
# flags. We only decode the subset actually produced by the SEVIRI cloud run.
VARIABLES_RETRIEVED_BITS: dict[str, int] = {
    "optical_depth":         128,
    "effective_radius":      256,
    "cloud_top_pressure":    512,
    "cloud_fraction":       1024,
    "optical_depth_2":      2048,
    "effective_radius_2":   4096,
    "cloud_top_pressure_2": 8192,
    "cloud_fraction_2":    16384,
    "swansea_gamma":       32768,
    "surface_temperature": 65536,
}


def decode_variables_retrieved(var_ret: xr.DataArray) -> xr.Dataset:
    """Return boolean layers for the independent per-variable retrieval flags."""
    v = var_ret.fillna(0).astype("int64")
    return xr.Dataset({name: ((v & mask) != 0) for name, mask in VARIABLES_RETRIEVED_BITS.items()})


# ---------------------------------------------------------------------------
# channels_used — SEVIRI channels 1..11 as independent bits
# ---------------------------------------------------------------------------
CHANNELS_USED_BITS: dict[int, int] = {
    1:    2,
    2:    4,
    3:    8,
    4:   16,
    5:   32,
    6:   64,
    7:  128,
    8:  256,
    9:  512,
    10: 1024,
    11: 2048,
}


def decode_channels_used(channels: xr.DataArray) -> xr.Dataset:
    """Return boolean layers named ``ch1..ch11`` for each SEVIRI channel."""
    v = channels.fillna(0).astype("int64")
    return xr.Dataset(
        {f"ch{c}": ((v & mask) != 0) for c, mask in CHANNELS_USED_BITS.items()}
    )
