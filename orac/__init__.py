"""Reader and summariser for ORAC SEVIRI L2 cloud retrievals (R10 / R11)."""

from .metadata import (
    OracFile,
    SlotRecord,
    discover_slots,
    expected_slot_count,
    parse_filename,
)
from .io import julian_to_datetime, open_paired, open_slot, read_prior_file
from .flags import (
    CLDTYPE_NAMES,
    QCFLAG_BITS,
    cldtype_names,
    decode_channels_used,
    decode_qcflag,
    decode_variables_retrieved,
    qc_pass_mask,
)
from .summary import DEFAULT_SUMMARY_VARS, missing_slot_report, monthly_summary, per_slot_stats
from .subset import bbox_subset, nearest_pixel

__all__ = [
    "OracFile", "SlotRecord", "discover_slots", "expected_slot_count", "parse_filename",
    "julian_to_datetime", "open_paired", "open_slot", "read_prior_file",
    "CLDTYPE_NAMES", "QCFLAG_BITS", "cldtype_names",
    "decode_channels_used", "decode_qcflag", "decode_variables_retrieved", "qc_pass_mask",
    "DEFAULT_SUMMARY_VARS", "missing_slot_report", "monthly_summary", "per_slot_stats",
    "bbox_subset", "nearest_pixel",
]
