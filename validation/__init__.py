"""ORAC × EarthCARE validation module.

Sample-level collocation of ATLID-driven L2 references against ORAC SEVIRI L2
retrievals. See docs/validation_pipeline.md for the design overview.
"""
from .collocate import match_track_to_seviri
from .readers import read_aebd_track
from .reference import cot_from_aebd

__all__ = ["read_aebd_track", "cot_from_aebd", "match_track_to_seviri"]
