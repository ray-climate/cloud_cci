"""Filename parsing and slot discovery for ORAC SEVIRI L2 output.

The data tree is::

    <root>/YYYY/MM/DD/HHMM/
        ESACCI-L2-CLOUD-CLD-SEVIRI_ORAC_MSG3_YYYYMMDDhhmm_R10.primary.nc
        ESACCI-L2-CLOUD-CLD-SEVIRI_ORAC_MSG3_YYYYMMDDhhmm_R10.secondary.nc
        ESACCI-L2-CLOUD-CLD-SEVIRI_ORAC_MSG3_YYYYMMDDhhmm_R11.primary.nc
        ESACCI-L2-CLOUD-CLD-SEVIRI_ORAC_MSG3_YYYYMMDDhhmm_R11.secondary.nc

The HHMM folder name is the SEVIRI slot label; the filename timestamp is the
scan start (~12 min later). We treat the filename timestamp as authoritative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

Retrieval = Literal["R10", "R11"]
Kind = Literal["primary", "secondary"]

_FILENAME_RE = re.compile(
    r"^ESACCI-L2-CLOUD-CLD-SEVIRI_ORAC_(?P<platform>[A-Z0-9]+)"
    r"_(?P<ts>\d{12})_(?P<retrieval>R\d+)\.(?P<kind>primary|secondary)\.nc$"
)


@dataclass(frozen=True)
class OracFile:
    """A single ORAC output file."""

    path: Path
    platform: str
    retrieval: Retrieval
    kind: Kind
    scan_time: datetime
    slot_folder: str

    @property
    def date(self) -> str:
        return self.scan_time.strftime("%Y-%m-%d")


@dataclass
class SlotRecord:
    """A single 15-min SEVIRI slot with whichever of the 4 files are present."""

    scan_time: datetime
    slot_folder: str
    slot_dir: Path
    files: dict[tuple[Retrieval, Kind], OracFile] = field(default_factory=dict)

    def get(self, retrieval: Retrieval, kind: Kind) -> OracFile | None:
        return self.files.get((retrieval, kind))

    def has(self, retrieval: Retrieval) -> bool:
        return (retrieval, "primary") in self.files and (retrieval, "secondary") in self.files

    def missing(self) -> list[tuple[Retrieval, Kind]]:
        expected = [(r, k) for r in ("R10", "R11") for k in ("primary", "secondary")]
        return [key for key in expected if key not in self.files]


def parse_filename(path: str | Path) -> OracFile:
    """Parse an ORAC SEVIRI filename into an :class:`OracFile`."""
    path = Path(path)
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Not an ORAC SEVIRI filename: {path.name}")
    ts = datetime.strptime(m.group("ts"), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    slot_folder = path.parent.name if path.parent.name.isdigit() else ""
    return OracFile(
        path=path,
        platform=m.group("platform"),
        retrieval=m.group("retrieval"),  # type: ignore[arg-type]
        kind=m.group("kind"),  # type: ignore[arg-type]
        scan_time=ts,
        slot_folder=slot_folder,
    )


def discover_slots(
    root: str | Path,
    start: datetime | None = None,
    end: datetime | None = None,
    retrievals: Iterable[Retrieval] = ("R10", "R11"),
) -> list[SlotRecord]:
    """Walk ``root/YYYY/MM/DD/HHMM`` and return one :class:`SlotRecord` per slot.

    ``start`` and ``end`` are inclusive/exclusive on the scan time (UTC). Missing
    files are not an error; they are simply absent from ``SlotRecord.files``.
    """
    root = Path(root)
    retrievals = tuple(retrievals)
    if start is not None and start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end is not None and end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    slots: dict[Path, SlotRecord] = {}

    # The tree is shallow enough that a glob over slot dirs is cheap.
    for slot_dir in sorted(root.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/[0-9][0-9][0-9][0-9]")):
        for f in slot_dir.iterdir():
            if not f.name.endswith(".nc"):
                continue
            try:
                of = parse_filename(f)
            except ValueError:
                continue
            if of.retrieval not in retrievals:
                continue
            if start is not None and of.scan_time < start:
                continue
            if end is not None and of.scan_time >= end:
                continue
            rec = slots.get(slot_dir)
            if rec is None:
                rec = SlotRecord(scan_time=of.scan_time, slot_folder=of.slot_folder, slot_dir=slot_dir)
                slots[slot_dir] = rec
            rec.files[(of.retrieval, of.kind)] = of

    return sorted(slots.values(), key=lambda r: r.scan_time)


def expected_slot_count(year: int, month: int) -> int:
    """Return the number of 15-min slots in a given calendar month (96 × days)."""
    from calendar import monthrange

    return 96 * monthrange(year, month)[1]
