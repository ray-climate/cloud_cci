#!/usr/bin/env python3
"""Extract ATLID footprint ground track positions with interpolated timestamps."""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence
import xml.etree.ElementTree as ET

GML_NS = "http://www.opengis.net/gml/3.2"
NAMESPACES = {"gml": GML_NS}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a metadata XML file or a directory containing them.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("atlid_positions.csv"),
        help="Destination CSV file (default: atlid_positions.csv).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for XML files recursively inside directories.",
    )
    return parser.parse_args(argv)


def read_metadata_file(path: Path) -> ET.Element:
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse XML from {path}: {exc}") from exc


def parse_time(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def extract_positions_with_times(root: ET.Element) -> List[tuple[float, float, datetime]]:
    begin_elem = root.find(".//gml:beginPosition", NAMESPACES)
    end_elem = root.find(".//gml:endPosition", NAMESPACES)
    if begin_elem is None or end_elem is None:
        raise ValueError("Missing gml:beginPosition or gml:endPosition in XML tree")
    begin = parse_time(begin_elem.text or "")
    end = parse_time(end_elem.text or "")

    pos_list_elem = root.find(".//gml:posList", NAMESPACES)
    if pos_list_elem is None or not pos_list_elem.text:
        raise ValueError("Missing gml:posList data in XML tree")
    values = [float(part) for part in pos_list_elem.text.split()]
    if len(values) % 2:
        raise ValueError("gml:posList does not contain pairs of coordinates")

    lats = values[0::2]
    lons = values[1::2]
    if not lats:
        raise ValueError("gml:posList does not contain any coordinates")

    positions = list(zip(lats, lons))
    count = len(positions)
    if count == 1:
        times = [begin]
    else:
        total_duration = end - begin
        step = total_duration / (count - 1)
        times = [begin + step * index for index in range(count)]
    return [(lat, lon, time) for (lat, lon), time in zip(positions, times)]


def format_timestamp(dt: datetime) -> str:
    iso_value = dt.astimezone(timezone.utc).isoformat(timespec="milliseconds")
    if iso_value.endswith("+00:00"):
        iso_value = iso_value[:-6] + "Z"
    return iso_value


def prepare_rows(path: Path) -> List[tuple[float, float, str]]:
    root = read_metadata_file(path)
    return [
        (lat, lon, format_timestamp(timestamp))
        for lat, lon, timestamp in extract_positions_with_times(root)
    ]


def iter_metadata_files(path: Path, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    pattern = "*.XML"
    if recursive:
        yield from sorted(p for p in path.rglob(pattern) if p.is_file())
    else:
        yield from sorted(p for p in path.glob(pattern) if p.is_file())


def write_csv(rows: List[tuple[float, float, str]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["latitude", "longitude", "timestamp"])
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    files = list(iter_metadata_files(args.input_path, args.recursive))
    if not files:
        raise SystemExit(f"No XML files found at {args.input_path}")

    rows: List[tuple[float, float, str]] = []
    previous_group: str | None = None
    base_dir = args.input_path if args.input_path.is_dir() else None
    for file_path in files:
        if base_dir is not None:
            try:
                relative = file_path.relative_to(base_dir)
            except ValueError:
                relative = file_path
            if relative.parts:
                if len(relative.parts) > 1:
                    group = relative.parts[0]
                elif file_path.parent == base_dir:
                    group = base_dir.name
                else:
                    group = relative.parts[0]
            else:
                group = base_dir.name
        else:
            group = file_path.name
        if group != previous_group:
            print(f"Processing {group}...", file=sys.stderr)
            previous_group = group
        try:
            file_rows = prepare_rows(file_path)
        except ValueError as exc:
            raise SystemExit(f"Failed to process {file_path}: {exc}") from exc
        rows.extend(file_rows)

    write_csv(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
