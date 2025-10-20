from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def extract_first_text(root: ET.Element, suffix: str) -> str:
    for elem in root.iter():
        if elem.tag.endswith(suffix) and elem.text:
            return elem.text.strip()
    raise ValueError(f"No element ending with {suffix!r} found")


def collect_polygons(root: ET.Element, suffix: str = "posList") -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    for elem in root.iter():
        if elem.tag.endswith(suffix) and elem.text:
            parts = elem.text.strip().split()
            if len(parts) % 2:
                raise ValueError("Coordinate list must contain an even number of values")
            coords: list[tuple[float, float]] = []
            for index in range(0, len(parts), 2):
                try:
                    lat = float(parts[index])
                    lon = float(parts[index + 1])
                except ValueError as exc:
                    raise ValueError(f"Non-numeric coordinate in {suffix}") from exc
                coords.append((lat, lon))
            polygons.append(coords)
    if not polygons:
        raise ValueError(f"No elements ending with {suffix!r} found")
    return polygons


def compute_central_time(start_iso: str, stop_iso: str) -> datetime:
    start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    stop = datetime.fromisoformat(stop_iso.replace("Z", "+00:00"))
    midpoint = start + (stop - start) / 2
    return midpoint.astimezone(timezone.utc)


def flatten_polygons(polygons: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    return [coord for polygon in polygons for coord in polygon]


def compute_central_coordinates(polygons: list[list[tuple[float, float]]]) -> tuple[float, float]:
    flat = flatten_polygons(polygons)
    if not flat:
        raise ValueError("Polygon coordinate list is empty")
    lats = [lat for lat, _ in flat]
    lons = [lon for _, lon in flat]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def ensure_closed_ring(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not coords:
        return coords
    if coords[0] == coords[-1]:
        return coords
    return coords + [coords[0]]


def coords_to_ring(coords: list[tuple[float, float]]) -> list[list[float]]:
    ring = ensure_closed_ring(coords)
    return [[lon, lat] for lat, lon in ring]


def polygons_to_geometry(polygons: list[list[tuple[float, float]]]) -> dict[str, object]:
    if len(polygons) == 1:
        return {"type": "Polygon", "coordinates": [coords_to_ring(polygons[0])]}
    return {
        "type": "MultiPolygon",
        "coordinates": [[coords_to_ring(polygon)] for polygon in polygons],
    }


def safe_filename_from_time(central_time: str, suffix: str = ".geojson") -> str:
    safe = "".join(ch for ch in central_time if ch.isalnum())
    if not safe:
        raise ValueError("Central time string produced an empty filename")
    return f"{safe}{suffix}"


def resolve_target_path(base: Path) -> Path:
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = base.with_name(f"{base.stem}_{index}{base.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def write_polygon_file(
    central_time: str,
    polygons: list[list[tuple[float, float]]],
    manifest_path: Path,
    polygon_dir: Path | None,
) -> Path:
    destination = polygon_dir or manifest_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    filename = safe_filename_from_time(central_time)
    target = resolve_target_path(destination / filename)
    feature = {
        "type": "Feature",
        "geometry": polygons_to_geometry(polygons),
        "properties": {
            "central_time": central_time,
            "manifest": manifest_path.name,
        },
    }
    target.write_text(json.dumps(feature, indent=2))
    return target


def manifest_group_key(path: Path) -> str:
    parts = path.name.split("_")
    if len(parts) >= 9:
        return "_".join(parts[:9])
    return path.stem


def manifest_priority_key(path: Path) -> tuple[str, str]:
    parts = path.name.split("_")
    third_time = parts[9] if len(parts) > 9 else ""
    return third_time, path.name


def select_latest_manifests(paths: list[Path]) -> list[Path]:
    chosen: dict[str, Path] = {}
    for candidate in paths:
        key = manifest_group_key(candidate)
        if key not in chosen or manifest_priority_key(candidate) > manifest_priority_key(chosen[key]):
            chosen[key] = candidate
    return list(chosen.values())


def parse_manifest(path: Path) -> tuple[str, float, float, list[list[tuple[float, float]]]]:
    root = ET.fromstring(path.read_text())
    start_iso = extract_first_text(root, "startTime")
    stop_iso = extract_first_text(root, "stopTime")
    polygons = collect_polygons(root)
    central_time = compute_central_time(start_iso, stop_iso).isoformat().replace("+00:00", "Z")
    central_lat, central_lon = compute_central_coordinates(polygons)
    return central_time, central_lat, central_lon, polygons


def date_from_string(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def run_single(manifest_path: Path, polygon_dir: Path | None) -> int:
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    central_time, central_lat, central_lon, polygons = parse_manifest(manifest_path)
    destination_dir = polygon_dir or Path.cwd() / "polygons"
    polygon_path = write_polygon_file(central_time, polygons, manifest_path, destination_dir)
    print(f"central_time: {central_time}")
    print(f"central_lat: {central_lat}")
    print(f"central_lon: {central_lon}")
    print(f"polygon_file: {polygon_path}")
    return 0


def run_batch(
    root: Path,
    start_date: date,
    end_date: date,
    output: Path,
    polygon_dir: Path | None,
) -> int:
    records = []
    day = start_date
    step = timedelta(days=1)
    destination_dir = polygon_dir or output.parent / "polygons"
    while day <= end_date:
        day_dir = root / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
        day_label = day.isoformat()
        if not day_dir.is_dir():
            print(f"{day_label}: no manifests found")
            day += step
            continue
        candidates = list(day_dir.rglob("*.manifest"))
        if not candidates:
            print(f"{day_label}: no manifests found")
            day += step
            continue
        selected = select_latest_manifests(candidates)
        print(f"{day_label}: processing {len(selected)} manifest(s)")
        for manifest_path in sorted(selected, key=lambda item: item.name):
            central_time, central_lat, central_lon, polygons = parse_manifest(manifest_path)
            polygon_path = write_polygon_file(central_time, polygons, manifest_path, destination_dir)
            try:
                polygon_file_value = str(polygon_path.relative_to(destination_dir))
            except ValueError:
                polygon_file_value = str(polygon_path)
            records.append(
                {
                    "central_time": central_time,
                    "central_lat": central_lat,
                    "central_lon": central_lon,
                    "filename": manifest_path.name,
                    "polygon_file": polygon_file_value,
                }
            )
        day += step
    if not records:
        print("No manifest files found in the requested range", file=sys.stderr)
        return 1
    records.sort(key=lambda row: row["central_time"])  # chronological csv
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "central_time",
                "central_lat",
                "central_lon",
                "filename",
                "polygon_file",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} records to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute central time and coordinates from Sentinel-3 SLSTR manifests.")
    parser.add_argument("--manifest", type=Path, help="Path to a single manifest file")
    parser.add_argument("--batch-root", type=Path, help="Root directory containing dated subfolders")
    parser.add_argument("--start-date", type=date_from_string, help="Start date (YYYY-MM-DD) inclusive")
    parser.add_argument("--end-date", type=date_from_string, help="End date (YYYY-MM-DD) inclusive")
    parser.add_argument("--output", type=Path, help="CSV file to write batch results")
    parser.add_argument(
        "--polygon-dir",
        type=Path,
        help="Directory where polygon GeoJSON files will be written (defaults to ./polygons or output.parent/polygons)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.batch_root or args.start_date or args.end_date or args.output:
        if not (args.batch_root and args.start_date and args.end_date and args.output):
            parser.error("--batch-root, --start-date, --end-date, and --output must be provided together for batch mode")
        return run_batch(args.batch_root, args.start_date, args.end_date, args.output, args.polygon_dir)

    manifest_path = args.manifest or Path("sample_data/S3A_SL_1_RBT____20240801T235820_20240802T000120_20240803T091114_0179_115_187_3420_PS1_O_NT_004.manifest")
    return run_single(manifest_path, args.polygon_dir)


if __name__ == "__main__":
    raise SystemExit(main())
