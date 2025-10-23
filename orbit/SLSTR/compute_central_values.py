from __future__ import annotations

import argparse
import csv
import json
import math
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
    return sum(lats) / len(lats), circular_mean(lons)


def circular_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Longitude list is empty")
    radians = [math.radians(value) for value in values]
    sin_sum = sum(math.sin(value) for value in radians)
    cos_sum = sum(math.cos(value) for value in radians)
    angle = math.degrees(math.atan2(sin_sum, cos_sum))
    if angle > 180:
        return angle - 360
    if angle <= -180:
        return angle + 360
    return angle


def remove_adjacent_duplicates(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for coord in coords:
        if not cleaned or coord != cleaned[-1]:
            cleaned.append(coord)
    return cleaned


def drop_duplicate_endpoint(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(coords) > 1 and coords[0] == coords[-1]:
        return coords[:-1]
    return coords


def wrap_to_180(value: float) -> float:
    wrapped = ((value + 180.0) % 360.0) - 180.0
    if wrapped == -180.0 and value > 0:
        return 180.0
    if wrapped == 180.0 and value < 0:
        return -180.0
    if abs(wrapped) < 1e-12:
        return 0.0
    return wrapped


def unwrap_longitudes(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not coords:
        return []
    unwrapped: list[tuple[float, float]] = [coords[0]]
    offset = 0.0
    prev_lon = coords[0][1]
    for lat, lon in coords[1:]:
        adjusted = lon + offset
        delta = adjusted - prev_lon
        if delta > 180:
            offset -= 360
            adjusted = lon + offset
        elif delta < -180:
            offset += 360
            adjusted = lon + offset
        unwrapped.append((lat, adjusted))
        prev_lon = adjusted
    return unwrapped


def boundaries_between(lon1: float, lon2: float) -> list[float]:
    boundaries: list[float] = []
    epsilon = 1e-9
    if lon2 > lon1 + epsilon:
        boundary = math.ceil((lon1 + epsilon) / 180.0) * 180.0
        while boundary < lon2 - epsilon:
            boundaries.append(boundary)
            boundary += 180.0
    elif lon2 < lon1 - epsilon:
        boundary = math.floor((lon1 - epsilon) / 180.0) * 180.0
        while boundary > lon2 + epsilon:
            boundaries.append(boundary)
            boundary -= 180.0
    return boundaries


def split_ring_across_meridians(coords: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    core = remove_adjacent_duplicates(drop_duplicate_endpoint(coords))
    if len(core) < 3:
        return []
    unwrapped = unwrap_longitudes(core)
    first_lat, first_lon = unwrapped[0]
    last_lon = unwrapped[-1][1]
    closure_lon = first_lon
    while closure_lon - last_lon > 180:
        closure_lon -= 360
    while closure_lon - last_lon < -180:
        closure_lon += 360
    cycle = unwrapped + [(first_lat, closure_lon)]
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = [cycle[0]]
    for index in range(1, len(cycle)):
        prev = cycle[index - 1]
        curr = cycle[index]
        prev_point = prev
        for boundary in boundaries_between(prev_point[1], curr[1]):
            denom = curr[1] - prev_point[1]
            if abs(denom) < 1e-9:
                continue
            fraction = (boundary - prev_point[1]) / denom
            boundary_lat = prev_point[0] + fraction * (curr[0] - prev_point[0])
            intersection = (boundary_lat, boundary)
            if not current or current[-1] != intersection:
                current.append(intersection)
            if len(current) >= 2:
                segments.append(current)
            if boundary % 180 == 0 and boundary != 0:
                shifted = boundary - 360 if boundary > 0 else boundary + 360
                current = [(boundary_lat, shifted)]
            else:
                current = [intersection]
            prev_point = intersection
        if index < len(cycle) - 1:
            current.append(curr)
    if len(current) >= 2:
        segments.append(current)
    return segments


def classify_zone(coords: list[tuple[float, float]]) -> int:
    epsilon = 1e-9
    positive = any((lon > epsilon) and (abs(lon) < 180 - epsilon) for _, lon in coords)
    negative = any((lon < -epsilon) and (abs(lon) < 180 - epsilon) for _, lon in coords)
    if positive and not negative:
        return 1
    if negative and not positive:
        return -1
    return 0


def align_boundary_signs(coords: list[tuple[float, float]], zone: int) -> list[tuple[float, float]]:
    aligned: list[tuple[float, float]] = []
    for lat, lon in coords:
        if abs(abs(lon) - 180.0) < 1e-9:
            if zone < 0:
                lon = -180.0
            elif zone > 0:
                lon = 180.0
        aligned.append((lat, lon))
    return aligned


def normalise_polygons(polygons: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    normalised: list[list[tuple[float, float]]] = []
    for polygon in polygons:
        segments = split_ring_across_meridians(polygon)
        if not segments:
            cleaned = remove_adjacent_duplicates(drop_duplicate_endpoint(polygon))
            if len(cleaned) >= 3:
                normalised.append(ensure_closed_ring(cleaned))
            continue

        processed: list[tuple[int, list[tuple[float, float]]]] = []
        for segment in segments:
            rewrapped = [(lat, wrap_to_180(lon)) for lat, lon in segment]
            cleaned = remove_adjacent_duplicates(rewrapped)
            if len(cleaned) < 2:
                continue
            zone = classify_zone(cleaned)
            processed.append((zone, cleaned))

        if not processed:
            continue

        grouped: list[tuple[int, list[tuple[float, float]]]] = []
        for zone, coords in processed:
            if not grouped:
                grouped.append((zone, coords[:]))
                continue
            prev_zone, prev_coords = grouped[-1]
            if zone == prev_zone or zone == 0 or prev_zone == 0:
                combined_zone = zone if prev_zone == 0 else prev_zone if zone == 0 else prev_zone
                merged = prev_coords + coords[1:]
                grouped[-1] = (combined_zone, remove_adjacent_duplicates(merged))
            else:
                grouped.append((zone, coords[:]))

        if len(grouped) > 1:
            first_zone, first_coords = grouped[0]
            last_zone, last_coords = grouped[-1]
            if first_zone == last_zone or first_zone == 0 or last_zone == 0:
                combined_zone = first_zone if first_zone != 0 else last_zone
                merged = last_coords + first_coords[1:]
                grouped[0] = (combined_zone, remove_adjacent_duplicates(merged))
                grouped.pop()

        for zone, coords in grouped:
            cleaned = remove_adjacent_duplicates(coords)
            if len(cleaned) < 3:
                continue
            aligned = align_boundary_signs(cleaned, zone)
            normalised.append(ensure_closed_ring(aligned))

    if normalised:
        return normalised
    fallback: list[list[tuple[float, float]]] = []
    for polygon in polygons:
        cleaned = remove_adjacent_duplicates(drop_duplicate_endpoint(polygon))
        if len(cleaned) >= 3:
            fallback.append(ensure_closed_ring(cleaned))
    return fallback


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
    return central_time, central_lat, central_lon, normalise_polygons(polygons)


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
