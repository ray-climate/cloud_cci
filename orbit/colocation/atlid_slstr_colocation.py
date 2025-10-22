#!/usr/bin/env python3
"""Find ATLID observations that fall inside SLSTR polygons within a time window."""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

TIME_WINDOW = timedelta(hours=4)
EPSILON = 1e-9


@dataclass(frozen=True)
class Candidate:
    dt: datetime
    path: Path
    folder: str


def parse_geojson_timestamp(path: Path) -> datetime:
    stem = path.stem
    if stem.endswith("Z"):
        stem = stem[:-1]
    if "T" not in stem:
        raise ValueError(f"Filename {path.name} does not contain a timestamp separator 'T'")
    date_part, time_part = stem.split("T", 1)
    if len(date_part) != 8:
        raise ValueError(f"Unexpected date segment in filename {path.name}")
    if len(time_part) < 6:
        raise ValueError(f"Unexpected time segment in filename {path.name}")
    base = datetime.strptime(date_part + time_part[:6], "%Y%m%d%H%M%S")
    if len(time_part) > 6:
        micros = int(time_part[6:].ljust(6, "0")[:6])
        base = base.replace(microsecond=micros)
    return base.replace(tzinfo=timezone.utc)


def build_candidates(folders: Sequence[Path]) -> tuple[List[Candidate], List[datetime]]:
    items: List[Candidate] = []
    for folder in folders:
        for path in sorted(folder.glob("*.geojson")):
            dt = parse_geojson_timestamp(path)
            items.append(Candidate(dt=dt, path=path, folder=folder.name))
    items.sort(key=lambda c: c.dt)
    timeline = [c.dt for c in items]
    return items, timeline


@lru_cache(maxsize=None)
def load_geometries(path_str: str) -> List[Dict]:
    path = Path(path_str)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as error:
        print(f"Warning: skipping invalid GeoJSON file {path}: {error}", file=sys.stderr)
        return []
    return list(iter_geometries(data))


def iter_geometries(obj: Dict) -> Iterator[Dict]:
    obj_type = obj.get("type")
    if obj_type == "FeatureCollection":
        for feature in obj.get("features", []):
            geometry = feature.get("geometry")
            if geometry:
                yield from iter_geometries(geometry)
    elif obj_type == "Feature":
        geometry = obj.get("geometry")
        if geometry:
            yield from iter_geometries(geometry)
    elif obj_type in {"Polygon", "MultiPolygon", "GeometryCollection", "Point", "MultiPoint"}:
        yield obj
    else:
        # Fallback for unexpected container structures
        geometry = obj.get("geometry") if isinstance(obj, dict) else None
        if isinstance(geometry, dict):
            yield from iter_geometries(geometry)


def point_on_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    if min(x1, x2) - EPSILON <= px <= max(x1, x2) + EPSILON and min(y1, y2) - EPSILON <= py <= max(y1, y2) + EPSILON:
        cross = abs((x2 - x1) * (py - y1) - (px - x1) * (y2 - y1))
        if cross <= EPSILON:
            return True
    return False


def point_in_linear_ring(lon: float, lat: float, ring: Sequence[Sequence[float]]) -> bool:
    coords = list(ring)
    if len(coords) < 3:
        return False
    if coords[0][:2] == coords[-1][:2]:
        coords = coords[:-1]
    inside = False
    count = len(coords)
    for idx in range(count):
        x1, y1 = coords[idx][0], coords[idx][1]
        x2, y2 = coords[(idx + 1) % count][0], coords[(idx + 1) % count][1]
        if point_on_segment(lon, lat, x1, y1, x2, y2):
            return True
        if ((y1 > lat) != (y2 > lat)):
            if y2 != y1:
                xinters = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            else:
                xinters = x1
            if abs(xinters - lon) <= EPSILON:
                return True
            if lon < xinters:
                inside = not inside
    return inside


def point_in_polygon(lon: float, lat: float, polygon: Sequence[Sequence[Sequence[float]]]) -> bool:
    if not polygon:
        return False
    if not point_in_linear_ring(lon, lat, polygon[0]):
        return False
    for ring in polygon[1:]:
        if point_in_linear_ring(lon, lat, ring):
            return False
    return True


def point_in_geometry(lon: float, lat: float, geometry: Dict) -> bool:
    gtype = geometry.get("type")
    if gtype == "Polygon":
        return point_in_polygon(lon, lat, geometry.get("coordinates", []))
    if gtype == "MultiPolygon":
        for polygon in geometry.get("coordinates", []):
            if point_in_polygon(lon, lat, polygon):
                return True
        return False
    if gtype == "GeometryCollection":
        for sub_geom in geometry.get("geometries", []):
            if point_in_geometry(lon, lat, sub_geom):
                return True
        return False
    if gtype == "Point":
        coords = geometry.get("coordinates", [])
        return len(coords) >= 2 and abs(coords[0] - lon) <= EPSILON and abs(coords[1] - lat) <= EPSILON
    if gtype == "MultiPoint":
        for coords in geometry.get("coordinates", []):
            if len(coords) >= 2 and abs(coords[0] - lon) <= EPSILON and abs(coords[1] - lat) <= EPSILON:
                return True
        return False
    return False


def parse_atlid_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def find_matches(atlid_rows: Iterable[Dict[str, str]], candidates: Sequence[Candidate], timeline: Sequence[datetime]) -> List[Dict[str, object]]:
    matches: List[Dict[str, object]] = []
    for row in atlid_rows:
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid latitude/longitude in row: {row}") from error
        timestamp_str = row.get("timestamp")
        if not timestamp_str:
            raise ValueError(f"Missing timestamp in row: {row}")
        atlid_dt = parse_atlid_timestamp(timestamp_str)
        window_start = atlid_dt - TIME_WINDOW
        window_end = atlid_dt + TIME_WINDOW

        left = bisect_left(timeline, window_start)
        right = bisect_right(timeline, window_end)
        for idx in range(left, right):
            candidate = candidates[idx]
            geometries = load_geometries(str(candidate.path))
            match_found = False
            for geometry in geometries:
                if point_in_geometry(lon, lat, geometry):
                    match_found = True
                    break
            if match_found:
                time_diff = candidate.dt - atlid_dt
                matches.append(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "atlid_timestamp": timestamp_str,
                        "slstr_folder": candidate.folder,
                        "geojson_filename": candidate.path.name,
                        "slstr_timestamp": candidate.dt.isoformat(),
                        "time_diff_seconds": time_diff.total_seconds(),
                    }
                )
    return matches


def load_atlid_rows(atlid_path: Path) -> List[Dict[str, str]]:
    with atlid_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_matches(output_path: Path, matches: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "latitude",
        "longitude",
        "atlid_timestamp",
        "slstr_folder",
        "geojson_filename",
        "slstr_timestamp",
        "time_diff_seconds",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in matches:
            writer.writerow(row)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    atlid_path = repo_root / "ATLID" / "atlid_positions.csv"
    slstr_folders = [
        repo_root / "SLSTR" / "polygon_slstr3a",
        repo_root / "SLSTR" / "polygon_slstr3b",
    ]
    output_path = script_dir / "atlid_slstr_matches.csv"

    candidates, timeline = build_candidates(slstr_folders)
    atlid_rows = load_atlid_rows(atlid_path)
    matches = find_matches(atlid_rows, candidates, timeline)
    write_matches(output_path, matches)
    print(f"Wrote {len(matches)} matches to {output_path}")


if __name__ == "__main__":
    main()
