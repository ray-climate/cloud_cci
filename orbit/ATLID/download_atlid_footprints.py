#!/usr/bin/env python3
"""Download ATLID footprint metadata for a given date range.

The script queries the ESA SO-Cat catalogue for the EarthCARE L1 Instrument
Checked collection and downloads the corresponding ATLID metadata files. Files
are organised in per-day subdirectories based on the acquisition start time and
basic progress feedback is printed while the download is running.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import requests

SO_CAT_SEARCH_URL = "https://ec-pdgs-discovery.eo.esa.int/socat/EarthCAREL1InstChecked/search"
METADATA_BASE_URL = "https://ec-pdgs-dissemination1.eo.esa.int/oads/meta/EarthCAREL1InstChecked/metadata"
DEFAULT_CHUNK_DAYS = 1
MAX_RECORDS_LIMIT = 10000
DEFAULT_TIMEOUT = 120
DEFAULT_RETRIES = 3
RETRY_BACKOFF = 2.0


@dataclass(frozen=True)
class ProductEntry:
    dip_filename: str
    product_uri: str
    begin_acquisition: str

    @property
    def metadata_filename(self) -> str:
        return f"{self.dip_filename}_MD.XML"

    @property
    def metadata_url(self) -> str:
        return f"{METADATA_BASE_URL}/{self.metadata_filename}"

    @property
    def acquisition_date_str(self) -> str:
        value = (self.begin_acquisition or "").strip()
        if "T" in value:
            value = value.split("T", 1)[0]
        return value


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (inclusive) in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date (inclusive) in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("orbit/ATLID/ATLID_footprint"),
        help="Destination directory for downloaded metadata files.",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=DEFAULT_CHUNK_DAYS,
        help="Number of days per catalogue query (inclusive)."
             " Use smaller values if the server returns more than"
             f" {MAX_RECORDS_LIMIT} records per request.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query the catalogue but do not download any files.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds for catalogue and download requests.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Number of retries per metadata file before giving up.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information.",
    )
    return parser.parse_args(argv)


def to_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid date '{value}': {exc}") from exc


def iter_date_chunks(start: date, end: date, chunk_days: int) -> Iterable[Tuple[date, date]]:
    cursor = start
    delta = timedelta(days=chunk_days - 1)
    while cursor <= end:
        chunk_end = min(cursor + delta, end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def fetch_manifest(
    session: requests.Session,
    chunk_start: date,
    chunk_end: date,
    timeout: int,
    verbose: bool = False,
) -> Tuple[List[ProductEntry], int]:
    payload = {
        "service": "SimpleOnlineCatalogue",
        "version": "1.2",
        "request": "search",
        "format": "text/tab-separated-values",
        "pageCount": "50",
        "pageNumber": "1",
        "query.beginAcquisition.start": chunk_start.isoformat(),
        "query.beginAcquisition.stop": chunk_end.isoformat(),
        "query.productType": "ATL_NOM_1B",
    }
    response = session.post(SO_CAT_SEARCH_URL, data=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise SystemExit(
            f"Catalogue query failed for {chunk_start} to {chunk_end}: {exc}"
        ) from exc

    record_count = int(response.headers.get("X-ESA-SOCat-Record-Count", "0"))
    if record_count >= MAX_RECORDS_LIMIT:
        raise SystemExit(
            "Query returned >= {limit} records (actual: {count}) for {start} to {end}. "
            "Use a smaller chunk size via --chunk-days."
            .format(limit=MAX_RECORDS_LIMIT, count=record_count,
                    start=chunk_start, end=chunk_end)
        )

    text_stream = io.StringIO(response.text)
    reader = csv.DictReader(text_stream, delimiter="\t")
    rows: List[ProductEntry] = []
    for row in reader:
        if not row:
            continue
        dip_filename = row.get("dipFilename", "").strip()
        product_uri = row.get("productURI", "").strip()
        begin_acquisition = row.get("beginAcquisition", "").strip()
        if not dip_filename:
            if verbose:
                print(
                    f"Skipping catalogue row without dipFilename in {chunk_start} to {chunk_end}.",
                    file=sys.stderr,
                )
            continue
        rows.append(
            ProductEntry(
                dip_filename=dip_filename,
                product_uri=product_uri,
                begin_acquisition=begin_acquisition,
            )
        )
    return rows, record_count


def download_metadata(
    entry: ProductEntry,
    destination: Path,
    timeout: int,
    retries: int,
    verbose: bool = False,
) -> bool:
    if destination.exists():
        return False

    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    attempt = 0
    while True:
        try:
            with requests.get(entry.metadata_url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
            tmp_path.replace(destination)
            return True
        except requests.RequestException as exc:
            attempt += 1
            if attempt > retries:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                if verbose:
                    print(
                        f"ERROR: Failed to download {entry.metadata_url}: {exc}",
                        file=sys.stderr,
                    )
                return False
            sleep_time = RETRY_BACKOFF ** (attempt - 1)
            if verbose:
                print(
                    f"Retry {attempt}/{retries} for {entry.metadata_url} "
                    f"after {sleep_time:.1f}s due to {exc}",
                    file=sys.stderr,
                )
            time.sleep(sleep_time)


def format_chunk_label(chunk_start: date, chunk_end: date) -> str:
    if chunk_start == chunk_end:
        return chunk_start.isoformat()
    return f"{chunk_start.isoformat()}..{chunk_end.isoformat()}"


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    start = to_iso_date(args.start_date)
    end = to_iso_date(args.end_date)
    if start > end:
        raise SystemExit("--start-date must be on or before --end-date")
    if args.chunk_days < 1:
        raise SystemExit("--chunk-days must be >= 1")

    output_dir = args.output.expanduser().resolve()
    if args.verbose:
        print(f"Output directory: {output_dir}")

    session = requests.Session()

    total_expected = 0
    total_records = 0
    total_downloaded = 0
    total_skipped = 0
    chunks_processed = 0

    for chunk_start, chunk_end in iter_date_chunks(start, end, args.chunk_days):
        chunks_processed += 1
        chunk_label = format_chunk_label(chunk_start, chunk_end)
        if args.verbose:
            print(f"Querying {chunk_label} ...", flush=True)
        entries, record_count = fetch_manifest(
            session,
            chunk_start,
            chunk_end,
            timeout=args.timeout,
            verbose=args.verbose,
        )
        total_expected += record_count
        total_records += len(entries)

        if args.verbose:
            print(
                f"  Found {len(entries)} entries (catalogue reported {record_count}).",
                flush=True,
            )
        else:
            print(
                f"{chunk_label}: {len(entries)} files listed (catalogue reported {record_count}).",
                flush=True,
            )

        if args.dry_run:
            if args.verbose:
                if len(entries) == 0:
                    print("  No entries returned for this chunk.", flush=True)
                else:
                    print("  Dry-run: nothing downloaded for this chunk.", flush=True)
            else:
                print(f"{chunk_label}: dry-run, nothing downloaded.")
            continue

        if not entries:
            if args.verbose:
                print("  No entries returned for this chunk.", flush=True)
            else:
                print(f"{chunk_label}: no entries returned.")
            continue

        chunk_downloaded = 0
        chunk_skipped = 0
        total_in_chunk = len(entries)

        for index, entry in enumerate(entries, start=1):
            date_str = entry.acquisition_date_str or chunk_start.isoformat()
            dest_dir = output_dir / date_str
            dest_path = dest_dir / entry.metadata_filename
            downloaded = download_metadata(
                entry,
                dest_path,
                timeout=args.timeout,
                retries=args.retries,
                verbose=args.verbose,
            )
            if downloaded:
                chunk_downloaded += 1
            else:
                chunk_skipped += 1

            if args.verbose:
                status = "downloaded" if downloaded else "exists"
                print(
                    f"    ({index}/{total_in_chunk}) {status}: {dest_path}",
                    flush=True,
                )
            else:
                progress = (
                    f"{chunk_label}: downloading {index}/{total_in_chunk}"
                )
                print(progress, end="\r", flush=True)

        if not args.verbose:
            print(" " * 80, end="\r")
            print(
                f"{chunk_label}: downloaded {chunk_downloaded}, skipped {chunk_skipped} existing files.",
                flush=True,
            )
        else:
            print(
                f"  Downloaded {chunk_downloaded}, skipped {chunk_skipped} existing files.",
                flush=True,
            )

        total_downloaded += chunk_downloaded
        total_skipped += chunk_skipped

    if args.verbose:
        print(
            f"Completed {chunks_processed} catalogue queries."
            f" Entries processed: {total_records}."
            f" Expected according to headers: {total_expected}.",
            flush=True,
        )
    summary = (
        f"Finished: downloaded {total_downloaded} files, "
        f"skipped {total_skipped} already present."
    )
    print(summary)
    if args.dry_run:
        print("(dry run: no files downloaded)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
