"""Thin CLI over the orac package.

Usage::

    python -m orac summarise   --root /gws/ssde/j25a/cloud_ecv/data_out/seviri --year 2026 --month 2 --out seviri_2026_02.csv
    python -m orac dump-vars   --root /gws/ssde/j25a/cloud_ecv/data_out/seviri --time 2026-02-25T00:12
    python -m orac missing     --root /gws/ssde/j25a/cloud_ecv/data_out/seviri --year 2026 --month 2 --out missing.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .io import open_slot
from .metadata import discover_slots
from .summary import missing_slot_report, monthly_summary


def _cmd_summarise(args: argparse.Namespace) -> int:
    df = monthly_summary(args.root, args.year, args.month,
                         retrievals=tuple(args.retrievals),
                         qc_rules=args.qc_rules)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    return 0


def _cmd_dump_vars(args: argparse.Namespace) -> int:
    ts = datetime.fromisoformat(args.time).replace(tzinfo=timezone.utc)
    slots = discover_slots(args.root, start=ts, end=ts.replace(minute=ts.minute + 1))
    if not slots:
        print(f"No slot found for {args.time}")
        return 1
    ds = open_slot(slots[0], args.retrieval)
    for name in sorted(ds.variables):
        da = ds[name]
        print(f"{name:50s} {str(da.dtype):10s} {da.dims}")
    return 0


def _cmd_missing(args: argparse.Namespace) -> int:
    df = missing_slot_report(args.root, args.year, args.month,
                             retrievals=tuple(args.retrievals))
    df.to_csv(args.out, index=False)
    n_missing = int((~df[["R10_primary", "R11_primary"]].all(axis=1)).sum())
    print(f"Wrote {len(df)} rows ({n_missing} slots missing at least one primary) to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m orac")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summarise", help="Walk a month and write a per-slot stats CSV")
    s.add_argument("--root", required=True, type=Path)
    s.add_argument("--year", required=True, type=int)
    s.add_argument("--month", required=True, type=int)
    s.add_argument("--retrievals", nargs="+", default=["R10", "R11"])
    s.add_argument("--qc-rules", default="default", choices=["default", "strict", "permissive"])
    s.add_argument("--out", required=True, type=Path)
    s.set_defaults(func=_cmd_summarise)

    d = sub.add_parser("dump-vars", help="List variables in one slot's primary+secondary")
    d.add_argument("--root", required=True, type=Path)
    d.add_argument("--time", required=True, help="ISO-8601 scan time, e.g. 2026-02-25T00:12")
    d.add_argument("--retrieval", default="R10", choices=["R10", "R11"])
    d.set_defaults(func=_cmd_dump_vars)

    m = sub.add_parser("missing", help="Report which 15-min slots are missing")
    m.add_argument("--root", required=True, type=Path)
    m.add_argument("--year", required=True, type=int)
    m.add_argument("--month", required=True, type=int)
    m.add_argument("--retrievals", nargs="+", default=["R10", "R11"])
    m.add_argument("--out", required=True, type=Path)
    m.set_defaults(func=_cmd_missing)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
