"""One-shot: backfill ORAC variables into existing matches CSVs.

Samples one or more ORAC variables at each matched
``(sev_along_track, sev_across_track)`` pixel without re-running the
kd-tree collocation. Each variable becomes a new column ``<var>_orac``
in the matches CSV.

Idempotent per variable: skips columns already populated.
"""
from __future__ import annotations

import argparse
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from orac.io import open_slot
from orac.metadata import discover_slots


def augment_one(csv: Path, seviri_root: Path, retrieval: str,
                variables: tuple[str, ...]) -> str:
    df = pd.read_csv(csv)
    new_cols = [f"{v}_orac" for v in variables]
    needed = [v for v, c in zip(variables, new_cols)
              if c not in df.columns or df[c].isna().all()]
    if not needed:
        return "skip"

    for v in needed:
        df[f"{v}_orac"] = np.nan

    df["sev_scan_time"] = pd.to_datetime(df["sev_scan_time"], errors="coerce")
    valid = df["valid_match"].astype(bool).values
    if not valid.any():
        df.to_csv(csv, index=False)
        return "no-valid"

    sub = df[valid].dropna(subset=["sev_scan_time"])
    for st, group in sub.groupby("sev_scan_time"):
        slot_dt = pd.Timestamp(st).to_pydatetime().replace(tzinfo=timezone.utc)
        slots = discover_slots(seviri_root,
                               slot_dt - timedelta(minutes=1),
                               slot_dt + timedelta(minutes=1),
                               retrievals=(retrieval,))
        if not slots:
            continue
        ds = open_slot(slots[0], retrieval, variables=tuple(needed),
                       include_secondary=True)
        at = group["sev_along_track"].astype(int).values
        ac = group["sev_across_track"].astype(int).values
        for v in needed:
            if v not in ds.variables:
                continue
            arr = np.asarray(ds[v].squeeze(drop=True).values)
            df.loc[group.index, f"{v}_orac"] = arr[at, ac]

    df.to_csv(csv, index=False)
    return "done"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--matches-dir", required=True)
    p.add_argument("--seviri-root",
                   default="/gws/ssde/j25a/cloud_ecv/data_out/seviri")
    p.add_argument("--retrieval", default="R11", choices=("R10", "R11"))
    p.add_argument("--variables", default="lsflag,phase",
                   help="comma-separated ORAC variable names to sample "
                        "(stored as <var>_orac)")
    args = p.parse_args()
    variables = tuple(v.strip() for v in args.variables.split(",") if v.strip())

    csvs = sorted(Path(args.matches_dir).glob("matches_cot_*.csv"))
    print(f"Augmenting {len(csvs)} CSVs in {args.matches_dir} with {variables}")
    counts = {"done": 0, "skip": 0, "no-valid": 0}
    for i, csv in enumerate(csvs, 1):
        status = augment_one(csv, Path(args.seviri_root), args.retrieval, variables)
        counts[status] = counts.get(status, 0) + 1
        if i % 100 == 0 or i == len(csvs):
            print(f"  [{i:4d}/{len(csvs)}] {counts}")
    print(f"Summary: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
