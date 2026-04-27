"""One-shot: add `lsflag_orac` column to existing matches CSVs.

The first monthly run only sampled (cot, cldmask). This script samples
ORAC's land/sea flag at each matched (along_track, across_track) pixel
without re-running the kd-tree collocation.

Idempotent: skips rows that already have lsflag_orac.
"""
from __future__ import annotations

import argparse
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from orac.io import open_slot
from orac.metadata import discover_slots


def augment_one(csv: Path, seviri_root: Path, retrieval: str) -> str:
    df = pd.read_csv(csv)
    if "lsflag_orac" in df.columns and df["lsflag_orac"].notna().any():
        return "skip"

    df["lsflag_orac"] = np.nan
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
        ds = open_slot(slots[0], retrieval, variables=("lsflag",),
                       include_secondary=True)
        if "lsflag" not in ds.variables:
            continue
        arr = np.asarray(ds["lsflag"].squeeze(drop=True).values)
        at = group["sev_along_track"].astype(int).values
        ac = group["sev_across_track"].astype(int).values
        df.loc[group.index, "lsflag_orac"] = arr[at, ac]

    df.to_csv(csv, index=False)
    return "done"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--matches-dir", required=True)
    p.add_argument("--seviri-root",
                   default="/gws/ssde/j25a/cloud_ecv/data_out/seviri")
    p.add_argument("--retrieval", default="R11", choices=("R10", "R11"))
    args = p.parse_args()

    csvs = sorted(Path(args.matches_dir).glob("matches_cot_*.csv"))
    print(f"Augmenting {len(csvs)} CSVs in {args.matches_dir}")
    counts = {"done": 0, "skip": 0, "no-valid": 0}
    for i, csv in enumerate(csvs, 1):
        status = augment_one(csv, Path(args.seviri_root), args.retrieval)
        counts[status] = counts.get(status, 0) + 1
        if i % 100 == 0 or i == len(csvs):
            print(f"  [{i:4d}/{len(csvs)}] {counts}")
    print(f"Summary: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
