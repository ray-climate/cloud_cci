"""Filter SLSTR matches CSVs to daytime pixels (illum_orac == 1).

ORAC's solar COT/CER retrieval cannot run at night, where it defaults to the
first-guess prior (constant cot=6.3). At December polar latitudes much of the
scene is in polar night, so the COT comparison MUST be restricted to daytime.
CTH is thermal and does not need this filter.

Usage: python scripts/slstr_filter_day.py '<src_glob>' <out_dir>
"""
import os
import sys
from glob import glob

import pandas as pd

src_glob, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
n_in = n_out = kept = 0
for p in sorted(glob(src_glob)):
    df = pd.read_csv(p)
    n_in += len(df)
    if "illum_orac" in df.columns:
        df = df[df["illum_orac"] == 1]
    if len(df):
        df.to_csv(os.path.join(out_dir, os.path.basename(p)), index=False)
        n_out += 1
        kept += len(df)
print(f"day-filter: {n_out} frames with daytime rows, {kept}/{n_in} rows kept "
      f"({100*kept/max(n_in,1):.1f}%) -> {out_dir}")
