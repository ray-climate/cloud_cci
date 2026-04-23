#!/usr/bin/env python3
"""Generate publication-quality full-disk figures of ORAC SEVIRI cloud products.

Default: pick a daytime slot over Africa/Europe (13:12 UTC) and write one PNG
per variable plus a 2×3 summary panel.

Usage::

    python scripts/make_full_disk_figures.py                       # 2026-02-25 13:12 UTC, R10
    python scripts/make_full_disk_figures.py --time 2026-02-15T12:12 --retrieval R11
    python scripts/make_full_disk_figures.py --out figures/2026-02-25
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from orac import discover_slots, open_slot
from orac.plots import (
    PRODUCT_SPECS,
    plot_cldtype,
    plot_full_disk,
    plot_phase,
    plot_product_suite,
    save_figure,
    set_publication_style,
)


DEFAULT_ROOT = "/gws/ssde/j25a/cloud_ecv/data_out/seviri"
DEFAULT_TIME = "2026-02-25T13:12"
DEFAULT_OUT  = Path("figures")


def pick_slot(root: str, when: datetime):
    slots = discover_slots(root,
                           start=when - timedelta(minutes=7),
                           end=when + timedelta(minutes=8))
    if not slots:
        raise SystemExit(f"No slot found near {when.isoformat()}")
    return min(slots, key=lambda s: abs(s.scan_time - when))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--time", default=DEFAULT_TIME,
                   help="ISO-8601 target scan time, UTC")
    p.add_argument("--retrieval", default="R10", choices=["R10", "R11"])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--no-pdf", action="store_true", help="Skip PDF output")
    args = p.parse_args()

    set_publication_style()

    when = datetime.fromisoformat(args.time).replace(tzinfo=timezone.utc)
    slot = pick_slot(args.root, when)
    print(f"→ Using slot {slot.scan_time.isoformat()}  ({slot.slot_dir})")

    wanted = ["lat", "lon", "cot", "cer", "ctp", "cth", "ctt", "cwp",
              "cc_total", "phase", "cldtype", "cldmask", "qcflag"]
    ds = open_slot(slot, args.retrieval, variables=wanted)
    print(f"→ Loaded {len(ds.data_vars)} variables from R={args.retrieval}")

    out_dir = args.out / f"{slot.scan_time:%Y%m%d_%H%M}_{args.retrieval}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Six-panel summary
    fig = plot_product_suite(ds, variables=("cot", "cer", "ctp", "cth", "ctt", "cwp"))
    written = save_figure(fig, out_dir / "panel_cloud_products",
                          also_pdf=not args.no_pdf)
    print("  panel_cloud_products  ->", *written)
    plt.close(fig)

    from orac.plots import _make_axes

    # 2. Individual continuous-product maps
    for var in ("cot", "cer", "ctp", "ctp_corrected", "cth", "cth_corrected",
                "ctt", "cwp", "cc_total"):
        if var not in ds.variables:
            continue
        fig = plt.figure(figsize=(7.2, 7.4))
        ax = _make_axes(fig)
        plot_full_disk(ds, var, ax=ax, cloud_only=(var != "cc_total"))
        written = save_figure(fig, out_dir / f"{var}",
                              also_pdf=not args.no_pdf)
        print(f"  {var:14s}  ->", *written)
        plt.close(fig)

    # 3. Phase (categorical)
    fig = plt.figure(figsize=(7.2, 7.4))
    ax = _make_axes(fig)
    plot_phase(ds, ax=ax)
    written = save_figure(fig, out_dir / "phase",
                          also_pdf=not args.no_pdf)
    print("  phase         ->", *written)
    plt.close(fig)

    # 4. Cloud type (categorical, legend outside)
    fig = plt.figure(figsize=(9.0, 7.4))
    ax = _make_axes(fig)
    plot_cldtype(ds, ax=ax)
    written = save_figure(fig, out_dir / "cldtype",
                          also_pdf=not args.no_pdf)
    print("  cldtype       ->", *written)
    plt.close(fig)

    print(f"\nAll figures written to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
