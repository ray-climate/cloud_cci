# `orac` — ORAC SEVIRI L2 reader & summariser

This package reads the ESA Cloud CCI L2 ORAC output for MSG-3/SEVIRI
processed by STFC RAL and stored on JASMIN at
`/gws/ssde/j25a/cloud_ecv/data_out/seviri/`.

## Data layout

```
<root>/YYYY/MM/DD/HHMM/
    ESACCI-L2-CLOUD-CLD-SEVIRI_ORAC_MSG3_YYYYMMDDhhmm_R10.primary.nc
    ESACCI-L2-CLOUD-CLD-SEVIRI_ORAC_MSG3_YYYYMMDDhhmm_R10.secondary.nc
    ESACCI-L2-CLOUD-CLD-SEVIRI_ORAC_MSG3_YYYYMMDDhhmm_R11.primary.nc
    ESACCI-L2-CLOUD-CLD-SEVIRI_ORAC_MSG3_YYYYMMDDhhmm_R11.secondary.nc
```

* NetCDF-4, CF-1.4
* Grid: `along_track × across_track = 3712 × 3712` (full SEVIRI disk).
* Temporal: 15-min slots, 96 per day. Folder `HHMM` is the slot label,
  filename timestamp is the scan start (~12 min later). Filename time is
  authoritative.
* ~1.3 GB per slot; one month ≈ 3.5 TB.

## R10 vs R11

| | `Prior_File` | Semantics |
|---|---|---|
| **R10** | empty | "Clean" ORAC retrieval, climatological prior |
| **R11** | path to previous-slot R10 `liquid-water` variant | Sequential retrieval: FG, AP, AP uncertainty taken from R10 at scan − 15 min |

The **variable set is identical** in both. Differences to look for:
retrieved `cot`/`cer`/`ctp`, plus `niter`, `costja`, `degrees_of_freedom_signal`,
and the `*_ap`/`*_fg` diagnostics in the secondary file.

The R11 `Prior_File` points to a single-phase (`R10liquid-water`) intermediate
that lives outside `data_out/seviri/` (in `/home/users/drobbins/…`). It's a
processing intermediate; treat the R10 file in `data_out/seviri/` as the
canonical "clean" reference.

## Quick start

```python
from orac import discover_slots, open_slot, open_paired, monthly_summary

root = "/gws/ssde/j25a/cloud_ecv/data_out/seviri"
slots = discover_slots(root)
ds = open_slot(slots[0], "R10", variables=["cot", "cer", "cth", "qcflag"])

# R10/R11 side by side:
both = open_paired(slots[0], variables=["cot", "niter", "costja"])
both.sel(retrieval="R11")["cot"] - both.sel(retrieval="R10")["cot"]

# Whole-month summary:
df = monthly_summary(root, year=2026, month=2)
```

CLI:

```bash
python -m orac summarise --root /gws/ssde/j25a/cloud_ecv/data_out/seviri \
    --year 2026 --month 2 --out seviri_2026_02.csv
python -m orac dump-vars --root /gws/ssde/j25a/cloud_ecv/data_out/seviri \
    --time 2026-02-25T00:12 --retrieval R10
```

## Primary file — variables

Each retrieved quantity has a matching `*_uncertainty`.

| Group | Variables |
|---|---|
| Geolocation / time | `lat`, `lon`, `time` (Julian day, per pixel) |
| Viewing geometry | `solar_zenith_view_no1`, `satellite_zenith_view_no1`, `rel_azimuth_view_no1`, `sat_azimuth_view_no1` |
| Cloud retrieval | `cot`, `cer`, `ctp`/`ctp_corrected`, `cth`/`cth_corrected`, `ctt`/`ctt_corrected`, `cwp`, `cc_total` |
| Channel products | `cloud_albedo_in_channel_no_{1,2,4}`, `cee_in_channel_no_{4,9,10}` |
| Mask / phase / type | `cldmask`, `ann_phase`, `cphcot`, `cccot_pre`, `phase`, `phase_pavolonis`, `cldtype` |
| Quality | `niter`, `costja`, `costjm`, `qcflag`, `channels_used`, `variables_retrieved` |
| Surface / ancillary | `stemp`, `lsflag`, `lusflag`, `dem`, `illum` |

## Secondary file — variables

Diagnostics and forward-model inputs:

* pixel indices: `scanline_u`, `scanline_v`
* a priori + first guess for `cot`, `cer`, `ctp`, `stemp` — e.g. `cot_ap`, `cot_fg`
* measured albedo / reflectance / BT per channel
* first-guess reflectance / BT per channel
* residuals: `*_residual_in_channel_no_*`
* `measurement_uncertainty_in_channel_no_*`
* `degrees_of_freedom_signal`

## Pitfalls to remember

1. Folder `HHMM` ≠ filename time — trust the filename.
2. `stemp`, `stemp_ap`, `stemp_fg` all use `add_offset = 100` K. `mask_and_scale=True`
   in `xarray.open_dataset` handles this automatically — never bypass it.
3. `time` uses `days since -4712-01-01 12:00:00` (Julian Date). Use
   `orac.julian_to_datetime()` — do not let xarray auto-decode it.
4. Off-disk pixels have fill lat/lon; mask with `np.isfinite(lat) & np.isfinite(lon)`
   before any statistic.
5. `qcflag`, `cldtype`, `variables_retrieved`, `channels_used` are bitmask /
   categorical — use the helpers in `orac.flags`, never a naive comparison.
6. R10 has ~2.6% missing slots; R11 is nearly complete. Always check
   `SlotRecord.has("R10")` / `.has("R11")` before loading.
