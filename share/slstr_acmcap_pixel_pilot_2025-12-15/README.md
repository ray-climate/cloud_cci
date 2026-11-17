# SLSTR-pixel × EarthCARE ACM-CAP collocation — pilot, 15 December 2025

A one-day pilot for review before the method is run on the whole month.
Contact: Rui Song.

The CSV is not in the git repository (it holds unpublished retrieval values); it
is on JASMIN at `/gws/pw/j07/nceo_aerosolfire/slstr_v5.1_new_snowice_val/`.

| File | Content |
|---|---|
| `slstr_acmcap_pixels_2025-12-15.csv` | 56,375 rows × 93 columns, 34 MB |
| `quicklook.png` | Where/when the matches are, and first ORAC-vs-ACM-CAP comparisons |
| `README.md` | This file |

## What one row is

**One ORAC SLSTR pixel, overflown by one EarthCARE pass, within ±60 min.**

- SLSTR: Sentinel-3A, ORAC `v5.1_new_snowice`, `l2b/2025/12`, primary files.
- EarthCARE: ACM-CAP (ATLID + CPR + MSI synergy), baseline `EXBC`, processor 11.50.
- The pixel is the unit. All ACM-CAP profiles (~1 km apart) whose nearest pixel
  centre is this pixel are combined into the row; `n_profiles` says how many
  (1 for 89% of rows, 2 for 11%).
- The row key is `slstr_granule` + `slstr_along_track` + `slstr_across_track`
  + `ec_frame`. A pixel appears twice only if two successive EarthCARE orbits
  both cross it inside the window (9 pixels in this file).
- **Nothing is filtered.** Night, twilight, clear sky and poor-quality
  retrievals are all present. Choose your own cuts from the columns below.
- **Time window:** every pair with |`dt_s`| ≤ 3600 s is kept. If Sentinel-3A
  sees the same EarthCARE profile on two orbits inside the window, both pixels
  are rows (this happens for 11% of profiles), so a tighter cut on `dt_s` is exact.

To join to another ORAC version on the same L1 grid, use the granule time and
orbit in `slstr_granule` with `slstr_along_track` / `slstr_across_track`
(0-based indices into the 1200 × 1500 arrays).

## Coverage of this pilot

| Cut | Pixels | of which ORAC daytime (`orac_illum == 1`) |
|---|---|---|
| ±60 min (all rows) | 56,375 | 22,655 |
| ±30 min | 32,849 | 14,406 |
| ±10 min | 15,382 | 4,888 |
| ±5 min | 7,098 | 3,418 |

- 31 of the day's 122 ACM-CAP frames have matches, against 58 SLSTR granules.
- All matches are polar, |lat| 70.8–83.0°: 27,416 Arctic, 28,959 Antarctic.
  The two satellites only pass the same place close in time near the poles.
- In December the Arctic is in polar night, so all daytime pixels are Antarctic.

## Columns

Empty cell = missing / fill value.

### Match and keys

| Column | Meaning |
|---|---|
| `ec_frame`, `ec_file` | EarthCARE orbit+frame ID and ACM-CAP filename |
| `ec_index_nearest` | 0-based profile index (in the ACM-CAP file) of the profile nearest the pixel centre |
| `ec_index_first`, `ec_index_last` | Range of profile indices combined in this row |
| `n_profiles` | Number of ACM-CAP profiles in the pixel |
| `ec_time`, `ec_lat`, `ec_lon` | Time (UTC) and position of the nearest profile |
| `slstr_granule` | ORAC primary filename |
| `slstr_orbit` | Sentinel-3A absolute orbit number |
| `slstr_along_track`, `slstr_across_track` | 0-based pixel indices in the granule |
| `slstr_time`, `slstr_lat`, `slstr_lon` | Pixel scan time (UTC) and centre position |
| `distance_km` | Great-circle distance, nearest profile to pixel centre (≤ 1.5 km; median 0.42) |
| `dt_s` | `ec_time` − `slstr_time` in seconds; positive = EarthCARE later |

### ORAC (`orac_*`)

Values copied unchanged from the primary file at the pixel (scale/offset
applied), named `orac_<variable>`:

`cot`, `cer`, `cwp`, `cth`, `cth_corrected`, `ctt`, `ctt_corrected`, `ctp`,
`ctp_corrected` and their `_uncertainty`; `phase`, `phase_pavolonis`,
`ann_phase` (+ uncertainty), `cldtype`, `cldmask` (+ uncertainty), `cccot_pre`,
`cc_total`; `qcflag`, `costja`, `costjm`, `niter`; `illum`, `lsflag`,
`lusflag`, `dem`, `stemp`; `solar_zenith_view_no1`,
`satellite_zenith_view_no1`, `rel_azimuth_view_no1`.

Added: `orac_cot_3x3_mean`, `_std`, `_n` and `orac_cth_3x3_mean`, `_std`, `_n`
— mean, standard deviation and valid count over the 3 × 3 pixels centred on
the match (scene homogeneity).

### ACM-CAP (`acmcap_*`)

ACM-CAP is a profile product (242 height bins). Each profile is reduced to
column values as below, then averaged over the `n_profiles` profiles in the
pixel. "Ice" is everything ACM-CAP calls ice (cloud ice and snow).

| Column | Unit | Definition |
|---|---|---|
| `acmcap_cot_liquid` | – | ACM-CAP `liquid_optical_depth` (empty where the product has none) |
| `acmcap_cot_ice` | – | ∫ `ice_extinction` dz |
| `acmcap_cot_total` | – | liquid + ice (missing liquid counted as 0) |
| `acmcap_cer_liquid_col`, `acmcap_cer_ice_col` | µm | Extinction-weighted effective radius over the whole column |
| `acmcap_cer_liquid_top`, `acmcap_cer_ice_top` | µm | Same, weighted over the uppermost 1 unit of total optical depth |
| `acmcap_lwp` | g m⁻² | ∫ `liquid_water_content` dz |
| `acmcap_iwp` | g m⁻² | ACM-CAP `ice_water_path` |
| `acmcap_rwp` | g m⁻² | ∫ `rain_water_content` dz |
| `acmcap_ice_frac_top` | 0–1 | Ice share of the uppermost 1 unit of optical depth: 0 = liquid top, 1 = ice top |
| `acmcap_phase_col` | flag | 0 clear, 1 liquid only, 2 ice only, 3 both; a phase counts if its optical depth ≥ 0.01. Nearest profile. |
| `acmcap_cth` | km | Height of the highest bin with liquid + ice extinction ≥ 1e-5 m⁻¹ (geometric top, as seen by lidar/radar) |
| `acmcap_ctt` | K | Model (X-MET) temperature at `acmcap_cth` |
| `acmcap_z_tau1`, `acmcap_t_tau1` | km, K | Height and temperature where optical depth from the top reaches 1; empty if total COT < 1 |
| `acmcap_cot_total_range`, `acmcap_cth_range` | –, km | max − min over the profiles in the pixel (0 when `n_profiles` = 1) |
| `acmcap_cot_liquid_lnerr`, `acmcap_iwp_lnerr` | – | ACM-CAP one-sigma error in the natural log of liquid optical depth / ice water path |
| `acmcap_aod` | – | ACM-CAP `aerosol_optical_depth` |
| `acmcap_cost_function` | – | ACM-CAP `cost_function` |
| `acmcap_elevation_m`, `acmcap_tropopause_m` | m | ACM-CAP `elevation`, `tropopause_height` |
| `acmcap_quality_good_frac` | 0–1 | Fraction of the pixel's profiles with `quality_status` = 0 |

Status flags, taken from the nearest profile (not averaged):

| Column | Values |
|---|---|
| `acmcap_quality_status` | 0 good, 1 unconverged, 2–7 optional radar/passive measurements failed (odd = also unconverged), higher = worse; see the product's `definition` attribute |
| `acmcap_convergence_status` | 0 converged, 1 empty state, 2 max iterations, 3 max sub-iterations, 4 cost out of range, 5 failed, 6 error |
| `acmcap_synergy_status` | 0 all nominal, 1 optional active failed, 2 optional passive failed, 3 all optional failed, 4 mandatory failed |
| `acmcap_atlid_assim_status`, `acmcap_cpr_assim_status`, `acmcap_msi_sw_assim_status`, `acmcap_msi_lw_assim_status` | 0 successful, 1 not attempted, 2 failed, 3 disabled |

## Things to know before using it

- **Cloud-top height.** `acmcap_cth` is the geometric top; a passive retrieval
  sits lower in thin-topped cloud. In this file (±10 min, quality 0) the median
  ORAC `cth` − `acmcap_cth` is −2.3 km, but ORAC `cth` − `acmcap_z_tau1` is +0.2 km.
- **Height datum.** The ACM-CAP file does not state the reference of `height`.
  `orac_dem` and `acmcap_elevation_m` agree to 2 m in the median, so the two
  appear to share a datum. Please confirm what ORAC `cth` is measured from.
- **Effective radius.** ACM-CAP gives a profile, ORAC a single value. Both a
  column-weighted and a top-weighted version are provided; which is the fairer
  comparison is open.
- **Surface.** All daytime pixels are over Antarctic snow/ice at solar zenith
  63–75° in this file. Check `orac_qcflag` (bit 4 = snow/ice surface).
- **Parallax.** Not corrected. `orac_satellite_zenith_view_no1` is provided so
  high cloud at large view angles can be screened.
- **Clear sky.** ACM-CAP rows with `acmcap_cot_total` = 0 are kept, so cloud
  detection can be assessed too.

## How it was made

Code: `validation/collocate_slstr_pixel.py`, `validation/reference.py`
(`accap_column_properties`), `validation/readers.py` (`read_accap_columns`) in
the `cloud_cci` repository.

    python -m validation slstr-pixel-collocate --start 2025-12-15 --end 2025-12-16 \
        --out validation_data/slstr_pixel_acmcap_pilot_2025-12-15 \
        --merged validation_data/slstr_pixel_acmcap_pilot_2025-12-15.csv
    python scripts/slstr_pixel_quicklook.py <csv> quicklook.png 10

Checks made on the pilot: no duplicated row keys; pixels, ORAC COT, ACM-CAP
liquid optical depth and time differences agree with the earlier
profile-based collocation of the same frames; integrating ACM-CAP liquid
extinction and ice water content reproduces the product's own
`liquid_optical_depth` and `ice_water_path` to better than 0.001%.

## Feedback wanted

1. Is one row per pixel, with these keys, usable in your evaluation workflow?
2. Are the ACM-CAP column definitions (top optical depth of 1, extinction
   threshold for cloud top) the right ones to compare with ORAC?
3. Any ORAC or ACM-CAP variables missing, or not needed?
4. CSV is fine for one day; for the full month (~1.5 million rows, ~1 GB)
   would NetCDF or Parquet be preferable?
