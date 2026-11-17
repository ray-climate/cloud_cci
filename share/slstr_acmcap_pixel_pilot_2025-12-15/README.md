# SLSTR × EarthCARE collocation — pilot (15 Dec 2025)

One day of ORAC SLSTR pixels matched to EarthCARE ACM-CAP (cloud properties)
and A-CTH (cloud-top height).

- `slstr_acmcap_pixels_2025-12-15.csv` — 56,375 rows × 91 columns (on JASMIN
  only: `/gws/pw/j07/nceo_aerosolfire/slstr_v5.1_new_snowice_val/`)
- `quicklook.png` — where the matches are, plus a first ORAC vs EarthCARE look

## Collocation strategy

- **One row = one SLSTR pixel** (Sentinel-3A, ORAC `v5.1_new_snowice`) that
  EarthCARE flew over. Each ACM-CAP profile (~1 km apart) goes to its nearest
  pixel centre, within 1.5 km.
- If two profiles fall in the same pixel (11% of rows) they are averaged;
  `n_profiles` says how many.
- A-CTH is on a slightly different along-track grid, so each ACM-CAP profile
  takes the A-CTH profile nearest in time (same frame, within 0.15 s).
- **Time window is ±60 min**, and the time difference is saved as `dt_s`
  (EarthCARE − SLSTR, seconds), so you can pick your own threshold.
  ±10 min leaves 15,382 pixels; ±30 min leaves 32,849.
- If two Sentinel-3A orbits see the same place inside the window, both are kept.
- **Nothing is filtered**: night, clear sky and poor-quality retrievals are all
  in there, with flags.
- All matches are polar (71–83° latitude), because that is where the two orbits
  cross close in time. Daytime pixels are Antarctic only in December.

## Output format

Empty cell = missing.

**Keys and match quality**

| Column | Meaning |
|---|---|
| `slstr_granule`, `slstr_along_track`, `slstr_across_track` | ORAC primary file and 0-based pixel indices — use these to join to another ORAC version |
| `slstr_time`, `slstr_lat`, `slstr_lon` | Pixel scan time and position |
| `ec_frame`, `ec_file`, `ec_index_nearest` | ACM-CAP frame, file and nearest profile index |
| `ec_time`, `ec_lat`, `ec_lon` | Nearest profile time and position |
| `n_profiles`, `distance_km`, `dt_s` | Profiles in the pixel, distance to pixel centre, time difference |

**ORAC — `orac_*`**

Copied straight from the primary file with the same names: `cot`, `cer`, `cwp`,
`cth`, `ctt`, `ctp` (plus `_corrected` and `_uncertainty`), `phase`,
`ann_phase`, `cldtype`, `cldmask`, `qcflag`, `costja`, `costjm`, `niter`,
`illum`, `lsflag`, `dem`, `stemp`, and the solar/satellite angles.
`orac_cot_3x3_*` and `orac_cth_3x3_*` are the mean/std over the surrounding
3×3 pixels.

**ACM-CAP — `acmcap_*`**

ACM-CAP is a profile, so each one is reduced to column values:

| Column | Meaning |
|---|---|
| `cot_liquid`, `cot_ice`, `cot_total` | Optical depth (ice = integrated ice extinction) |
| `cer_liquid_col`, `cer_ice_col` | Effective radius (µm), extinction-weighted over the column |
| `lwp`, `iwp`, `rwp` | Liquid, ice and rain water path (g m⁻²) |
| `phase_col` | 0 clear, 1 liquid only, 2 ice only, 3 both (a phase counts if its optical depth ≥ 0.01) |
| `quality_status`, `convergence_status`, `synergy_status`, `*_assim_status` | ACM-CAP flags as in the product (0 = good) |
| `cot_liquid_lnerr`, `iwp_lnerr`, `aod`, `cost_function`, `elevation_m`, `tropopause_m` | Straight from the product |

**A-CTH — `acth_*`**

Cloud-top height comes from the A-CTH product (ACM-CAP has none):

| Column | Meaning |
|---|---|
| `acth_cth_thick_km` | `ATLID_thick_cloud_top_height` — top of the uppermost optically thick layer (km) |
| `acth_cth_km` | `ATLID_cloud_top_height` — lidar-detected top, including thin cirrus (km) |
| `acth_tropopause_km` | `tropopause_height_wmo` (km) |
| `acth_quality_status` | −1 no cloud, 0 good, 1 low confidence, 2–3 warnings, 4 bad |
| `acth_confidence`, `acth_cloud_class` | As in the product (class: 1 thick, 2 thin, 3 thin over thick, 4 thick over thick, 5 thin over thin) |

## Worth knowing

- `acth_cth_thick_km` has spurious values far above the tropopause (up to
  39 km, 40% of its values in this file). Screen with
  `acth_cth_thick_km <= acth_tropopause_km + 2`.
- No parallax correction.

Made with `python -m validation slstr-pixel-collocate` (code in
`validation/collocate_slstr_pixel.py`).
