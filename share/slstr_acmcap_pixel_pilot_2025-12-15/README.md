# SLSTR × EarthCARE ACM-CAP collocation — pilot (15 Dec 2025)

One day of ORAC SLSTR pixels matched to EarthCARE ACM-CAP.

- `slstr_acmcap_pixels_2025-12-15.csv` — 56,375 rows × 93 columns (on JASMIN
  only: `/gws/pw/j07/nceo_aerosolfire/slstr_v5.1_new_snowice_val/`)
- `quicklook.png` — where the matches are, plus a first ORAC vs ACM-CAP look

## Collocation strategy

- **One row = one SLSTR pixel** (Sentinel-3A, ORAC `v5.1_new_snowice`) that
  EarthCARE flew over. Each ACM-CAP profile (~1 km apart) goes to its nearest
  pixel centre, within 1.5 km.
- If two profiles fall in the same pixel (11% of rows) they are averaged;
  `n_profiles` says how many.
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
| `cer_liquid_top`, `cer_ice_top` | Same, but only the top 1 unit of optical depth |
| `lwp`, `iwp`, `rwp` | Liquid, ice and rain water path (g m⁻²) |
| `ice_frac_top` | Ice share of the top 1 unit of optical depth (0 = liquid top, 1 = ice top) |
| `phase_col` | 0 clear, 1 liquid only, 2 ice only, 3 both |
| `cth`, `ctt` | Geometric cloud top (km) and temperature there (K) |
| `z_tau1`, `t_tau1` | Height and temperature where optical depth from the top reaches 1 |
| `quality_status`, `convergence_status`, `synergy_status`, `*_assim_status` | ACM-CAP flags as in the product (0 = good) |
| `cot_liquid_lnerr`, `iwp_lnerr`, `aod`, `cost_function`, `elevation_m`, `tropopause_m` | Straight from the product |

## Worth knowing

- ORAC `cth` sits ~2.3 km below ACM-CAP's geometric top but within 0.2 km of
  `z_tau1`, so `z_tau1` is probably the fairer comparison.
- For effective radius I wasn't sure whether column or cloud-top weighting is
  the better match to ORAC, so both are there.
- No parallax correction.

Made with `python -m validation slstr-pixel-collocate` (code in
`validation/collocate_slstr_pixel.py`).
