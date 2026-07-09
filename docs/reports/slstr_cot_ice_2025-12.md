# ORAC SLSTR ice-cloud optical-thickness validation against EarthCARE A-EBD — December 2025

Ice-phase cloud optical thickness (COT) validation of **ORAC SLSTR** (Sentinel-3A)
against the ATLID-only column extinction from `ATL_EBD_2A` (A-EBD), for **December
2025**. Companion to the SLSTR CTH report (`docs/reports/slstr_cth_2025-12.md`) and
the SEVIRI COT report (`docs/reports/cot_validation_2026-02.md`). Method inherited
from SEVIRI; the collocation is the polar-crossing SLSTR × ATLID match (60-min
window, ~0.4 km spatial).

## 1. The sample is Antarctic-summer, high-sun-zenith daytime only

Two filters shape what this comparison can see, and they compound:

1. **Polar-only collocation.** SLSTR × EarthCARE coincidences occur only at ~78–82°
   (see the CTH report §1).
2. **COT is daytime-only.** ORAC's optical-depth retrieval needs reflected
   sunlight; at night it defaults to a constant first-guess prior, so all night
   pixels are removed (`illum_orac == 1`). This drops **86 % of the matched
   profiles** (665 k of 4.6 M kept).

In December these two filters intersect at a single place: **the Antarctic in
polar summer**. The daytime ice sample is **100 % Southern hemisphere** (the
Arctic is in polar night — zero northern daytime matches) at **solar zenith angle
median 69° (10–90th pct 65–74°)** — i.e. a low sun, ~16–25° above the horizon,
over sea-ice, ice shelf and snow.

This is the **hardest regime that exists for a passive solar COT retrieval**:
bright cold surface, long atmospheric slant path, and frequent partial cloud. Read
the numbers below in that light — they are not a general ORAC-ice-COT verdict, they
are the polar-summer bright-surface stress test.

## 2. Reference and method

- **Reference**: A-EBD column-integrated 355 nm extinction, `τ355 = Σ ext·Δh` over
  good-quality bins (`validation/reference.py:cot_from_aebd`). A profile is
  flagged **attenuated** when the lidar is extinguished before the cloud base
  (τ a lower bound); the headline **excludes attenuated** profiles, which are kept
  as a separate diagnostic stratum.
- **Ice-only filter**: SLSTR pixels ORAC retrieves as ice (`phase_orac == 2`).
- **Compared quantity**: ORAC `cot` vs ATLID column τ355, on log axes (COT spans
  three decades, so **`r_log`** — Pearson of log10 τ — is the meaningful
  correlation; linear `r` is reported only for continuity and is ≈ 0).

Coverage after day-filter and ice-only: **N = 52 822 SLSTR pixels** (pixel view),
56 863 sample-level.

## 3. Headline results (ice-only, daytime, not-attenuated)

| stratum (pixel view)   |   N    | bias (τ) | RMSE (τ) | r_log |
| ---------------------- | ------ | -------- | -------- | ----- |
| **all (= S-polar)**    | 52 822 | **+7.19**| **16.4** | 0.17  |
| ocean (sea-ice)        | 20 892 | +5.29    | 12.4     | 0.37  |
| land (ice-sheet/snow)  | 31 930 | +8.43    | 18.5     | 0.05  |
| tdiff < 3 min          |  3 464 | +6.18    | 14.6     | 0.25  |
| τ passive (τ>0.3)      | 40 731 | +6.19    | 14.8     | 0.23  |
| attenuated (diagnostic)| 20 835 | +6.95    | 15.9     | 0.12  |

Bias/RMSE in linear τ; r_log in log10 space.

The headline:

> **In Antarctic-summer daytime, ORAC SLSTR overestimates ATLID ice-cloud column
> optical thickness by ≈ +7 (RMSE ≈ 16), with weak correlation (r_log ≈ 0.17).
> The overestimate is worse over the bright ice sheet (land +8.4) than over
> sea-ice (ocean +5.3).**

## 4. Interpretation — why so much worse than SEVIRI ice COT (+1.3)?

The SEVIRI ice-COT bias was +1.3 (mid-latitude / tropical, moderate sun). The
+7 here is the same retrieval pushed into its worst corner:

1. **Bright-surface + partial-cloud + 1-D forward model.** ORAC's plane-parallel
   forward model assumes a fully cloud-filled pixel. Over a bright ice/snow
   surface the clear-sky contribution to the TOA reflectance is large, and a
   partially-cloudy pixel is reproduced by retrieving a *thicker* cloud than is
   present. This is exactly the mechanism behind the SEVIRI **polar water-COT
   +18** bias — here it hits ice COT over Antarctica.
2. **High solar zenith (median 69°).** Low-sun geometry lengthens the slant path
   and amplifies forward-model and surface-reflectance error.
3. **Phase-classification noise.** Over cold bright scenes ORAC's ice/liquid
   split is less reliable; ice-only selection then mixes in mis-classified
   pixels, degrading the correlation (r_log 0.05 over the ice sheet).

The **ocean (sea-ice) subset is the most trustworthy** (bias +5.3, r_log 0.37 —
comparable to SEVIRI's ice correlation); the ice-sheet subset is where both the
bias and the noise blow up.

## 5. Figures

`figures/slstr_cot_ice_2025-12/`:

- `cot_scatter.png` — sample/pixel joint histograms, log axes. A broad density
  cloud sitting above the 1:1 line (the +7 offset), with the principal blob near
  ATLID τ ≈ 2–4 / ORAC τ ≈ 4–8. No clean ridge — the low r_log is visible as the
  width of the cloud.
- `cot_diagnostic.png` — coloured by latitude (all S-polar), distance (< 2 km),
  Δt, and attenuation.
- `cot_bias_by_stratum_pixel.png`, `cot_r_by_stratum_pixel.png` — the land-vs-ocean
  contrast.

## 6. Conclusions

1. **This is a polar-summer, high-SZA, bright-surface stress test**, not a general
   ice-COT verdict — daytime + polar-crossing in December can only sample the
   Antarctic.
2. **ORAC SLSTR overestimates ice COT by ≈ +7 here**, worst over the ice sheet
   (+8.4) — the bright-surface / partial-cloud / high-SZA mechanism, the same one
   that drives the SEVIRI polar water-COT bias.
3. **Sea-ice (ocean) is the cleaner subset** (r_log 0.37). The ice-sheet subset is
   dominated by surface-coupling and phase noise.
4. **Contrast with CTH is the key message**: the *thermal* cloud-top retrieval is
   nearly unbiased in this same polar regime (−0.57 km), while the *solar* optical
   retrieval is badly inflated. Thermal vs solar, not the collocation, is the
   divide.
5. Treat polar-daytime ice COT as a **flagged regime** for the `v5.1_new_snowice`
   assessment: the new surface handling has not removed the bright-surface COT
   inflation.

## 7. Reproducibility

```bash
python -m validation slstr-collocate \
    --start 2025-12-01 --end 2026-01-01 --max-time-diff-min 60 \
    --out validation_data/slstr_cot_2025-12
python scripts/slstr_filter_day.py \
    'validation_data/slstr_cot_2025-12/matches_cot_*.csv' \
    validation_data/slstr_cot_2025-12_day
python -m validation evaluate \
    --matches 'validation_data/slstr_cot_2025-12_day/matches_cot_*.csv' \
    --out validation_data/slstr_cot_ice_2025-12.csv
python -m validation figures \
    --matches 'validation_data/slstr_cot_2025-12_day/matches_cot_*.csv' \
    --label "SLSTR cot Dec-2025 (ice, polar, day)" --out figures/slstr_cot_ice_2025-12
```

Inputs: A-EBD under `earthcare_data/ATL_EBD_2A/2025/12/`; ORAC SLSTR L2 under
`/gws/ssde/j25a/cloud_ecv/data_out/slstr/v5.1_new_snowice/slstra/l2b/2025/12/`.
`scripts/slstr_filter_day.py` enforces the daytime (`illum_orac == 1`) restriction
that the solar COT retrieval requires.
