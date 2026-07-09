# ORAC SLSTR water-cloud optical-thickness validation against EarthCARE ACM-CAP — December 2025

Liquid-cloud optical thickness (COT) validation of **ORAC SLSTR** (Sentinel-3A)
against the ACM-CAP synergy `liquid_optical_depth` (`ACM_CAP_2B`, ATLID + CPR +
MSI), for **December 2025**. Completes the SLSTR trio with CTH
(`docs/reports/slstr_cth_2025-12.md`) and ice COT
(`docs/reports/slstr_cot_ice_2025-12.md`). Method inherited from the SEVIRI
water-COT study; collocation is the polar-crossing SLSTR × ATLID match (60-min,
~0.4 km).

## 1. Sample — Antarctic-summer daytime (as for ice COT)

The same two filters compound as in the ice-COT report: the collocation is
polar-only (~78–82°), and COT needs sunlight so night pixels are dropped
(`illum_orac == 1`, **14.9 % of matches kept**). In December this is the
**Antarctic in polar summer** — Southern hemisphere, solar zenith ~69°, over
sea-ice, ice shelf and open Southern Ocean. Read the numbers as the polar-summer
bright-surface behaviour, not a global water-COT verdict.

## 2. Reference and method

- **Reference**: ACM-CAP `liquid_optical_depth` — a per-profile scalar liquid τ
  from the variational ATLID+CPR+MSI retrieval (read directly, no integration).
- **Headline stratum** `water_pure_pixel`: SLSTR pixels where ≥ 80 % of the
  cloudy ATLID profiles are liquid-only — the apples-to-apples liquid comparison.
- **QC**: ACM-CAP `quality_status == 0` (`qc_strict`).
- **Compared quantity**: ORAC `cot` vs ACM-CAP liquid τ, log axes; **`r_log`** is
  the meaningful correlation.

Coverage: **N = 173 015 SLSTR pixels** (pixel view; sample-level identical to 3 sf).

## 3. Headline results (qc_strict, pixel, water-pure)

| stratum                    |    N    | bias (τ) | RMSE (τ) | r_log |
| -------------------------- | ------- | -------- | -------- | ----- |
| **all (= S-polar)**        | 173 015 | **+3.12**| 35.8     | 0.11  |
| ocean (sea-ice / S. Ocean) |  70 648 | +5.46    | 37.4     | 0.17  |
| land (ice shelf / coast)   | 102 367 | +1.50    | 34.6     | 0.07  |
| **phase_agree_liquid**     | 134 663 | **−0.65**| **20.3** | 0.15  |
| tdiff < 3 min              |   9 368 | +2.21    | 35.5     | 0.20  |

The headline:

> **In Antarctic-summer daytime, ORAC SLSTR overestimates ACM-CAP liquid τ by only
> ≈ +3 overall — and by essentially zero (−0.65, RMSE 20) once both instruments
> agree the column is liquid (`phase_agree_liquid`). Correlation is weak
> (r_log ≈ 0.11–0.15), the expected noise of polar-daytime COT.**

Key reads:

- **`phase_agree_liquid` is the trustworthy comparison** and it is nearly
  unbiased (−0.65) with RMSE roughly half the all-stratum value. The +3.1
  all-stratum bias is therefore largely a **phase-mismatch artefact** — pixels
  where ORAC retrieves ice/mixed while ACM-CAP reports liquid (and vice versa)
  contribute τ that is not a like-for-like comparison.
- **Better than both the SEVIRI polar water-COT (+18)** and the SLSTR ice COT
  (+7). Two plausible contributors: SLSTR's ~1 km nadir footprint suffers less
  sub-pixel partial-cloud inflation than SEVIRI's 3–7 km pixels, and the liquid
  scenes here are lower/optically simpler than the ice regime.
- **Ocean bias (+5.5) > land (+1.5)** — the reverse of the ice-COT case;
  low-sun sea-ice / Southern-Ocean liquid is where the residual overestimate sits.
- **Low r_log** (0.11–0.20) — polar-daytime COT is intrinsically noisy (phase
  ambiguity, high SZA, partial cloud), consistent with the SEVIRI water-COT
  experience that COT validates far more loosely than CTH.

## 4. Figures

`figures/slstr_cot_water_2025-12/`:

- `cot_water_scatter.png` — sample/pixel joint histograms, log axes. Principal
  density near ACM-CAP τ ≈ 10 / ORAC τ ≈ 7 (close to 1:1), with an upper-left
  spray of ORAC-high / ACM-CAP-low points that drives the positive tail — the
  partial-cloud / bright-surface inflation.
- `cot_water_bias_by_stratum_pixel.png`, `cot_water_r_by_stratum_pixel.png` —
  ocean-vs-land and the phase-agreement contrast.
- `cot_water_qc_sensitivity.png` — bias/RMSE across QC modes.

## 5. Conclusions

1. **Polar-summer daytime liquid COT is only mildly biased (+3), and unbiased
   where phase agrees (−0.65).** This is the best of the three SLSTR variables
   relative to its SEVIRI analogue.
2. **The all-stratum bias is dominated by phase mismatch**, not a systematic τ
   error; restrict to `phase_agree_liquid` for the physical comparison.
3. **COT correlation is weak** (r_log ~0.1–0.2) — the polar-daytime regime
   (phase ambiguity + high SZA + partial cloud) is intrinsically hard, as for
   SEVIRI.
4. Combined message across the three variables: **thermal CTH is excellent in the
   polar regime (−0.57 km); solar COT is regime-limited** — ice COT inflated
   (+7, bright ice sheet), water COT modest (+3, ~0 on matched phase).

## 6. Reproducibility

```bash
python -m validation slstr-synergy-collocate \
    --start 2025-12-01 --end 2026-01-01 --max-time-diff-min 60 \
    --out validation_data/slstr_synergy_2025-12
python scripts/slstr_filter_day.py \
    'validation_data/slstr_synergy_2025-12/matches_synergy_*.csv' \
    validation_data/slstr_synergy_2025-12_day
python -m validation cot-water-evaluate \
    --matches 'validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv' \
    --out validation_data/slstr_cot_water_2025-12.csv
python -m validation cot-water-figures \
    --matches 'validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv' \
    --qc-mode qc_strict --label "SLSTR cot water Dec-2025 (polar, day)" \
    --out figures/slstr_cot_water_2025-12
```

Inputs: ACM-CAP under `earthcare_data/ACM_CAP_2B/2025/12/`; ORAC SLSTR L2 under
`/gws/ssde/j25a/cloud_ecv/data_out/slstr/v5.1_new_snowice/slstra/l2b/2025/12/`.
Daytime restriction via `scripts/slstr_filter_day.py` (`illum_orac == 1`).
