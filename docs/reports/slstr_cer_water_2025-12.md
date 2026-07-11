# ORAC SLSTR water-cloud effective-radius validation against EarthCARE ACM-CAP — December 2025

Liquid-cloud effective-radius (CER) validation of **ORAC SLSTR** (Sentinel-3A)
against ACM-CAP `liquid_effective_radius`, December 2025. Uses the same daytime,
polar (Antarctic-summer) synergy collocation as the water-COT report
(`docs/reports/slstr_cot_water_2025-12.md`) — CER rode along in the same match
(`cer_orac`, `cer_water_atlid` already in the synergy CSVs), so it needed no new
collocation.

## 1. Sample and method

- **Reference**: ACM-CAP `liquid_effective_radius` (µm), pixel-aggregated over
  liquid-only ATLID profiles.
- **Coverage**: daytime (`illum_orac == 1`) → Antarctic summer, high SZA (~69°),
  N = 173 015 SLSTR pixels (pixel view), water-pure.
- **Metric**: CER is not heavy-tailed like COT, so **linear `r` is the meaningful
  correlation** here (not `r_log`).

## 2. Headline results (qc_strict, pixel)

CER is right-skewed, so the **median bias** is the headline and the mean is a
skew-sensitive diagnostic.

| stratum        |    N    | **median bias** | mean bias (skewed) | RMSE (µm) |   R    |
| -------------- | ------- | --------------- | ------------------ | --------- | ------ |
| **all**        | 173 015 | **+1.06 µm**    | +8.37              | 19.9      | −0.11  |
| ocean          |  70 648 | +0.78           | +6.33              | 17.7      | −0.02  |
| land           | 102 367 | +1.38           | +9.78              | 21.4      | −0.15  |

*95% CI on the median is tight and excludes zero* (all-stratum
**+1.06 [+1.03, +1.09] µm**, N = 173 015) — the small median bias is significant;
the near-zero R is the real story.

The headline:

> **On the median, ORAC SLSTR liquid effective radius is nearly unbiased (+1 µm)
> — but has essentially no skill (R ≈ −0.1).** The +8 µm often quoted is a *mean*
> skewed by a high-radius tail; the typical bias is small. The real problem is not
> bias but the **absence of correlation**: in Antarctic-summer daytime (bright
> ice, high SZA) ORAC's SWIR radius retrieval does not track the true particle
> size, it just scatters around roughly the right central value.

## 3. Interpretation

CER is the **weakest** of the SLSTR retrievals in this regime — worse even than
COT. The near-zero / negative correlation means ORAC's CER is not tracking the
real particle size at all here. The drivers are the familiar polar-daytime stack,
but CER is especially exposed:

- ORAC CER comes from the **SWIR (1.6 / 2.2 µm) reflectance**, which over bright
  snow/ice at **high solar zenith** has very low cloud-to-surface contrast — the
  information content for particle size is nearly gone.
- **Phase and partial-cloud** contamination (as for COT) add scatter with no
  compensating signal.

This is a **flagged regime**, not a general CER verdict: it says nothing about
ORAC CER at moderate sun over dark surfaces (which the SLSTR × EarthCARE geometry
cannot sample — see the CTH report §1). It does say that **CER should not be
trusted over bright polar surfaces in low-sun conditions**, which is directly
relevant to the `v5.1_new_snowice` assessment.

## 4. Figures

`figures/slstr_cer_water_2025-12/`: `cer_water_scatter.png` (a near-structureless
cloud — the visual signature of R ≈ 0), bias/R-by-stratum, QC sensitivity.

## 5. Where CER sits among the SLSTR variables

| variable | polar-daytime skill |
| -------- | ------------------- |
| CTH (thermal)        | **good** — bias −0.57 km, R 0.58 |
| water COT (solar)    | modest — +3 (−0.65 phase-agreed), r_log 0.11 |
| ice COT (solar)      | poor — +7, r_log 0.17 |
| **CER (solar/SWIR)** | **worst — +8 µm, R ≈ −0.1** |

The gradient is the same physical story: **thermal retrievals survive the polar
bright-surface regime; solar retrievals degrade, and the SWIR-based CER degrades
most.**

## 6. Reproducibility

```bash
# CER rides in the water-COT synergy collocation; only evaluate/figures differ:
python -m validation cer-water-evaluate \
    --matches 'validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv' \
    --out validation_data/slstr_cer_water_2025-12.csv
python -m validation cer-water-figures \
    --matches 'validation_data/slstr_synergy_2025-12_day/matches_synergy_*.csv' \
    --qc-mode qc_strict --label "SLSTR cer water Dec-2025 (polar, day)" \
    --out figures/slstr_cer_water_2025-12
```
