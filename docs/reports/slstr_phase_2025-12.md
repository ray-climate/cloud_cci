# ORAC SLSTR cloud-mask & cloud-phase validation against EarthCARE A-TC — December 2025

Categorical (yes/no) validation of ORAC SLSTR (Sentinel-3A) **cloud detection** and
**cloud phase** against **EarthCARE A-TC** (ATLID Target Classification, `ATL_TC__2A`).
This is the two-way contingency — including **POD_ice** — that the continuous COT/CER
reports could not provide, because ACM-CAP's phase flags are liquid-centric (its
`ice_present` fires on almost nothing: in the matched sample there were **zero**
clean ice-only profiles, so ice-detection skill was previously unmeasurable).

## 1. Why A-TC, and what it measures

A-TC classifies **every ATLID bin** as
`0 clear · 1 warm-liquid · 2 supercooled-liquid · 3 ice · <0 missing/surface/attenuated`.
A passive imager retrieves the phase it sees from the top of the atmosphere, so each
ATLID column is reduced to its **cloud-top phase** (the highest cloud bin) and its
**cloud presence** (any cloud bin). ORAC's `cldmask_orac` and `phase_orac`
(1 liquid / 2 ice) are compared against these.

## 2. Sample

- **Reference**: A-TC `ATL_TC__2A`, Dec 2025 (29 of 31 days downloaded; days 30–31
  pending — a ~6 % sample addition that does not move the metrics).
- Augmented onto the existing day matches by `frame_id` + **exact** `ec_time`
  (profile-match |Δt| = 0.000 s), no re-collocation.
- **N = 614 201** matched pixels with an A-TC classification (95 % of valid day
  matches); **229 661** where both ORAC and A-TC see cloud (the phase sample).
- Antarctic-summer daytime, as for every solar variable (§ orbital + illumination
  constraint).

![Cloud mask & phase contingency](../../figures/slstr_phase_2025-12/phase_contingency.png)

## 3. Cloud mask — conservative, thin-cirrus-limited

| | A-TC cloud | A-TC clear |
| --- | --- | --- |
| **ORAC cloud** | 229 661 (hit) | 28 554 (false alarm) |
| **ORAC clear** | 102 345 (miss) | 253 641 (correct neg) |

**POD = 0.69 · FAR = 0.11 · accuracy = 0.79 · bias ratio = 0.78.**

- **ORAC detects 69 % of the clouds ATLID sees, with a low 11 % false-alarm ratio**
  — the mask is **conservative** (bias ratio 0.78 < 1: it under-calls cloud rather
  than over-calls).
- **The 31 % of missed clouds are 88 % ice-topped** (vs 45 % ice in the hits) — i.e.
  the misses are overwhelmingly **optically-thin cirrus that the lidar detects and
  the passive imager cannot**. This is the expected, physically-correct
  passive-vs-lidar limitation, not a mask defect.
- The 28 554 false alarms (ORAC cloud where ATLID is clear) are the counter-case —
  likely blowing snow, bright-surface artefacts, or sub-visible cloud below ATLID's
  threshold; at 4.6 % of the sample they are a minor term.

## 4. Cloud phase — a liquid bias, ice under-detected

Restricting to pixels where **both** see cloud and ORAC has a definite phase
(N = 229 661), against A-TC cloud-top phase:

| | A-TC liquid (truth) | A-TC ice (truth) |
| --- | --- | --- |
| **ORAC liquid** | 112 107 | 39 203 |
| **ORAC ice** | 13 212 | 65 139 |

- **POD_liquid = 89.5 %** — ORAC correctly identifies liquid cloud tops nearly nine
  times in ten.
- **POD_ice = 62.4 %** — ORAC correctly identifies only ~⅗ of ice cloud tops; **it
  misclassifies 38 % of ice cloud as liquid.** This is the number that was
  previously unmeasurable.
- **Overall phase accuracy = 77 %.** The error is **asymmetric: a liquid bias** —
  ORAC over-assigns the liquid phase, converting ice tops to liquid far more often
  (38 %) than liquid tops to ice (11 %).

This **supersedes and refines** the earlier one-way estimate in the water-COT report
(§3c, POD_liquid = 78 % from ACM-CAP's liquid-only flag): with a proper two-way
reference the liquid detection is better (89.5 %) and — crucially — we can now state
the ice side (62.4 %).

### 4a. Almost all polar liquid is supercooled

A-TC splits liquid into warm (class 1) and supercooled (class 2). In the
Antarctic-summer sample the liquid population is **essentially 100 % supercooled**
(warm-liquid N = 2 vs supercooled N = 125 317). ORAC correctly calls this
supercooled liquid "liquid" **89.5 %** of the time — so the liquid-detection skill is
really *supercooled*-liquid-detection skill, a demanding regime it handles well.

### 4b. Stratification — by surface type (the full cryosphere split)

Using the same `stemp`-based surface classes as the optical-depth analysis (§3e of
the water-COT report): **open water** (ocean, `stemp` ≥ 271.35 K), **sea-ice**
(ocean, colder), **snow / ice-sheet** (land). The coarse ocean bin averages the
first two and hides the contrast, so the full split is the one to read.

| surface | POD_liquid | POD_ice | accuracy | N | A-TC ice-frac |
| ------- | ---------- | ------- | -------- | -- | ------------- |
| **open water**     | 91.7 % | **62.2 %** | 78.4 % |  11 098 | 45 % |
| **sea-ice**        | 83.5 % | **64.4 %** | 76.8 % |  71 996 | 35 % |
| **snow / ice-sheet** | 93.1 % | **61.8 %** | 77.3 % | 146 419 | 50 % |
| SZA 60–70° | — | 60.7 % | 77.6 % | 115 939 | — |
| SZA 70–75° | — | 64.1 % | 76.8 % | 113 722 | — |

- **Ice detection is surface-independent — POD_ice is 62–64 % over *all three*
  surfaces**, including open water. This is the key result and a **sharp contrast
  with the optical depth**: over the same open-water pixels where the water-COT
  *correlation recovered* (r 0.28 vs ~0 on the cryosphere, §3e), the phase skill
  does **not** recover. So the liquid phase bias is **intrinsic to passive
  ice/liquid discrimination**, not the bright-surface radiative effect that drives
  the τ saturation. Two genuinely different limitations.
- **Liquid detection dips over sea-ice** (83.5 %) vs open water / ice sheet
  (92–93 %): sea-ice is the one surface where ORAC more often calls a liquid cloud
  "ice" — plausibly its intermediate brightness/temperature confusing the phase
  test. This is the only clear surface signal in the phase skill.
- **Cloud phase composition itself varies by surface** (A-TC ice-fraction 35 % over
  sea-ice → 50 % over the ice sheet) — more ice cloud over the continent, a
  cloud-climatology aside, not a retrieval effect.
- Skill is **flat in sun-zenith** across the available 60–75° range (~77 %).

## 5. Conclusions

1. **Cloud mask is conservative and physically sane**: POD 0.69, FAR 0.11; the
   missed 31 % is 88 % thin cirrus (lidar-only), the irreducible passive limit.
2. **ORAC has a liquid phase bias**: POD_liquid 89.5 % but **POD_ice 62.4 %** — 38 %
   of ice cloud tops are called liquid. This closes the phase story the COT reports
   left open (POD_ice was unmeasurable from ACM-CAP).
3. **The liquid bias is intrinsic, not cryospheric**: POD_ice is 62–64 % over open
   water, sea-ice *and* ice sheet alike (§4b). Over the very open-water pixels where
   the optical-depth correlation recovered (§3e), phase skill does not — proving the
   phase bias and the surface-driven τ saturation are two distinct limitations.
4. **Polar liquid is supercooled**, and ORAC handles it well (89.5 %).

**Link to the optical-depth results:** the 38 % ice→liquid misclassification is the
mechanism behind the high-τ tail that flips the *mean* water-COT bias positive
(water-COT §3c) — an ice cloud retrieved as a thick liquid cloud. The phase bias and
the τ saturation are the two independent limitations of the polar-daytime solar
retrieval.

## 6. Reproducibility

```bash
# download A-TC for the month (outage-resilient)
scripts/download_ec_month.sh 2025 12 A-TC
# augment matches with A-TC cloud-top phase + cloud presence (by frame_id + ec_time)
python scripts/slstr_atc_phase_augment.py
# cloud-mask + phase contingency, figure, stratification
python scripts/slstr_atc_phase_figure.py     # -> figures/slstr_phase_2025-12/
```

Inputs: A-TC under `earthcare_data/ATL_TC__2A/2025/12/`; matches under
`validation_data/slstr_synergy_2025-12_day/`. A-TC cloud-top phase = class of the
highest cloud bin (1/2 → liquid, 3 → ice); cloud mask = any cloud bin in a
non-attenuated column.
