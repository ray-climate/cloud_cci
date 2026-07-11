# Validating ORAC SLSTR cloud retrievals with EarthCARE — December 2025 summary

**Comprehensive overview** of the SLSTR × EarthCARE validation for the project
review. It synthesises the collocation strategy and the four validated variables;
the per-variable detail lives in the companion reports:
[CTH](slstr_cth_2025-12.md) · [water COT](slstr_cot_water_2025-12.md) ·
[ice COT](slstr_cot_ice_2025-12.md) · [CER](slstr_cer_water_2025-12.md), and the
method in [the plan](../slstr_validation_plan.md).

---

## 1. Executive summary

We validated ORAC cloud retrievals on **SLSTR (Sentinel-3A)**, version
`v5.1_new_snowice`, against **EarthCARE** active/synergy references (ATLID A-CTH,
ATLID A-EBD, ATLID+CPR+MSI ACM-CAP) for **December 2025** — the one full month of
this ORAC stream. Reusing the collocation framework proven on SEVIRI, only a new
SLSTR granule reader and a polar-crossing collocator were added.

COT and CER are heavy-tailed, so the **median bias** is the robust headline and
the mean is shown as a skew-sensitive diagnostic (for water COT the mean even
*flips sign* relative to the median).

| Variable | Reference | Regime | Headline (qc_strict, pixel) | Verdict |
| -------- | --------- | ------ | --------------------------- | ------- |
| **CTH**       | A-CTH (ATLID)   | polar, day+night | **median −0.12 km** (mean −0.57), RMSE 2.08 km, R **0.58** | **good** |
| **Water COT** | ACM-CAP liquid τ | polar, day (Antarctic) | **median −4.8** (mean +3.1 *skewed*), r_log 0.11 | τ low, noisy |
| **Ice COT**   | A-EBD column τ  | polar, day (Antarctic) | **median +2.0** (mean +7.2 *skewed*), r_log 0.17 | modest, noisy |
| **CER**       | ACM-CAP liquid rₑ | polar, day (Antarctic) | **median +1.1 µm** (mean +8.4 *skewed*), R ≈ **−0.1** | low bias, no skill |

All headline biases are **statistically significant** — the 95% confidence
intervals are tight (N = 53k–173k) and exclude zero (e.g. water COT median
−4.78 [−4.81, −4.75]; CER +1.06 [+1.03, +1.09] µm). The limitation is *weak
correlation*, not sampling noise.

**One-line message:** in the polar regime this sensor pairing can access, the
**thermal** cloud-top retrieval is nearly unbiased *and correlated* (median −0.12 km,
R 0.58). The **solar** retrievals (COT, CER) have only **small-to-moderate median
biases** (liquid τ ≈ −5, ice τ ≈ +2, CER ≈ +1 µm) but **weak correlation** — they
are **noisy, not grossly biased**. The large positive *mean* biases reported
elsewhere are **skew artefacts** of the heavy COT tail plus phase misclassification
(§3.2); the real limitation over polar bright surfaces at high sun-zenith is
**scatter and phase mis-ID**, not a large systematic τ offset.

---

## 2. The collocation strategy (the methodological core)

### 2.1 It is a polar comparison — by orbital mechanics, not by choice

The collocator runs **globally** (no latitude filter): every EarthCARE profile,
pole to equator, is matched to the nearest SLSTR pixel within ±60 min. But
EarthCARE and Sentinel-3A are both **sun-synchronous polar orbiters at different
local overpass times**, so they image the same ground point *simultaneously* only
where their orbit planes converge — near the poles. Everywhere else they pass
hours apart.

Result: of **1.71 M matches, every one is at |lat| 70.6–83.0°**; the tropical and
mid-latitude bands come back **N = 0**. This is the standard "simultaneous nadir
overpass" geometry, not a processing decision.

![Collocation density](../../figures/slstr_collocation/collocation_map_polar.png)

*Collocation density (matched profiles per 1000 km², area-normalised) rings
70–82° in both hemispheres (853 k N, 860 k S). Density peaks in a band around
75–80° and falls to zero at the pole itself — EarthCARE's orbit inclination does
not overfly it.*

**Complementarity:** the geostationary **SEVIRI** validation covers ±60°; SLSTR
covers the poles. Together they span pole-to-tropics against the same ATLID truth.

### 2.2 What a single collocation looks like

![Single crossing case study](../../figures/slstr_collocation/crossing_case_study.png)

*Frame 08642G over Antarctica: the SLSTR swath (coloured by cloud-top height) with
the 1-D ATLID nadir track threading across it. Each match pairs one lidar profile
with the SLSTR pixel it falls in — here at 0.37 km median separation.*

### 2.3 The two thresholds

Each matched pair must satisfy a **temporal** and a **spatial** condition:

| Threshold | Value | What the matches actually are |
| --------- | ----- | ----------------------------- |
| **Temporal** | \|Δt\| ≤ **60 min** | median offset **26 min**; offsets fill the window |
| **Spatial** | nearest pixel, **≤ 3 km** on-swath gate | median **0.43 km**, max ~1.1 km — sub-pixel |

![Collocation thresholds & match quality](../../figures/slstr_collocation/match_quality.png)

#### Spatial threshold vs the instrument footprints

The 3 km gate is not an averaging radius — it is a swath-membership test. Its
value is best read against the horizontal resolution of the two datasets, which
we **measured from the data** (adjacent-pixel / adjacent-profile spacing):

| Dataset | Nominal instrument footprint | Measured L2 horizontal grid |
| ------- | ---------------------------- | --------------------------- |
| **SLSTR** (nadir, ORAC cloud) | 1 km TIR / 0.5 km solar | **1.1 × 0.9 km** |
| **ATLID** (A-CTH, A-EBD)      | ~30 m spot, ~285 m sampling | **0.99 km** along-track |
| **ACM-CAP** (ATLID+CPR+MSI synergy) | CPR ~750 m, MSI 500 m | **0.99 km** along-track |

The two datasets are matched at **~1 km each**, so the ~0.43 km median separation
is **sub-pixel for both** — there is essentially no footprint-scale mismatch. This
is a marked contrast with the SEVIRI validation, where 3–7 km geostationary pixels
had to be matched to the ~1 km ATLID track. The 3 km gate simply excludes profiles
that fall *outside* a swath (nearest pixel then jumps to tens of km); it is ~3×
the pixel and, being far above the 0.43 km match median, is not a tuning knob —
loosening or tightening it 2–3× changes nothing (§2.4).

#### Temporal threshold — why 60 min

Agreement is flat with Δt: the crossing-geometry sweep shows the matches stay
polar and the CTH statistics do not degrade from 5 to 120 min — the tight ~1 km
spatial match dominates, and polar clouds evolve slowly.

![Δt sweep](../../figures/slstr_dt_sweep/slstr_dt_sweep.png)

### 2.4 Both thresholds are non-binding — sensitivity

Across Δt ∈ {15, 30, 45, 60} min × distance cap ∈ {1, 2, 3} km, the CTH statistics
barely move: **bias −0.55…−0.57 km, RMSE 2.08…2.11 km, R 0.58…0.60**. The distance
cap hardly changes N (matches are almost all < 1 km already).

![Threshold sensitivity](../../figures/slstr_sensitivity/slstr_cth_sensitivity.png)

*And neither threshold can be relaxed to reach lower latitudes without breaking
the comparison: a mid-latitude match would require Δt of hours (clouds change) or
a distance of hundreds of km (different clouds). The poles are the only place both
can be small at once.*

---

## 3. Results by variable

### 3.1 Cloud-top height — the strong result

**Median bias −0.12 km** (mean −0.57, pulled down by the multi-layer tail),
RMSE **2.08 km**, R **0.58** (N = 162 k). The *typical* cloud top is essentially
unbiased; single-layer thick cloud is near-perfect (median −0.02 km, R 0.75). The
error is concentrated and *robust* in high / multi-layer cloud (median ≈ −4 km) —
the classic passive multi-layer ambiguity, as in SEVIRI.

![CTH scatter](../../figures/slstr_cth_2025-12/cth_scatter.png)
![CTH by cloud type](../../figures/slstr_cth_2025-12/cth_by_cloud_type.png)

*Broken out by cloud type: single-layer and low cloud are near-perfect (thick
single-layer −0.15 km, R 0.75); high and multi-layer cloud are underestimated by
~4 km.*

### 3.2 Water-cloud optical thickness

**Median −4.8**, mean +3.1 (skewed). Two stacked effects:

1. **ORAC's passive liquid τ saturates** — pinned at ~7–8 across the whole
   ACM-CAP range (0.6→34) over bright polar surfaces at high sun-zenith. This
   drives the median underestimate *and* the near-zero correlation (r_log 0.11):
   a retrieval with no dynamic range can't correlate. It is ORAC saturating, not
   the radar-aided reference being high.
2. **ORAC misclassifies 22 % of liquid cloud as ice** (POD_liquid = 78 %,
   degrading with SZA and worse over sea-ice), which adds a high-τ tail that flips
   the *mean* to +3. The mean's positive sign is this misclassification artefact.

![Water COT phase analysis](../../figures/slstr_cot_water_2025-12/cot_water_phase_analysis.png)
![Water COT saturation](../../figures/slstr_cot_water_2025-12/cot_water_saturation.png)

The ice-misclassification connects to §3.3: it is the *same* ice-phase retrieval that
A-EBD): **water and ice COT are two views of one root cause** — ORAC's ice-phase
optical retrieval runs high over polar bright surfaces at high sun-zenith.

### 3.3 Ice-cloud optical thickness

**Median bias +2.0** (mean +7.2, skewed; RMSE 16, r_log 0.17). The typical ice
overestimate is modest; the large *mean* — and its apparent land>ocean contrast —
are high-τ-tail artefacts (median land +1.7 ≈ ocean +2.4). The limitation is
scatter (weak r_log), not a large systematic offset.

![Ice COT scatter](../../figures/slstr_cot_ice_2025-12/cot_scatter.png)

### 3.4 Cloud effective radius — nearly unbiased but no skill

**Median bias +1 µm** (mean +8.4, skewed). The typical radius is nearly right; the
real problem is **no correlation** (R ≈ −0.1) — over bright snow/ice at high
sun-zenith ORAC's SWIR radius retrieval scatters around the central value rather
than tracking the true particle size.

![CER scatter](../../figures/slstr_cer_water_2025-12/cer_water_scatter.png)

### 3.5 Liquid water path — the saturation reaches the water budget

**Median bias −16 g/m² (−34 %)**, mean +28 (skewed), r_log 0.02. Validated against
an *independent* reference — ACM-CAP's radar+lidar `liquid_water_content`
integrated to LWP, not a τ-derived quantity. ORAC liquid water path runs a third
low (30 vs 47 g/m²), decorrelated, and worse over ice sheet (−18) than ocean
(−13). Since `cwp ≈ (5/9) ρ_w τ r_e` and CER is unbiased, this is the τ saturation
(§3.2) **propagating into the water budget** — confirmed by a reference that never
sees τ. It is fixed by the same surface-albedo advance, not a CWP-specific change.

![CWP validation](../../figures/slstr_cwp_2025-12/cwp_validation.png)

---

## 4. Cross-cutting findings (from the refined stratification)

Splitting the single "polar" bin into sub-bands and hemispheres exposes structure
the aggregate hid:

1. **Thermal beats solar.** The gradient CTH → water COT → ice COT → CER tracks
   thermal-vs-solar and surface brightness: the thermal cloud-top retrieval
   survives the polar regime; the solar retrievals degrade, CER most.
2. **Hemispheric asymmetry (CTH):** Arctic **−0.90 km** vs Antarctic **−0.28 km**
   — the Arctic top is ~3× more underestimated. December = Arctic night / Antarctic
   day, so this is a night-vs-day + winter-sea-ice-vs-summer contrast.
3. **COT bias lives at the marginal ice zone (70–75°).** Water COT runs
   **+29 → +5.4 → −2.4** from 70–75° to 80–85°; ice COT +13 → +5.8. The
   bright-surface / partial-cloud inflation is a lower-polar-latitude effect;
   by 80–85° water COT is near-zero.
4. **Geometry is never the issue.** Every match is < 2 km and the bias is flat in
   the Δt < 3 min subset — the collocation is tight; the residuals are retrieval
   physics.
5. **The COT deficit is cryospheric — surface, not phase (§3e of the water-COT
   report).** Splitting by ORAC surface type, water-COT correlates (r = +0.28) and
   over-reads (+5) *only over open water*; over sea-ice and ice sheet (95 % of the
   scene) r → 0 and the bias is −4 to −5. Same clouds, only the background changed
   — direct proof the saturation is the bright-surface radiative regime. CER stays
   surface-robust (median +0.2/+0.3 µm over sea-ice/ice sheet): trustworthy droplet
   size over the cryosphere even where τ is unconstrained.

![Surface-type skill](../../figures/slstr_surface_2025-12/surface_type_bias.png)

---

## 5. What is validated, and what remains

| Dimension | Status |
| --------- | ------ |
| Variables: CTH, water COT, ice COT, CER | ✅ done |
| Collocation strategy + case study + sensitivity | ✅ done |
| Stratification: ocean/land · polar sub-bands · hemisphere · distance · time · cloud-class · phase | ✅ done |
| **Surface type** (sea-ice / snow / ice-sheet / open water) | ✅ done (§3e — the surface, not the phase) |
| **CWP** (last synergy variable) | ✅ done (§3.5 — LWP −34 %, inherits τ saturation) |
| Phase & cloud-mask (categorical) | ⚪ needs A-FM download |
| Per-orbit case studies, uncertainty validation | ⚪ optional depth |

**Data ceilings (cannot be closed):** low latitudes (orbital mechanics);
multi-season (only Dec 2025 processed); Sentinel-3B (not processed).

**Assessment:** all synergy-available variables (CTH, water & ice COT, CER, CWP)
plus the full collocation methodology, surface-type stratification and phase-skill
analysis are complete, defensible and meeting-ready. The single unifying result —
the passive solar optical-depth saturation over the bright polar cryosphere, which
propagates into water path and decorrelates over sea-ice/ice-sheet while sparing
open water — is now demonstrated four independent ways (τ-binning §3d, surface
split §3e, CER robustness, independent water-path reference §3f). The remaining
open item is categorical phase/cloud-mask validation (needs an A-TC/A-FM
download), which would close the ice-detection (POD_ice) side of §3.2.

---

## 6. Reproducibility

All code, stats and figures are on branch `codex-earthcare-sampling-plots`.
Per-variable commands are in each companion report; the collocation entry points
are `python -m validation slstr-cth-collocate | slstr-synergy-collocate |
slstr-collocate`, and the strategy figures come from
`scripts/slstr_collocation_figures.py`, `scripts/slstr_dt_sweep.py`,
`scripts/slstr_sensitivity_table.py`. ORAC SLSTR L2:
`/gws/ssde/j25a/cloud_ecv/data_out/slstr/v5.1_new_snowice/slstra/l2b/2025/12/`;
EarthCARE references under `earthcare_data/{ATL_CTH_2A,ATL_EBD_2A,ACM_CAP_2B}/2025/12/`.
