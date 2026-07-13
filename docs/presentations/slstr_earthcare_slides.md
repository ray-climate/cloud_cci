# Validating ORAC SLSTR cloud retrievals against EarthCARE

### Sentinel-3A × EarthCARE — December 2025

**Continuous:** CTH · water/ice COT · CER · CWP  **Categorical:** cloud mask · phase
**References:** EarthCARE ATLID (A-CTH, A-EBD, A-TC) + ACM-CAP synergy

> Slide-style summary for mobile viewing. The editable PowerPoint is
> `docs/presentations/SLSTR_EarthCARE_validation_ESA.pptx`; full detail is in
> [`slstr_validation_summary_2025-12.md`](../reports/slstr_validation_summary_2025-12.md).

---

## 1 · Motivation & objective

- **ORAC retrieves cloud properties (CTH, COT, CER, CWP, phase, mask) from the SLSTR dual-view imager on Sentinel-3A.**
- Passive cloud retrieval is hardest over the bright, cold cryosphere — this ORAC build (`v5.1_new_snowice`) specifically revises snow / sea-ice surface handling.
- EarthCARE supplies an independent **active** reference (ATLID lidar + CPR radar + MSI synergy) whose physics ORAC does not share — so disagreement reveals real retrieval error.
- **Objective: quantify bias, scatter and skill for every EarthCARE-validatable ORAC SLSTR variable over December 2025** — the one full processed month of this stream.

---

## 2 · Reference strategy — the most independent measurement wins

*ATLID active lidar first; MSI passive products never (Holz 2008; Cloud_cci PVIR v6).*

- **Active lidar / radar shares no physics with a passive imager** — a disagreement is a genuine ORAC error, not a shared passive bias.
- MSI-derived products are excluded on principle: agreement there could be two passive sensors making the same mistake.

| ORAC SLSTR variable | EarthCARE reference | Independence |
|---|---|---|
| Cloud-top height (`cth`) | **A-CTH** (ATLID) | pure lidar |
| Water COT / CER / CWP | **ACM-CAP** (ATLID+CPR+MSI) | radar+lidar synergy |
| Ice COT | **A-EBD** (ATLID ∫α dz) | pure lidar |
| Cloud phase & mask | **A-TC** (ATLID target class.) | pure lidar |

---

## 3 · Collocation — a polar comparison by orbital mechanics

*Two sun-synchronous orbiters coincide only near their track crossings.*

![Collocation density map, polar](../../figures/slstr_collocation/collocation_map_polar.png)

- **EarthCARE and Sentinel-3A are both sun-synchronous at different local overpass times** → simultaneous views only where the orbit planes converge, near the poles.
- **1.71 M matches, every one at |lat| 70.6–83.0°** (853k Arctic, 860k Antarctic); tropics and mid-latitudes return N = 0 — orbital geometry, not a filter.
- Complementary to the geostationary SEVIRI validation (±60°): SLSTR covers the poles against the same ATLID truth.

---

## 4 · Anatomy of a match — sub-pixel, well inside the thresholds

*One ATLID profile paired to the SLSTR pixel it falls in (frame 08642G, Antarctica).*

![Single crossing case study](../../figures/slstr_collocation/crossing_case_study.png)

![Collocation thresholds & match quality](../../figures/slstr_collocation/match_quality.png)

- **Temporal gate |Δt| ≤ 60 min (median 26 min); spatial nearest-pixel ≤ 3 km on-swath gate (median 0.43 km, max ~1.1 km).**
- SLSTR ≈ EarthCARE ≈ 1 km footprint (measured) → matches are sub-pixel for both; negligible footprint mismatch, unlike the 3–7 km SEVIRI pixels.
- The 3 km gate is a swath-membership test, not an averaging radius — it sits far above the 0.43 km match median.

---

## 5 · Both thresholds are non-binding — robustness

*CTH statistics stay flat across the temporal and spatial windows.*

![Delta-t sweep](../../figures/slstr_dt_sweep/slstr_dt_sweep.png)

![CTH threshold sensitivity](../../figures/slstr_sensitivity/slstr_cth_sensitivity.png)

- **Across Δt {15, 30, 45, 60} min × distance {1, 2, 3} km: CTH bias −0.55 to −0.57 km, RMSE 2.08–2.11 km, R 0.58–0.60 — essentially unchanged.**
- Polar clouds evolve slowly and matches are already < 1 km, so neither threshold tunes the result.
- Neither can be relaxed toward lower latitudes without breaking the comparison (hours in Δt, or hundreds of km in distance).

---

## 6 · Cloud-top height — the strong result

*ORAC `cth_corrected` vs A-CTH; thermal retrieval works day and night.*

![CTH scatter](../../figures/slstr_cth_2025-12/cth_scatter.png)

![CTH by cloud type](../../figures/slstr_cth_2025-12/cth_by_cloud_type.png)

- **Median bias −0.12 km** (mean −0.57, pulled by the multi-layer tail), RMSE 2.08 km, **R 0.58** (N = 162k) — the typical cloud top is essentially unbiased and correlated.
- Thick single-layer cloud is near-perfect (median −0.02 km, R 0.75); the error concentrates in high / multi-layer cloud (≈ −4 km) — the classic passive multi-layer ambiguity.
- Hemispheric contrast: **Arctic −0.90 km vs Antarctic −0.28 km** (December = Arctic night / Antarctic day).

---

## 7 · Water-cloud optical depth — the passive retrieval saturates

*ORAC vs ACM-CAP liquid τ (radar-aided synergy), phase-agree liquid.*

![Water COT saturation](../../figures/slstr_cot_water_2025-12/cot_water_saturation.png)

- **Report the median: −4.8.** The +3.1 mean is a heavy-tail skew artefact — it even flips sign.
- ORAC passive liquid τ is pinned ~7–8 across the entire ACM-CAP range (0.6 → 34): with no dynamic range it cannot correlate (r_log 0.11).
- It is **ORAC saturating, not the radar reference reading high** (adding CPR changes the bias by ~1 τ).

---

## 8 · The deficit is cryospheric — the surface, not the phase

*Solar-retrieval skill split by ORAC surface type.*

![Surface-type skill](../../figures/slstr_surface_2025-12/surface_type_bias.png)

- **Water COT correlates (r = +0.28) and over-reads only over open water;** over sea-ice & ice-sheet (95% of the scene) r → 0 and the bias is −4 to −5.
- Same clouds, only the background changed → the saturation is the **bright-surface radiative regime**, not an algorithm bug.
- CER stays surface-robust (median +0.2 / +0.3 µm over sea-ice / ice-sheet): trustworthy droplet size even where τ is unconstrained.

---

## 9 · Ice optical depth & effective radius — low bias, weak skill

*Ice COT vs A-EBD column τ · CER vs ACM-CAP liquid rₑ.*

![Ice COT scatter](../../figures/slstr_cot_ice_2025-12/cot_scatter.png)

![CER scatter](../../figures/slstr_cer_water_2025-12/cer_water_scatter.png)

- **Ice COT median +2.0** (mean +7.2 skewed; RMSE 16, r_log 0.17) — a modest typical overestimate; land ≈ ocean once the high-τ tail is removed.
- **CER median +1.1 µm** — nearly unbiased, but R ≈ −0.1: the SWIR radius scatters around the central value rather than tracking true particle size.
- Both are limited by scatter over bright, high-sun-zenith surfaces — not by a large systematic offset.

---

## 10 · Liquid water path — the saturation reaches the water budget

*ORAC `cwp` vs an independent ACM-CAP water-content reference (LWP).*

![CWP validation](../../figures/slstr_cwp_2025-12/cwp_validation.png)

- **Median bias −16 g m⁻² (−34%):** 30 vs 47 g m⁻², decorrelated (r_log 0.02), worse over ice sheet (−18) than ocean (−13).
- Validated against radar+lidar liquid-water-content — a reference that **never sees τ** — so it independently confirms the τ saturation propagates into the water budget.
- Fixed by the same surface-albedo advance, not a CWP-specific change.

---

## 11 · Cloud mask & phase vs A-TC — the two-way contingency

*Categorical validation against ATLID Target Classification (N = 614k pixels).*

![Phase & cloud-mask contingency](../../figures/slstr_phase_2025-12/phase_contingency.png)

- **Cloud mask: POD 0.69, FAR 0.11, accuracy 0.79** — conservative; the missed 31% is 88% thin cirrus the lidar sees and a passive imager cannot (the irreducible passive limit).
- **Phase: POD_liquid 89.5%, POD_ice 62.4%** → ORAC calls 38% of ice cloud tops "liquid" — a liquid bias.
- This bias is **surface-independent** → a second, intrinsic limitation, distinct from the surface-driven τ saturation.

---

## 12 · All variables at a glance — December 2025

*Median-primary headline per variable (N = pixel-level matches).*

| Variable | Reference | Headline metric | Verdict |
|---|---|---|---|
| Cloud-top height | A-CTH | median **−0.12 km**, R 0.58 | **Strong** |
| Water-cloud COT | ACM-CAP | median **−4.8** (passive τ saturates ~7–8) | Regime-limited |
| Ice-cloud COT | A-EBD | median **+2.0** (r_log 0.17) | Weak corr. |
| Effective radius | ACM-CAP | median **+1.1 µm** (R ≈ −0.1) | Robust bias, no skill |
| Liquid water path | ACM-CAP LWP | median **−16 g m⁻²** (−34%) | Inherits τ saturation |
| Cloud mask | A-TC | POD 0.69 · FAR 0.11 | Conservative |
| Cloud phase | A-TC | POD_liq 90% · POD_ice 62% | Liquid-biased |

---

## 13 · Summary — what is validated, and what is next

**VALIDATED** (all median-primary, with confidence intervals):
- CTH · water-COT · ice-COT · CER · CWP · cloud mask · phase, plus the collocation methodology and surface-type / phase stratification.

**TWO INDEPENDENT LIMITATIONS OF THE POLAR-DAYTIME SOLAR RETRIEVAL:**
1. bright-surface optical-depth **saturation** — surface-driven, propagates into CWP, spares open water;
2. an intrinsic **liquid phase bias** — POD_ice 62%, surface-independent.

**TRUSTWORTHY OVER THE CRYOSPHERE:**
- thermal CTH (median −0.12 km, R 0.58) and CER droplet size are robust; the solar τ / water-path products are regime-limited.

**NEXT STEPS** (require new processing, not further analysis):
- an Arctic boreal-summer month · Sentinel-3B · other seasons; and ORAC-SLSTR vs ORAC-SEVIRI against the same ATLID truth.
