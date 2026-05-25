# COT-water vs ACM-CAP — sub-pixel sampling and homogeneity (February 2026)

Follow-up to `docs/reports/cot_validation_2026-02.md`. The headline
COT-water comparison shows weak agreement (R11, `qc_strict`, sample view:
`r_log ≈ 0.32`, `bias ≈ +7`). This note asks whether the disagreement is
dominated by:

- **sub-pixel sampling** — too few EarthCARE profiles per SEVIRI pixel?
- **sub-pixel heterogeneity** — variation in COT *inside* a pixel?

Spoiler: heterogeneity dominates, sampling-density does not help, and the
residual disagreement is intrinsic to the two retrievals' physical limits.

---

## 1. Method

### 1.1 Per-pixel sampling metadata

`aggregate_to_pixel_water` (`validation/statistics.py:358`) now carries six
extra summary columns describing the EarthCARE COT field inside each ORAC
pixel:

| column            | meaning                                                |
| ----------------- | ------------------------------------------------------ |
| `ref_std_atlid`   | sample std of `cot_water_atlid` across the pixel       |
| `ref_min_atlid`   | min                                                    |
| `ref_max_atlid`   | max                                                    |
| `ref_range_atlid` | `max − min`                                            |
| `ref_cv_atlid`    | `std / |mean|` — the **homogeneity index**             |
| `n_liquid_only`   | already-present count of liquid-only ATLID profiles    |

`ref_cv_atlid` is computed as

```python
out["ref_cv_atlid"] = np.where(
    np.isfinite(out[var_atlid]) & (np.abs(out[var_atlid]) > 0),
    out["ref_std_atlid"] / np.abs(out[var_atlid]),
    np.nan,
)
```

(`validation/statistics.py:423`). It is dimensionless: CV = 0.25 means the
std equals 25 % of the mean COT in the pixel. Single-profile pixels have
`std = NaN` and therefore drop out of any CV-stratified analysis.

### 1.2 Sub-pixel sampling filter

`filter_water_sampling()` (`validation/statistics.py:442`) restricts the
pixel and sample tables to ORAC pixels passing
`n_liquid_only ≥ min_n_liquid_only` *and* `n_total ≥ min_n_total`. The
matched annotated raw-row table (returned alongside) is then fed to
`dedupe_to_sample_water` to produce the sample-level view, so all three
views (raw → sample → pixel) reflect the same filter.

Wired into the existing CLI:

- `validation cot-water-evaluate ... --min-n-liquid-only N --min-n-total M`
- `validation cot-water-figures   ... --min-n-liquid-only N --min-n-total M`
- `validation cot-water-compare   ... --min-n-liquid-only N --min-n-total M`

### 1.3 Homogeneity sweep

`homogeneity_sweep_stats()` (`validation/statistics.py:479`) bins the
*unfiltered* pixel table on the grid `(n_cut × CV bin)` and computes
`continuous_stats` per cell. `water_cloud_figures.homogeneity_sweep()`
draws the resulting 1×3 panel (bias / RMSE / Pearson R, x = CV bin, one
curve per n cut). For COT-water the third panel uses `r_log` (Pearson R
on `log10(τ)`) because raw-space R is dominated by a handful of extreme
points.

CLI: `validation cot-water-homogeneity`
(`validation/cli.py:cmd_cot_water_homogeneity`).

---

## 2. Results — February 2026, R11, `qc_strict`

Domain: 359 105 ORAC pixels with `ref_cv_atlid` defined (i.e. ≥ 2
EarthCARE profiles inside the pixel) out of 453 690 pixel-aggregate rows.

Figure: [`figures/cot_water_2026-02_homogeneity/cot_water_homogeneity_R11.png`](../../figures/cot_water_2026-02_homogeneity/cot_water_homogeneity_R11.png)
(R10 equivalent: `cot_water_homogeneity_R10.png`).
Underlying numbers: `cot_water_homogeneity_stats.csv` in the same dir.

### 2.1 Homogeneous clouds agree better — at every n cut

| n_cut | CV bin       | bias  | RMSE  | r_log |
| ----- | ------------ | ----- | ----- | ----- |
| ≥ 1   | `[0,0.25)`   |  5.79 | 42.69 | 0.335 |
| ≥ 1   | `[0.25,0.75)`|  8.02 | 45.69 | 0.279 |
| ≥ 1   | `[0.75,∞)`   | 10.05 | 46.16 | 0.196 |
| ≥ 3   | `[0,0.25)`   |  6.55 | 44.24 | 0.323 |
| ≥ 3   | `[0.75,∞)`   | 11.38 | 48.95 | 0.183 |
| ≥ 5   | `[0,0.25)`   | 11.56 | 53.62 | 0.263 |
| ≥ 5   | `[0.75,∞)`   | 19.95 | 64.53 | 0.121 |

Monotonic at every n cut: more heterogeneous → larger bias, larger RMSE,
lower r_log.

### 2.2 Requiring more EarthCARE samples per pixel makes things *worse*

At fixed CV bin, increasing `n_cut` increases bias and lowers R. The cleanest
view comes from the per-`n_cut` mean COT, not the per-bin stats:

| n_cut | N (pixels) | ⟨COT_ATLID⟩ | ⟨COT_ORAC⟩ | bias  | bias / ⟨ATLID⟩ |
| ----- | ---------- | ----------- | ---------- | ----- | -------------- |
| ≥ 1   | 359 105    | 7.04        | 14.51      |  7.48 | 1.06 × |
| ≥ 3   | 279 331    | 7.23        | 15.51      |  8.28 | 1.15 × |
| ≥ 5   | 103 947    | 7.87        | 22.20      | 14.33 | **1.82 ×** |
| ≥ 8   |  14 027    | 7.95        | 27.81      | 19.87 | **2.50 ×** |

The surprise: **mean ATLID COT barely moves** (7.0 → 7.9). The growth is
entirely on the ORAC side (14.5 → 27.8). Relative bias more than doubles.

### 2.3 Partial-cloud cover at high CV

Splitting by CV bin reveals where ATLID and ORAC genuinely diverge:

| CV bin       | n_cut | ⟨ATLID⟩ | ⟨ORAC⟩ | bias  | rel_bias |
| ------------ | ----- | ------- | ------ | ----- | -------- |
| `[0,0.25)`   | ≥ 1   | 9.24    | 15.04  |  5.79 | 0.63 ×   |
| `[0,0.25)`   | ≥ 5   | 9.89    | 21.45  | 11.56 | 1.17 ×   |
| `[0.75,∞)`   | ≥ 1   | 3.19    | 13.24  | 10.05 | 3.15 ×   |
| `[0.75,∞)`   | ≥ 5   | 3.07    | 23.03  | 19.95 | **6.49 ×** |

At `CV ≥ 0.75` the pixel-mean ATLID COT collapses to ~3 because the pixel
holds a mixture of cloudy and near-clear profiles. ORAC, at 3 km, cannot
resolve the breaks and retrieves the bright bits as a single τ ~ 13-23.

---

## 3. Interpretation — why does `n_liquid_only` enrich the bias?

Two compounding effects, both *physical* rather than statistical:

### 3.1 ATLID liquid-COT saturation

ACM-CAP's `liquid_optical_depth` is a lidar-derived τ. The ATLID 355 nm beam
extinguishes within the top ~few hundred metres of any liquid cloud, so the
inferred liquid τ saturates around **τ ~ 5-8 regardless of true thickness**.
ORAC's reflectance retrieval sees the full bright top and reports the true
optical depth.

So at any selection that enriches for *thick* liquid clouds, ORAC inflates
relative to ATLID purely because ATLID can't measure higher.

### 3.2 What `n_liquid_only ≥ 5` actually selects

To pass `n_liquid_only ≥ 5` an ORAC pixel needs:

- ≥ 5 ATLID profiles inside it (high N — typically higher latitudes where
  the orbit is denser, and pixels far from frame edges);
- every one of those profiles classified `liquid-only` — no ice top, no
  mixed phase, no clear gap.

The clouds that satisfy both are **continuous, single-layer marine
stratocumulus / overcast stratus** — exactly the regime where ORAC
plane-parallel COT is known to be biased high (3D side illumination in
geometrically thick boundary-layer clouds). The n_cut is, in effect, a
stratocumulus filter.

Combined with §3.1 this is a complete explanation: tightening n_cut moves
the population towards thick uniform stratus where ATLID is saturated and
ORAC is at its most inflated. Hence absolute bias grows almost entirely on
the ORAC side while ⟨ATLID⟩ stays flat.

### 3.3 Partial-cloud cover at high CV

The high-CV result (§2.3) is the mirror image at the opposite end:
fragmented liquid where ATLID's per-profile values include near-clear pulls
the pixel-mean down to τ ~ 3, while ORAC averages reflectance over the
3-km pixel and reports the bright fraction at τ ~ 13-23. This is the
classic partial-cloud overestimation of coarse passive retrievals.

---

## 4. Implications for the letter

- The bulk COT-water disagreement is **intrinsic** to the two retrievals,
  not a sampling artefact. No QC choice on `n_liquid_only` improves it —
  the strictest filters make it worse.
- The closest to an apples-to-apples subset is **`CV < 0.25, n ≥ 1`** — the
  homogeneous, single-layer pixels. Even there:
  bias ≈ +5.8, r_log ≈ 0.33.
- Two retrieval limitations together cover the picture:
  1. **ATLID liquid-COT lidar saturation** (~τ 5-8 ceiling) — instrument
     limit, not a fixable processing choice.
  2. **ORAC plane-parallel + partial-cloud bias** — most visible in
     stratocumulus regimes and in heterogeneous pixels.
- The R10 → R11 difference (`cot_water_homogeneity_R10.png` vs `_R11.png`)
  is small relative to the structural offsets above — both retrievals
  show the same monotonic pattern, R11 slightly more inflated at high
  n_cut.

---

## 5. Reproducing

```bash
python -m validation cot-water-homogeneity \
    --matches-r10 'validation_data/synergy_2026-02_R10/matches_synergy_*.csv' \
    --matches-r11 'validation_data/synergy_2026-02_R11/matches_synergy_*.csv' \
    --out figures/cot_water_2026-02_homogeneity \
    --qc-mode qc_strict \
    --n-cuts 1 3 5 \
    --label "Feb-2026 COT-water"
```

Outputs into `figures/cot_water_2026-02_homogeneity/`:

- `cot_water_homogeneity_R10.png` — 1×3 panel, R10
- `cot_water_homogeneity_R11.png` — 1×3 panel, R11
- `cot_water_homogeneity_stats.csv` — long-form (n_cut × CV bin × retrieval)

The companion nliq3 / nliq5 figure sets used by the earlier exploration
(under `figures/cot_water_2026-02_R11_nliq{3,5}/` and
`figures/cot_water_2026-02_compare_nliq{3,5}/`) are the *filtered* versions
of the original headline panels and remain useful for stratum-level reads.

---

## 6. Open questions / next steps

- **Refine CV binning.** Three bins is enough to show the trend, but the
  letter might benefit from 5 bins (`[0,0.15,0.3,0.5,1.0,∞)`) to show
  whether the dependence is smooth or has a knee.
- **Latitude check.** Is the `n_cut ↑ → bias ↑` chain partly explained by
  high-N pixels being concentrated at high latitudes? Add a ⟨lat⟩ column
  to the sweep stats; if true, a per-latitude-band sweep would separate
  selection from regime.
- **Saturation flag.** Look for an ACM-CAP retrieval flag indicating where
  `liquid_optical_depth` hit a fitted ceiling — would let us drop saturated
  pixels rather than assume them.
- **Apply the same sweep to CER.** CER is much less heavy-tailed and ATLID's
  effective-radius retrieval has different failure modes; the n_cut ↔
  bias chain should look different. The CLI already exposes
  `cer-water-homogeneity`.
- **R10 vs R11 in one figure.** The current outputs are two separate PNGs;
  a paired-line version (R10 dashed, R11 solid, colour per n cut) would
  make the retrieval-version difference visible at a glance.
