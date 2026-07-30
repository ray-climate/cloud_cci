# ORAC SLSTR retrieval-uncertainty validation against EarthCARE — December 2025

Do ORAC's **reported per-pixel uncertainties** actually describe its errors? This is
the error-consistency (z-score) test of Sayer et al. (2020, AMT, §3.1) / the
normalised-discrepancy diagnostic. For a retrieved quantity `x` with reported
1-σ uncertainties on both sides,

```
delta = (x_ORAC - x_ref) / sqrt(sigma_ORAC^2 + sigma_ref^2)
```

If the uncertainties are **calibrated** and the errors **Gaussian**, `delta ~ N(0,1)`:
centred on 0, standard deviation 1, ~68 % within ±1. We bin `delta` by the
reference value `x_ref`, show its distribution (violin), and read off the two
diagnostics:

- **std(delta) ≈ 1?** — are the uncertainty *magnitudes* right? std > 1 →
  **over-confident** (stated sigma too small); < 1 → under-confident.
- **is delta Gaussian?** — QQ plot + skew/kurtosis; heavy tails / skew mean the
  errors are not the shape a single sigma can summarise.
- **mean(delta) ≈ 0?** — a residual bias the uncertainty does not account for.

> **This validates the reported *uncertainty*, not the retrieved *value*.** The two
> are independent: a retrieval can be accurate on the typical pixel (small median
> bias, good correlation — as CTH is) yet report **error bars that are too tight**.
> "Good value" and "honest uncertainty" are different properties; this test only
> asks the second. So a variable can look best in the bias/correlation validation
> and still fail here — that is not a contradiction, it is the uncertainty analysis
> exposing an error tail the median bias hid.

## 1. What uncertainties are available

| side | source | form |
| ---- | ------ | ---- |
| ORAC | `cot_uncertainty`, `cer_uncertainty`, `cth_uncertainty`, `cwp_uncertainty`, … | per-pixel, absolute (retrieval a-posteriori) |
| ACM-CAP | `liquid_optical_depth_error`, `*_effective_radius_error`, `ice_water_path_error` | per-profile, **fractional** (× value → absolute) |
| A-CTH | — (only confidence/consistency flags) | none → ATLID treated as truth (sigma_ref ≈ 0) |

So **water-COT** is the clean two-sided case; **CTH** is ORAC-sigma-only against a
near-truth lidar. (CER needs tau-weighted propagation of the per-bin radius error —
a follow-on.)

## 2. Water-cloud optical depth — calibrated only in the mid-τ sweet spot

Phase-agree liquid, qc_strict, daytime. sigma_ORAC (median 3.95 τ) and
sigma_ref = fractional_error × τ (median 1.35 τ). **N = 141 497.**

![COT error consistency](../../figures/slstr_uncertainty_2025-12/cot_error_consistency.png)

| τ (reference) band | median δ | robust std(δ) | reading |
| ------------------ | -------- | ------------- | ------- |
| thin (τ ≈ 0.7–2.5) | **+1 to +1.6** | 1.7–2.8 | over-reads thin cloud, over-confident |
| **mid (τ ≈ 4–8)**  | **≈ 0** | **≈ 1.0** | **well-calibrated** |
| thick (τ ≈ 12–27)  | **−1 to −2.6** | 2.3–2.8 | saturates, σ does not widen enough |

- **std(δ) traces a clear U** (panel b): it dips to **~1.0 at τ 4–8** — ORAC's COT
  uncertainty is *honest* there — and rises to **~2.5–3 at both the thin and thick
  ends**. The uncertainty is **over-confident wherever the retrieval is hardest.**
- **Overall δ: median −0.91, robust std 2.31, 48 % within ±1** (vs 68 %). Two
  failures compound: a **~1σ negative bias** (the saturation underestimate is *not*
  absorbed by the stated uncertainty) and **~2.3× over-dispersion**.
- **Not Gaussian** (panel c): skew −0.8, excess kurtosis +0.9; the QQ curve is
  S-shaped — a near-flat core (the mid-τ pile at δ≈0) with heavy tails.
- **Bottom line:** ORAC's COT uncertainty **does not "know about" its own
  saturation** — at high τ the error reaches −2.6 σ, far outside the stated 1 σ.
  It is trustworthy only for mid-optical-depth liquid cloud.

## 3. Cloud-top height — the uncertainty is a-priori-saturated and over-confident for high cloud

ATLID CTH is treated as truth (its precision ≈ one range bin ≪ ORAC's km-level σ).

- **The stated σ is largely non-informative:** it is pinned at a **20 km a-priori
  cap for 51 % (raw) / 63 % (corrected) of pixels** — the thermal retrieval added
  no height information there. Only ~36–46 % of pixels carry an informative σ.
- **On that informative subset, the story splits cleanly at ~6 km cloud height:**

![CTH error consistency](../../figures/slstr_uncertainty_2025-12/cth_error_consistency.png)

| cloud | N | median δ | robust std(δ) | within ±1 | σ vs actual \|error\| |
| ----- | -- | -------- | ------------- | --------- | --------------------- |
| **low (< 6 km)**  | 55 315 | +0.7 | **2.1** | 34 % | σ 0.24 km vs 0.49 km → **2×** |
| **high (≥ 6 km)** | 27 893 | **−43** | **60** | 6 % | σ 0.29 km vs 9.7 km → **33×** |

- **Low cloud: ORAC's CTH uncertainty is essentially fine** — δ sits in the ±1 band
  (robust std ~2, only mildly over-confident, as any retrieval is), and the QQ
  curve (green, panel c) roughly follows N(0,1). The stated ±0.24 km is close to
  the actual ±0.49 km error.
- **High cloud: it is catastrophically over-confident** — ORAC reports ±0.3 km when
  its top is really ~9.7 km wrong (33×), so δ collapses (robust std 60, only 6 %
  within ±1, QQ curve in red far off the diagonal). ORAC places **high thin-cirrus
  tops too low with unwarranted confidence** — the same missed-cirrus mechanism as
  the cloud-mask validation (88 % of missed cloud is thin ice).
- **std(δ) climbs monotonically with cloud height** (panel b), from ~1.7 at 3–5 km
  to ~40 at 14 km — the failure is entirely a *high-cloud* phenomenon.
- **Bottom line:** ORAC's CTH uncertainty is **trustworthy for low cloud** and
  **badly over-confident for high cloud** (plus a-priori-capped for the >50 % of
  pixels it cannot constrain). The large discrepancies come from high cloud, not
  from the retrieval being uniformly bad.

## 4. Conclusions

1. **ORAC's uncertainties are optimistic in exactly the hard regimes.** COT σ is
   calibrated for mid-τ liquid but over-confident for thin and (especially)
   saturated-thick cloud; CTH σ is a-priori-capped and over-confident for high
   cloud. Where the retrieval physics is easy the error bars are honest; where it
   struggles they are too tight.
2. **The stated uncertainty does not capture the two structural limitations** found
   earlier — the bright-surface τ saturation (a −2.6 σ effect at high τ) and the
   missed high cirrus (a −5 σ effect in CTH). A user propagating ORAC's σ would be
   over-confident about precisely the pixels that are worst.
3. **Method is general and re-runnable** — CER (needs τ-weighted reference-error
   propagation) and ice-COT / IWP are natural follow-ons using the same augment.

## 5. Reproducibility

```bash
# ORAC sigma (cot_uncertainty) + ACM-CAP fractional error, matched by ec_time
python scripts/slstr_uncertainty_augment.py          # -> .uncertainty_cache/cot_unc_pairs.parquet
# CTH sigma diagnostic (caches cth_unc.parquet)
python scripts/slstr_uncertainty_cth.py
# violin / std-vs-x_ref / QQ figures for COT and CTH
python scripts/slstr_uncertainty_figure.py           # -> figures/slstr_uncertainty_2025-12/
```

δ is the normalised discrepancy `(x_ORAC − x_ref)/sqrt(σ_ORAC² + σ_ref²)`; for CTH
σ_ref ≈ 0 (ATLID truth) and only the informative-σ subset (σ below the 20 km
a-priori cap) is shown. Reference: Sayer et al. (2020), *AMT* 13, 373 (§3.1).
