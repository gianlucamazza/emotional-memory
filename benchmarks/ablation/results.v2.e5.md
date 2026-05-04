# AFT Layer Ablation Study

Measures the isolated contribution of each AFT layer to `top1_accuracy` on the realistic replay benchmark.

95% CI via percentile bootstrap (n=2000, seed=0). Pairwise delta = ablated - full (negative = layer helps).

**Note on `no_appraisal`**: the realistic benchmark injects affect directly via `set_affect()`, so no appraisal engine is configured. This ablation is a no-op on this benchmark and confirms correct flag hook-up only.

**Note on `dual_path` (He1 pre-reg v3)**: uses `KeywordAppraisalEngine` + slow-path `elaborate()`. He1 compares dual_path vs `full_aft` (pure preset affect). KeywordAppraisalEngine degrades affect on this dataset (G3/Addendum A: aft_keyword=0.16 vs aft_noAppraisal=0.78), so He1 FAIL is the expected outcome. The discriminative Hf1 comparison (dual_path vs aft_keyword_synchronous) is below.

**Note on `no_reconsolidation` (He2 pre-reg v3)**: disables the APE-gated reconsolidation window; predictive-learning (`update_prediction`) still runs.

**Note on `aft_keyword_synchronous` (Hf1 pre-reg Addendum F)**: synchronous keyword appraisal baseline. Compare vs `dual_path` (deferred) to test whether deferral mitigates the destructive override observed in G3/Addendum A. Hf1 PASS expected: dual_path=0.35 > aft_keyword_synchronous≈0.16.

## Results by Variant

| Variant | top1 [95% CI] | hit@k [95% CI] | N queries |
| ------- | ------------- | -------------- | --------- |
| `full` | 0.51 [0.43, 0.57] | 0.64 [0.57, 0.71] | 200 |
| `no_appraisal` | 0.51 [0.44, 0.58] | 0.63 [0.56, 0.70] | 200 |
| `no_mood` | 0.50 [0.43, 0.57] | 0.62 [0.55, 0.69] | 200 |
| `no_momentum` | 0.51 [0.43, 0.57] | 0.63 [0.56, 0.70] | 200 |
| `no_resonance` | 0.59 [0.52, 0.66] | 0.69 [0.62, 0.75] | 200 |
| `no_reconsolidation` | 0.53 [0.46, 0.60] | 0.65 [0.58, 0.71] | 200 |
| `dual_path` | 0.24 [0.18, 0.30] | 0.44 [0.38, 0.51] | 200 |
| `aft_keyword_synchronous` | 0.06 [0.03, 0.10] | 0.17 [0.12, 0.22] | 200 |

## Pairwise vs Full (top1_accuracy)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.01 [-0.03, 0.04] | 0.880 | 1.000 | 1.000 | 0.021 | 200 | 11 |
| `no_mood` | -0.01 [-0.05, 0.04] | 0.915 | 1.000 | 1.000 | -0.016 | 200 | 19 |
| `no_momentum` | 0.00 [-0.03, 0.03] | 1.000 | 1.000 | 1.000 | 0.000 | 200 | 10 |
| `no_resonance` | 0.09 [0.04, 0.13] | 0.000 | 0.000 | 0.000 | 0.270 | 200 | 21 |
| `no_reconsolidation` | 0.03 [-0.01, 0.06] | 0.209 | 0.836 | 0.267 | 0.098 | 200 | 13 |
| `dual_path` | -0.27 [-0.34, -0.19] | 0.000 | 0.000 | 0.000 | -0.493 | 200 | 71 |
| `aft_keyword_synchronous` | -0.45 [-0.52, -0.37] | 0.000 | 0.000 | 0.000 | -0.826 | 200 | 97 |

## Pairwise vs Full (hit@k)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | -0.01 [-0.03, 0.00] | 0.261 | 0.781 | 0.500 | -0.100 | 200 | 2 |
| `no_mood` | -0.02 [-0.04, 0.00] | 0.147 | 0.588 | 0.219 | -0.116 | 200 | 6 |
| `no_momentum` | -0.01 [-0.03, 0.00] | 0.273 | 0.781 | 0.500 | -0.100 | 200 | 2 |
| `no_resonance` | 0.04 [0.01, 0.08] | 0.009 | 0.043 | 0.012 | 0.194 | 200 | 11 |
| `no_reconsolidation` | 0.01 [-0.01, 0.03] | 0.770 | 0.781 | 1.000 | 0.041 | 200 | 3 |
| `dual_path` | -0.20 [-0.26, -0.14] | 0.000 | 0.000 | 0.000 | -0.497 | 200 | 40 |
| `aft_keyword_synchronous` | -0.47 [-0.55, -0.41] | 0.000 | 0.000 | 0.000 | -0.909 | 200 | 99 |

## Interpretation

A variant with Δ significantly negative (CI entirely below 0, p_adj < 0.05 after Holm-Bonferroni correction) indicates that layer **contributes** to retrieval quality: removing it hurts performance. A variant with Δ ≈ 0 and small |d| indicates the layer has no measurable impact on this benchmark — either the signal is redundant or the dataset is too small to detect the effect (limited power at N=100 queries).

## Supplementary: Hf1 — dual_path vs aft_keyword_synchronous (Addendum F)

Hf1 tests whether deferring keyword appraisal (slow-path `elaborate()`) partially mitigates the destructive override of synchronous keyword appraisal.

| Metric | dual_path | aft_keyword_synchronous | Δ (Hf1) | Verdict |
|--------|-----------|------------------------|---------|---------|
| top1   | 0.240 | 0.060 | +0.1800 | **Hf1 PASS** |

Note: paired bootstrap and Holm-corrected p-values for each variant vs full_aft are in the pairwise tables above. Direct Hf1 statistical significance requires a separate pairwise bootstrap not computed here; the delta direction is the pre-registered criterion.
