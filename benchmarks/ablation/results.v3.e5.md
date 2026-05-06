# AFT Layer Ablation Study

Measures the isolated contribution of each AFT layer to `top1_accuracy` on the realistic replay benchmark.

95% CI via percentile bootstrap (n=2000, seed=1). Pairwise delta = ablated - full (negative = layer helps).

**Note on `no_appraisal`**: the realistic benchmark injects affect directly via `set_affect()`, so no appraisal engine is configured. This ablation is a no-op on this benchmark and confirms correct flag hook-up only.

**Note on `dual_path` (He1 pre-reg v3)**: uses `KeywordAppraisalEngine` + slow-path `elaborate()`. He1 compares dual_path vs `full_aft` (pure preset affect). KeywordAppraisalEngine degrades affect on this dataset (G3/Addendum A: aft_keyword=0.16 vs aft_noAppraisal=0.78), so He1 FAIL is the expected outcome. The discriminative Hf1 comparison (dual_path vs aft_keyword_synchronous) is below.

**Note on `no_reconsolidation` (He2 pre-reg v3)**: disables the APE-gated reconsolidation window; predictive-learning (`update_prediction`) still runs.

**Note on `aft_keyword_synchronous` (Hf1 pre-reg Addendum F)**: synchronous keyword appraisal baseline. Compare vs `dual_path` (deferred) to test whether deferral mitigates the destructive override observed in G3/Addendum A. Hf1 PASS expected: dual_path=0.35 > aft_keyword_synchronous≈0.16.

## Results by Variant

| Variant | top1 [95% CI] | hit@k [95% CI] | N queries |
| ------- | ------------- | -------------- | --------- |
| `full` | 0.69 [0.65, 0.73] | 0.78 [0.74, 0.82] | 500 |
| `no_appraisal` | 0.70 [0.65, 0.73] | 0.78 [0.75, 0.82] | 500 |
| `no_mood` | 0.68 [0.64, 0.72] | 0.77 [0.73, 0.81] | 500 |
| `no_momentum` | 0.70 [0.65, 0.74] | 0.78 [0.74, 0.82] | 500 |
| `no_resonance` | 0.75 [0.71, 0.78] | 0.81 [0.78, 0.85] | 500 |
| `no_reconsolidation` | 0.71 [0.67, 0.75] | 0.79 [0.75, 0.82] | 500 |
| `dual_path` | 0.43 [0.39, 0.47] | 0.64 [0.60, 0.68] | 500 |
| `aft_keyword_synchronous` | 0.09 [0.07, 0.11] | 0.23 [0.19, 0.26] | 500 |

## Pairwise vs Full (top1_accuracy)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.01 [-0.00, 0.02] | 0.218 | 0.437 | 0.267 | 0.062 | 500 | 13 |
| `no_mood` | -0.00 [-0.02, 0.02] | 0.756 | 0.756 | 0.839 | -0.018 | 500 | 24 |
| `no_momentum` | 0.01 [0.00, 0.02] | 0.044 | 0.132 | 0.062 | 0.100 | 500 | 5 |
| `no_resonance` | 0.06 [0.04, 0.08] | 0.000 | 0.000 | 0.000 | 0.252 | 500 | 30 |
| `no_reconsolidation` | 0.02 [0.01, 0.04] | 0.011 | 0.044 | 0.013 | 0.120 | 500 | 14 |
| `dual_path` | -0.26 [-0.30, -0.21] | 0.000 | 0.000 | 0.000 | -0.502 | 500 | 162 |
| `aft_keyword_synchronous` | -0.60 [-0.64, -0.55] | 0.000 | 0.000 | 0.000 | -1.147 | 500 | 312 |

## Pairwise vs Full (hit@k)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.00 [-0.01, 0.01] | 0.856 | 1.000 | 1.000 | 0.017 | 500 | 7 |
| `no_mood` | -0.01 [-0.02, 0.00] | 0.130 | 0.520 | 0.180 | -0.075 | 500 | 9 |
| `no_momentum` | 0.00 [-0.01, 0.01] | 1.000 | 1.000 | 1.000 | 0.000 | 500 | 2 |
| `no_resonance` | 0.03 [0.01, 0.05] | 0.001 | 0.005 | 0.000 | 0.165 | 500 | 17 |
| `no_reconsolidation` | 0.00 [-0.00, 0.01] | 0.424 | 1.000 | 0.625 | 0.045 | 500 | 4 |
| `dual_path` | -0.14 [-0.18, -0.11] | 0.000 | 0.000 | 0.000 | -0.380 | 500 | 82 |
| `aft_keyword_synchronous` | -0.56 [-0.60, -0.51] | 0.000 | 0.000 | 0.000 | -1.090 | 500 | 284 |

## Interpretation

A variant with Δ significantly negative (CI entirely below 0, p_adj < 0.05 after Holm-Bonferroni correction) indicates that layer **contributes** to retrieval quality: removing it hurts performance. A variant with Δ ≈ 0 and small |d| indicates the layer has no measurable impact on this benchmark — either the signal is redundant or the dataset is too small to detect the effect (limited power at N=100 queries).

## Supplementary: Hf1 — dual_path vs aft_keyword_synchronous (Addendum F)

Hf1 tests whether deferring keyword appraisal (slow-path `elaborate()`) partially mitigates the destructive override of synchronous keyword appraisal.

Pre-registered criterion (Addendum F): paired bootstrap (n=10,000, seed=0), one-tailed p < 0.05 **and** bootstrap CI for Δ_Hf1 fully above 0.

| Metric | dual_path | aft_kw_sync | Δ [95% CI] | p (one-tailed) | CI>0 | Verdict |
|--------|-----------|------------|------------|----------------|------|---------|
| top1 | 0.430 | 0.090 | 0.34 [0.29, 0.38] | 0.0000 | ✓ | **Hf1 PASS** |
| hit@k | 0.638 | 0.226 | 0.41 [0.36, 0.46] | 0.0000 | ✓ | — |
