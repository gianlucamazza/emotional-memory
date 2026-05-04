# AFT Layer Ablation Study

Measures the isolated contribution of each AFT layer to `top1_accuracy` on the realistic replay benchmark.

95% CI via percentile bootstrap (n=10000, seed=0). Pairwise delta = ablated - full (negative = layer helps).

**Note on `no_appraisal`**: the realistic benchmark injects affect directly via `set_affect()`, so no appraisal engine is configured. This ablation is a no-op on this benchmark and confirms correct flag hook-up only.

**Note on `dual_path` (He1 pre-reg v3)**: uses `KeywordAppraisalEngine` + slow-path `elaborate()`. He1 compares dual_path vs `full_aft` (pure preset affect). KeywordAppraisalEngine degrades affect on this dataset (G3/Addendum A: aft_keyword=0.16 vs aft_noAppraisal=0.78), so He1 FAIL is the expected outcome. The discriminative Hf1 comparison (dual_path vs aft_keyword_synchronous) is below.

**Note on `no_reconsolidation` (He2 pre-reg v3)**: disables the APE-gated reconsolidation window; predictive-learning (`update_prediction`) still runs.

**Note on `aft_keyword_synchronous` (Hf1 pre-reg Addendum F)**: synchronous keyword appraisal baseline. Compare vs `dual_path` (deferred) to test whether deferral mitigates the destructive override observed in G3/Addendum A. Hf1 PASS expected: dual_path=0.35 > aft_keyword_synchronous≈0.16.

## Results by Variant

| Variant | top1 [95% CI] | hit@k [95% CI] | N queries |
| ------- | ------------- | -------------- | --------- |
| `full` | 0.69 [0.60, 0.78] | 0.79 [0.71, 0.87] | 100 |
| `no_appraisal` | 0.70 [0.61, 0.79] | 0.79 [0.71, 0.87] | 100 |
| `no_mood` | 0.69 [0.60, 0.78] | 0.78 [0.70, 0.86] | 100 |
| `no_momentum` | 0.70 [0.61, 0.79] | 0.77 [0.69, 0.85] | 100 |
| `no_resonance` | 0.71 [0.62, 0.80] | 0.80 [0.72, 0.87] | 100 |
| `no_reconsolidation` | 0.69 [0.60, 0.78] | 0.78 [0.70, 0.86] | 100 |
| `dual_path` | 0.37 [0.28, 0.46] | 0.54 [0.44, 0.64] | 100 |
| `aft_keyword_synchronous` | 0.08 [0.03, 0.14] | 0.23 [0.15, 0.31] | 100 |

## Pairwise vs Full (top1_accuracy)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.01 [0.00, 0.03] | 0.625 | 1.000 | 1.000 | 0.099 | 100 | 1 |
| `no_mood` | 0.00 [-0.04, 0.04] | 1.000 | 1.000 | 1.000 | 0.000 | 100 | 4 |
| `no_momentum` | 0.01 [0.00, 0.03] | 0.625 | 1.000 | 1.000 | 0.099 | 100 | 1 |
| `no_resonance` | 0.02 [0.00, 0.05] | 0.270 | 1.000 | 0.500 | 0.141 | 100 | 2 |
| `no_reconsolidation` | 0.00 [-0.03, 0.03] | 1.000 | 1.000 | 1.000 | 0.000 | 100 | 2 |
| `dual_path` | -0.32 [-0.42, -0.23] | 0.000 | 0.000 | 0.000 | -0.648 | 100 | 34 |
| `aft_keyword_synchronous` | -0.61 [-0.70, -0.51] | 0.000 | 0.000 | 0.000 | -1.235 | 100 | 61 |

## Pairwise vs Full (hit@k)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.00 [-0.03, 0.03] | 1.000 | 1.000 | 1.000 | 0.000 | 100 | 2 |
| `no_mood` | -0.01 [-0.03, 0.00] | 0.627 | 1.000 | 1.000 | -0.099 | 100 | 1 |
| `no_momentum` | -0.02 [-0.05, 0.00] | 0.273 | 1.000 | 0.500 | -0.141 | 100 | 2 |
| `no_resonance` | 0.01 [-0.02, 0.04] | 0.760 | 1.000 | 1.000 | 0.057 | 100 | 3 |
| `no_reconsolidation` | -0.01 [-0.05, 0.02] | 0.757 | 1.000 | 1.000 | -0.057 | 100 | 3 |
| `dual_path` | -0.25 [-0.34, -0.17] | 0.000 | 0.000 | 0.000 | -0.570 | 100 | 25 |
| `aft_keyword_synchronous` | -0.56 [-0.66, -0.46] | 0.000 | 0.000 | 0.000 | -1.114 | 100 | 56 |

## Interpretation

A variant with Δ significantly negative (CI entirely below 0, p_adj < 0.05 after Holm-Bonferroni correction) indicates that layer **contributes** to retrieval quality: removing it hurts performance. A variant with Δ ≈ 0 and small |d| indicates the layer has no measurable impact on this benchmark — either the signal is redundant or the dataset is too small to detect the effect (limited power at N=100 queries).

## Supplementary: Hf1 — dual_path vs aft_keyword_synchronous (Addendum F)

Hf1 tests whether deferring keyword appraisal (slow-path `elaborate()`) partially mitigates the destructive override of synchronous keyword appraisal.

Pre-registered criterion (Addendum F): paired bootstrap (n=10,000, seed=0), one-tailed p < 0.05 **and** bootstrap CI for Δ_Hf1 fully above 0.

| Metric | dual_path | aft_keyword_synchronous | Δ [95% CI] | p (one-tailed) | CI above 0 | Verdict |
|--------|-----------|------------------------|------------|----------------|------------|---------|
| top1 | 0.370 | 0.080 | 0.29 [0.19, 0.39] | 0.0000 | ✓ | **Hf1 PASS** |
| hit@k | 0.540 | 0.230 | 0.31 [0.21, 0.41] | 0.0000 | ✓ | — |
