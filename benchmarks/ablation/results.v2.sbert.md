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
| `full` | 0.54 [0.47, 0.60] | 0.64 [0.57, 0.70] | 200 |
| `no_appraisal` | 0.54 [0.47, 0.60] | 0.64 [0.57, 0.70] | 200 |
| `no_mood` | 0.54 [0.47, 0.60] | 0.64 [0.57, 0.70] | 200 |
| `no_momentum` | 0.54 [0.47, 0.60] | 0.64 [0.57, 0.70] | 200 |
| `no_resonance` | 0.56 [0.49, 0.63] | 0.67 [0.60, 0.73] | 200 |
| `no_reconsolidation` | 0.55 [0.47, 0.61] | 0.64 [0.57, 0.70] | 200 |
| `dual_path` | 0.35 [0.28, 0.41] | 0.49 [0.42, 0.56] | 200 |
| `aft_keyword_synchronous` | 0.10 [0.06, 0.14] | 0.24 [0.18, 0.30] | 200 |

## Pairwise vs Full (top1_accuracy)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 200 | 0 |
| `no_mood` | 0.00 [-0.03, 0.03] | 1.000 | 1.000 | 1.000 | 0.000 | 200 | 8 |
| `no_momentum` | 0.01 [0.00, 0.01] | 0.629 | 1.000 | 1.000 | 0.070 | 200 | 1 |
| `no_resonance` | 0.03 [0.01, 0.06] | 0.022 | 0.108 | 0.031 | 0.175 | 200 | 6 |
| `no_reconsolidation` | 0.01 [-0.01, 0.03] | 0.444 | 1.000 | 0.625 | 0.070 | 200 | 4 |
| `dual_path` | -0.18 [-0.25, -0.12] | 0.000 | 0.000 | 0.000 | -0.400 | 200 | 49 |
| `aft_keyword_synchronous` | -0.44 [-0.52, -0.36] | 0.000 | 0.000 | 0.000 | -0.817 | 200 | 96 |

## Pairwise vs Full (hit@k)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 200 | 0 |
| `no_mood` | 0.00 [-0.01, 0.01] | 1.000 | 1.000 | 1.000 | 0.000 | 200 | 2 |
| `no_momentum` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 200 | 0 |
| `no_resonance` | 0.03 [0.00, 0.06] | 0.127 | 0.633 | 0.180 | 0.118 | 200 | 9 |
| `no_reconsolidation` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 200 | 0 |
| `dual_path` | -0.14 [-0.20, -0.09] | 0.000 | 0.000 | 0.000 | -0.367 | 200 | 35 |
| `aft_keyword_synchronous` | -0.40 [-0.47, -0.33] | 0.000 | 0.000 | 0.000 | -0.765 | 200 | 86 |

## Interpretation

A variant with Δ significantly negative (CI entirely below 0, p_adj < 0.05 after Holm-Bonferroni correction) indicates that layer **contributes** to retrieval quality: removing it hurts performance. A variant with Δ ≈ 0 and small |d| indicates the layer has no measurable impact on this benchmark — either the signal is redundant or the dataset is too small to detect the effect (limited power at N=100 queries).

## Supplementary: Hf1 — dual_path vs aft_keyword_synchronous (Addendum F)

Hf1 tests whether deferring keyword appraisal (slow-path `elaborate()`) partially mitigates the destructive override of synchronous keyword appraisal.

Pre-registered criterion (Addendum F): paired bootstrap (n=10,000, seed=0), one-tailed p < 0.05 **and** bootstrap CI for Δ_Hf1 fully above 0.

| Metric | dual_path | aft_kw_sync | Δ [95% CI] | p (one-tailed) | CI>0 | Verdict |
|--------|-----------|------------|------------|----------------|------|---------|
| top1 | 0.350 | 0.095 | 0.26 [0.19, 0.32] | 0.0000 | ✓ | **Hf1 PASS** |
| hit@k | 0.495 | 0.240 | 0.26 [0.18, 0.33] | 0.0000 | ✓ | — |
