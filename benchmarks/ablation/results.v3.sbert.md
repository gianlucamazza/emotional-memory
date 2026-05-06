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
| `full` | 0.64 [0.60, 0.69] | 0.72 [0.68, 0.76] | 500 |
| `no_appraisal` | 0.64 [0.60, 0.69] | 0.72 [0.68, 0.76] | 500 |
| `no_mood` | 0.65 [0.61, 0.69] | 0.72 [0.68, 0.76] | 500 |
| `no_momentum` | 0.64 [0.60, 0.69] | 0.72 [0.68, 0.76] | 500 |
| `no_resonance` | 0.66 [0.62, 0.70] | 0.73 [0.69, 0.77] | 500 |
| `no_reconsolidation` | 0.66 [0.62, 0.70] | 0.72 [0.68, 0.76] | 500 |
| `dual_path` | 0.48 [0.44, 0.53] | 0.62 [0.58, 0.67] | 500 |
| `aft_keyword_synchronous` | 0.13 [0.10, 0.16] | 0.29 [0.25, 0.32] | 500 |

## Pairwise vs Full (top1_accuracy)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.00 [-0.01, 0.01] | 1.000 | 1.000 | 1.000 | 0.000 | 500 | 2 |
| `no_mood` | 0.01 [-0.01, 0.02] | 0.518 | 1.000 | 0.607 | 0.035 | 500 | 15 |
| `no_momentum` | 0.00 [-0.01, 0.01] | 1.000 | 1.000 | 1.000 | 0.000 | 500 | 10 |
| `no_resonance` | 0.02 [0.00, 0.03] | 0.029 | 0.118 | 0.035 | 0.104 | 500 | 15 |
| `no_reconsolidation` | 0.02 [0.01, 0.03] | 0.009 | 0.048 | 0.008 | 0.127 | 500 | 8 |
| `dual_path` | -0.16 [-0.20, -0.12] | 0.000 | 0.000 | 0.000 | -0.374 | 500 | 104 |
| `aft_keyword_synchronous` | -0.51 [-0.56, -0.47] | 0.000 | 0.000 | 0.000 | -0.962 | 500 | 272 |

## Pairwise vs Full (hit@k)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 500 | 0 |
| `no_mood` | 0.00 [-0.00, 0.01] | 0.449 | 1.000 | 0.625 | 0.045 | 500 | 4 |
| `no_momentum` | 0.00 [0.00, 0.01] | 0.281 | 1.000 | 0.500 | 0.063 | 500 | 2 |
| `no_resonance` | 0.01 [0.00, 0.03] | 0.058 | 0.287 | 0.065 | 0.095 | 500 | 11 |
| `no_reconsolidation` | 0.00 [0.00, 0.01] | 0.281 | 1.000 | 0.500 | 0.063 | 500 | 2 |
| `dual_path` | -0.10 [-0.12, -0.07] | 0.000 | 0.000 | 0.000 | -0.305 | 500 | 54 |
| `aft_keyword_synchronous` | -0.43 [-0.47, -0.39] | 0.000 | 0.000 | 0.000 | -0.849 | 500 | 222 |

## Interpretation

A variant with Δ significantly negative (CI entirely below 0, p_adj < 0.05 after Holm-Bonferroni correction) indicates that layer **contributes** to retrieval quality: removing it hurts performance. A variant with Δ ≈ 0 and small |d| indicates the layer has no measurable impact on this benchmark — either the signal is redundant or the dataset is too small to detect the effect (limited power at N=100 queries).

## Supplementary: Hf1 — dual_path vs aft_keyword_synchronous (Addendum F)

Hf1 tests whether deferring keyword appraisal (slow-path `elaborate()`) partially mitigates the destructive override of synchronous keyword appraisal.

Pre-registered criterion (Addendum F): paired bootstrap (n=10,000, seed=0), one-tailed p < 0.05 **and** bootstrap CI for Δ_Hf1 fully above 0.

| Metric | dual_path | aft_kw_sync | Δ [95% CI] | p (one-tailed) | CI>0 | Verdict |
|--------|-----------|------------|------------|----------------|------|---------|
| top1 | 0.484 | 0.132 | 0.35 [0.31, 0.40] | 0.0000 | ✓ | **Hf1 PASS** |
| hit@k | 0.622 | 0.286 | 0.34 [0.29, 0.38] | 0.0000 | ✓ | — |
