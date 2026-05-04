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
| `full` | 0.54 [0.47, 0.60] | 0.64 [0.57, 0.70] | 200 |
| `no_appraisal` | 0.53 [0.46, 0.59] | 0.64 [0.57, 0.70] | 200 |
| `no_mood` | 0.52 [0.45, 0.58] | 0.64 [0.57, 0.70] | 200 |
| `no_momentum` | 0.56 [0.49, 0.62] | 0.65 [0.58, 0.71] | 200 |
| `no_resonance` | 0.56 [0.48, 0.62] | 0.67 [0.60, 0.73] | 200 |
| `no_reconsolidation` | 0.54 [0.47, 0.60] | 0.65 [0.58, 0.71] | 200 |
| `dual_path` | 0.34 [0.28, 0.41] | 0.48 [0.41, 0.55] | 200 |
| `aft_keyword_synchronous` | 0.09 [0.05, 0.13] | 0.22 [0.17, 0.28] | 200 |

## Pairwise vs Full (top1_accuracy)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | -0.01 [-0.03, 0.00] | 0.283 | 0.812 | 0.500 | -0.100 | 200 | 2 |
| `no_mood` | -0.02 [-0.05, 0.01] | 0.264 | 0.812 | 0.344 | -0.089 | 200 | 10 |
| `no_momentum` | 0.02 [0.01, 0.04] | 0.067 | 0.333 | 0.125 | 0.142 | 200 | 4 |
| `no_resonance` | 0.02 [-0.01, 0.05] | 0.203 | 0.812 | 0.289 | 0.100 | 200 | 8 |
| `no_reconsolidation` | 0.01 [-0.01, 0.03] | 0.831 | 0.831 | 1.000 | 0.031 | 200 | 5 |
| `dual_path` | -0.20 [-0.26, -0.14] | 0.000 | 0.000 | 0.000 | -0.437 | 200 | 47 |
| `aft_keyword_synchronous` | -0.45 [-0.52, -0.37] | 0.000 | 0.000 | 0.000 | -0.826 | 200 | 97 |

## Pairwise vs Full (hit@k)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 200 | 0 |
| `no_mood` | 0.01 [-0.02, 0.03] | 0.837 | 1.000 | 1.000 | 0.027 | 200 | 7 |
| `no_momentum` | 0.01 [0.00, 0.04] | 0.144 | 0.576 | 0.250 | 0.123 | 200 | 3 |
| `no_resonance` | 0.03 [0.01, 0.06] | 0.051 | 0.253 | 0.070 | 0.151 | 200 | 8 |
| `no_reconsolidation` | 0.01 [0.00, 0.03] | 0.287 | 0.863 | 0.500 | 0.100 | 200 | 2 |
| `dual_path` | -0.15 [-0.20, -0.10] | 0.000 | 0.000 | 0.000 | -0.388 | 200 | 34 |
| `aft_keyword_synchronous` | -0.41 [-0.48, -0.34] | 0.000 | 0.000 | 0.000 | -0.804 | 200 | 87 |

## Interpretation

A variant with Δ significantly negative (CI entirely below 0, p_adj < 0.05 after Holm-Bonferroni correction) indicates that layer **contributes** to retrieval quality: removing it hurts performance. A variant with Δ ≈ 0 and small |d| indicates the layer has no measurable impact on this benchmark — either the signal is redundant or the dataset is too small to detect the effect (limited power at N=100 queries).

## Supplementary: Hf1 — dual_path vs aft_keyword_synchronous (Addendum F)

Hf1 tests whether deferring keyword appraisal (slow-path `elaborate()`) partially mitigates the destructive override of synchronous keyword appraisal.

| Metric | dual_path | aft_keyword_synchronous | Δ (Hf1) | Verdict |
|--------|-----------|------------------------|---------|---------|
| top1   | 0.340 | 0.090 | +0.2500 | **Hf1 PASS** |

Note: paired bootstrap and Holm-corrected p-values for each variant vs full_aft are in the pairwise tables above. Direct Hf1 statistical significance requires a separate pairwise bootstrap not computed here; the delta direction is the pre-registered criterion.
