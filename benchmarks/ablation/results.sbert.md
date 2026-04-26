# AFT Layer Ablation Study

Measures the isolated contribution of each AFT layer to `top1_accuracy` on the realistic replay benchmark.

95% CI via percentile bootstrap (n=2000, seed=0). Pairwise delta = ablated - full (negative = layer helps).

**Note on `no_appraisal`**: the realistic benchmark injects affect directly via `set_affect()`, so no appraisal engine is configured. This ablation is a no-op on this benchmark and confirms correct flag hook-up only.

**Note on `dual_path` (He1 pre-reg v3)**: uses `KeywordAppraisalEngine` + slow-path `elaborate()`. He1 compares dual_path vs `full_aft` (pure preset affect). KeywordAppraisalEngine degrades affect on this dataset (G3/Addendum A: aft_keyword=0.16 vs aft_noAppraisal=0.78), so He1 FAIL is the expected outcome. The discriminative comparison (dual_path vs synchronous-keyword) is Addendum F.

**Note on `no_reconsolidation` (He2 pre-reg v3)**: disables the APE-gated reconsolidation window; predictive-learning (`update_prediction`) still runs.

## Results by Variant

| Variant | top1 [95% CI] | hit@k [95% CI] | N queries |
| ------- | ------------- | -------------- | --------- |
| `full` | 0.70 [0.61, 0.79] | 0.79 [0.71, 0.86] | 100 |
| `no_appraisal` | 0.70 [0.61, 0.79] | 0.79 [0.71, 0.86] | 100 |
| `no_mood` | 0.69 [0.60, 0.78] | 0.80 [0.72, 0.87] | 100 |
| `no_momentum` | 0.70 [0.61, 0.79] | 0.78 [0.70, 0.86] | 100 |
| `no_resonance` | 0.71 [0.62, 0.80] | 0.79 [0.71, 0.87] | 100 |
| `no_reconsolidation` | 0.70 [0.61, 0.79] | 0.79 [0.71, 0.86] | 100 |
| `dual_path` | 0.35 [0.26, 0.44] | 0.53 [0.43, 0.63] | 100 |

## Pairwise vs Full (top1_accuracy)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 100 | 0 |
| `no_mood` | -0.01 [-0.05, 0.02] | 0.753 | 1.000 | 1.000 | -0.057 | 100 | 3 |
| `no_momentum` | 0.00 [-0.03, 0.03] | 1.000 | 1.000 | 1.000 | 0.000 | 100 | 2 |
| `no_resonance` | 0.01 [-0.02, 0.04] | 0.748 | 1.000 | 1.000 | 0.057 | 100 | 3 |
| `no_reconsolidation` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 100 | 0 |
| `dual_path` | -0.35 [-0.45, -0.25] | 0.000 | 0.000 | 0.000 | -0.668 | 100 | 39 |

## Pairwise vs Full (hit@k)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 100 | 0 |
| `no_mood` | 0.01 [-0.02, 0.05] | 0.754 | 1.000 | 1.000 | 0.057 | 100 | 3 |
| `no_momentum` | -0.01 [-0.03, 0.00] | 0.630 | 1.000 | 1.000 | -0.099 | 100 | 1 |
| `no_resonance` | 0.00 [-0.03, 0.03] | 1.000 | 1.000 | 1.000 | 0.000 | 100 | 2 |
| `no_reconsolidation` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 100 | 0 |
| `dual_path` | -0.26 [-0.35, -0.18] | 0.000 | 0.000 | 0.000 | -0.585 | 100 | 26 |

## Interpretation

A variant with Δ significantly negative (CI entirely below 0, p_adj < 0.05 after Holm-Bonferroni correction) indicates that layer **contributes** to retrieval quality: removing it hurts performance. A variant with Δ ≈ 0 and small |d| indicates the layer has no measurable impact on this benchmark — either the signal is redundant or the dataset is too small to detect the effect (limited power at N=100 queries).
