# AFT Layer Ablation Study

Measures the isolated contribution of each AFT layer to `top1_accuracy` on the realistic replay benchmark.

95% CI via percentile bootstrap (n=2000, seed=0). Pairwise delta = ablated - full (negative = layer helps).

**Note on `no_appraisal`**: the realistic benchmark injects affect directly via `set_affect()`, so no appraisal engine is configured. This ablation is a no-op on this benchmark and confirms correct flag hook-up only.

**Note on `dual_path` (He1 pre-reg v3)**: uses `KeywordAppraisalEngine` + slow-path `elaborate()`. He1 compares dual_path vs `full_aft` (pure preset affect). KeywordAppraisalEngine degrades affect on this dataset (G3/Addendum A: aft_keyword=0.16 vs aft_noAppraisal=0.78), so He1 FAIL is the expected outcome. The discriminative comparison (dual_path vs synchronous-keyword) is Addendum F.

**Note on `no_reconsolidation` (He2 pre-reg v3)**: disables the APE-gated reconsolidation window; predictive-learning (`update_prediction`) still runs.

## Results by Variant

| Variant | top1 [95% CI] | hit@k [95% CI] | N queries |
| ------- | ------------- | -------------- | --------- |
| `full` | 0.20 [0.13, 0.28] | 0.23 [0.15, 0.32] | 100 |
| `no_appraisal` | 0.19 [0.12, 0.27] | 0.22 [0.14, 0.31] | 100 |
| `no_mood` | 0.18 [0.11, 0.26] | 0.22 [0.14, 0.31] | 100 |
| `no_momentum` | 0.19 [0.12, 0.27] | 0.22 [0.14, 0.31] | 100 |
| `no_resonance` | 0.18 [0.11, 0.26] | 0.22 [0.14, 0.31] | 100 |
| `no_reconsolidation` | 0.18 [0.11, 0.26] | 0.22 [0.14, 0.31] | 100 |
| `dual_path` | 0.11 [0.05, 0.18] | 0.15 [0.09, 0.22] | 100 |

## Pairwise vs Full (top1_accuracy)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | -0.01 [-0.03, 0.00] | 0.625 | 1.000 | 1.000 | -0.099 | 100 | 1 |
| `no_mood` | -0.02 [-0.05, 0.00] | 0.270 | 1.000 | 0.500 | -0.141 | 100 | 2 |
| `no_momentum` | -0.01 [-0.03, 0.00] | 0.625 | 1.000 | 1.000 | -0.099 | 100 | 1 |
| `no_resonance` | -0.02 [-0.06, 0.02] | 0.436 | 1.000 | 0.625 | -0.099 | 100 | 4 |
| `no_reconsolidation` | -0.02 [-0.05, 0.00] | 0.270 | 1.000 | 0.500 | -0.141 | 100 | 2 |
| `dual_path` | -0.09 [-0.15, -0.03] | 0.011 | 0.063 | 0.012 | -0.278 | 100 | 11 |

## Pairwise vs Full (hit@k)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | -0.01 [-0.03, 0.00] | 0.625 | 1.000 | 1.000 | -0.099 | 100 | 1 |
| `no_mood` | -0.01 [-0.03, 0.00] | 0.611 | 1.000 | 1.000 | -0.099 | 100 | 1 |
| `no_momentum` | -0.01 [-0.03, 0.00] | 0.625 | 1.000 | 1.000 | -0.099 | 100 | 1 |
| `no_resonance` | -0.01 [-0.05, 0.02] | 0.755 | 1.000 | 1.000 | -0.057 | 100 | 3 |
| `no_reconsolidation` | -0.01 [-0.03, 0.00] | 0.635 | 1.000 | 1.000 | -0.099 | 100 | 1 |
| `dual_path` | -0.08 [-0.14, -0.02] | 0.018 | 0.108 | 0.021 | -0.258 | 100 | 10 |

## Interpretation

A variant with Δ significantly negative (CI entirely below 0, p_adj < 0.05 after Holm-Bonferroni correction) indicates that layer **contributes** to retrieval quality: removing it hurts performance. A variant with Δ ≈ 0 and small |d| indicates the layer has no measurable impact on this benchmark — either the signal is redundant or the dataset is too small to detect the effect (limited power at N=100 queries).
