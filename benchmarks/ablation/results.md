# AFT Layer Ablation Study

Measures the isolated contribution of each AFT layer to `top1_accuracy` on the realistic replay benchmark.

95% CI via percentile bootstrap (n=2000, seed=0). Pairwise delta = ablated - full (negative = layer helps).

**Note on `no_appraisal`**: the realistic benchmark injects affect directly via `set_affect()`, so no appraisal engine is configured. This ablation is a no-op on this benchmark and confirms correct flag hook-up only.

## Results by Variant

| Variant | top1 [95% CI] | hit@k [95% CI] | N queries |
| ------- | ------------- | -------------- | --------- |
| `full` | 0.70 [0.61, 0.79] | 0.80 [0.72, 0.87] | 100 |
| `no_appraisal` | 0.70 [0.61, 0.79] | 0.80 [0.72, 0.87] | 100 |
| `no_mood` | 0.69 [0.60, 0.78] | 0.78 [0.69, 0.86] | 100 |
| `no_momentum` | 0.70 [0.61, 0.79] | 0.79 [0.71, 0.86] | 100 |
| `no_resonance` | 0.71 [0.62, 0.80] | 0.79 [0.71, 0.87] | 100 |

## Pairwise vs Full (top1_accuracy)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 100 | 0 |
| `no_mood` | -0.01 [-0.05, 0.02] | 0.753 | 1.000 | 1.000 | -0.057 | 100 | 3 |
| `no_momentum` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 100 | 0 |
| `no_resonance` | 0.01 [-0.02, 0.04] | 0.748 | 1.000 | 1.000 | 0.057 | 100 | 3 |

## Pairwise vs Full (hit@k)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 100 | 0 |
| `no_mood` | -0.02 [-0.05, 0.00] | 0.274 | 1.000 | 0.500 | -0.141 | 100 | 2 |
| `no_momentum` | -0.01 [-0.03, 0.00] | 0.625 | 1.000 | 1.000 | -0.099 | 100 | 1 |
| `no_resonance` | -0.01 [-0.03, 0.00] | 0.630 | 1.000 | 1.000 | -0.099 | 100 | 1 |

## Interpretation

A variant with Δ significantly negative (CI entirely below 0, p_adj < 0.05 after Holm-Bonferroni correction) indicates that layer **contributes** to retrieval quality: removing it hurts performance. A variant with Δ ≈ 0 and small |d| indicates the layer has no measurable impact on this benchmark — either the signal is redundant or the dataset is too small to detect the effect (limited power at N=100 queries).
