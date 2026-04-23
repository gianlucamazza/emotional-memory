# AFT Layer Ablation Study

Measures the isolated contribution of each AFT layer to `top1_accuracy` on the realistic replay benchmark.

95% CI via percentile bootstrap (n=2000, seed=0). Pairwise delta = ablated - full (negative = layer helps).

**Note on `no_appraisal`**: the realistic benchmark injects affect directly via `set_affect()`, so no appraisal engine is configured. This ablation is a no-op on this benchmark and confirms correct flag hook-up only.

## Results by Variant

| Variant | top1 [95% CI] | hit@k [95% CI] | N queries |
| ------- | ------------- | -------------- | --------- |
| `full` | 0.85 [0.70, 1.00] | 1.00 [1.00, 1.00] | 20 |
| `no_appraisal` | 0.85 [0.70, 1.00] | 1.00 [1.00, 1.00] | 20 |
| `no_mood` | 0.75 [0.55, 0.95] | 1.00 [1.00, 1.00] | 20 |
| `no_momentum` | 0.85 [0.70, 1.00] | 1.00 [1.00, 1.00] | 20 |
| `no_resonance` | 0.85 [0.70, 1.00] | 0.95 [0.85, 1.00] | 20 |

## Pairwise vs Full (top1_accuracy)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 20 | 0 |
| `no_mood` | -0.10 [-0.25, 0.00] | 0.253 | 1.000 | 0.500 | -0.312 | 20 | 2 |
| `no_momentum` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 20 | 0 |
| `no_resonance` | 0.00 [-0.15, 0.15] | 1.000 | 1.000 | 1.000 | 0.000 | 20 | 2 |

## Pairwise vs Full (hit@k)

| Variant | Δ [95% CI] | p (bootstrap) | p_adj (Holm) | p (McNemar) | d (Hedges g) | N | Discordant |
| ------- | ---------- | ------------- | ------------ | ----------- | ------------ | - | ---------- |
| `no_appraisal` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 20 | 0 |
| `no_mood` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 20 | 0 |
| `no_momentum` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | — | 20 | 0 |
| `no_resonance` | -0.05 [-0.15, 0.00] | 0.635 | 1.000 | 1.000 | -0.215 | 20 | 1 |

## Interpretation

A variant with Δ significantly negative (CI entirely below 0, p_adj < 0.05 after Holm-Bonferroni correction) indicates that layer **contributes** to retrieval quality: removing it hurts performance. A variant with Δ ≈ 0 and small |d| indicates the layer has no measurable impact on this benchmark — either the signal is redundant or the dataset is too small to detect the effect (N=20 queries has limited power).
