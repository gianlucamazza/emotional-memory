# AFT Layer Ablation Study

Measures the isolated contribution of each AFT layer to `top1_accuracy` on the realistic replay benchmark.

95% CI via percentile bootstrap (n=2000, seed=0). Pairwise delta = ablated - full (negative = layer helps).

**Note on `no_appraisal`**: the realistic benchmark injects affect directly via `set_affect()`, so no appraisal engine is configured. This ablation is a no-op on this benchmark and confirms correct flag hook-up only.

## Results by Variant

| Variant | top1 [95% CI] | hit@k [95% CI] | N queries |
| ------- | ------------- | -------------- | --------- |
| `full` | 0.60 [0.40, 0.80] | 0.65 [0.45, 0.85] | 20 |
| `no_appraisal` | 0.60 [0.40, 0.80] | 0.65 [0.45, 0.85] | 20 |
| `no_mood` | 0.55 [0.35, 0.75] | 0.60 [0.40, 0.80] | 20 |
| `no_momentum` | 0.60 [0.40, 0.80] | 0.65 [0.45, 0.85] | 20 |
| `no_resonance` | 0.55 [0.35, 0.75] | 0.60 [0.40, 0.80] | 20 |

## Pairwise vs Full (top1_accuracy)

| Variant | Δ [95% CI] | p (bootstrap) | p (McNemar) | N | Discordant |
| ------- | ---------- | ------------- | ----------- | - | ---------- |
| `no_appraisal` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 20 | 0 |
| `no_mood` | -0.05 [-0.15, 0.00] | 0.616 | 1.000 | 20 | 1 |
| `no_momentum` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 20 | 0 |
| `no_resonance` | -0.05 [-0.20, 0.10] | 0.750 | 1.000 | 20 | 3 |

## Pairwise vs Full (hit@k)

| Variant | Δ [95% CI] | p (bootstrap) | p (McNemar) | N | Discordant |
| ------- | ---------- | ------------- | ----------- | - | ---------- |
| `no_appraisal` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 20 | 0 |
| `no_mood` | -0.05 [-0.15, 0.00] | 0.606 | 1.000 | 20 | 1 |
| `no_momentum` | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 20 | 0 |
| `no_resonance` | -0.05 [-0.15, 0.00] | 0.621 | 1.000 | 20 | 1 |

## Interpretation

A variant with Δ significantly negative (CI entirely below 0, p < 0.05) indicates that layer **contributes** to retrieval quality: removing it hurts performance. A variant with Δ ≈ 0 indicates the layer has no measurable impact on this benchmark — either the signal is redundant or the dataset is too small to detect the effect.
