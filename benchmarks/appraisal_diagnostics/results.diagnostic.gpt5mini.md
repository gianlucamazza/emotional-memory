# Appraisal Diagnostics — WP-1a Report

N events: 750 | seed: 42 | dry_run: False

## Residuals (LLM - oracle)

| Dimension | Bias (mean) | 95% CI | Std | MAE | Pearson r |
|---|---|---|---|---|---|
| Valence | 0.186 | [0.159, 0.212] | 0.362 | 0.291 | 0.811 |
| Arousal | -0.138 | [-0.151, -0.125] | 0.180 | 0.185 | 0.369 |

## SEC Dimension Descriptives (LLM output, no oracle)

| Dimension | Mean | Std | Min | Max |
|---|---|---|---|---|
| novelty | 0.254 | 0.442 | -1.000 | 0.950 |
| goal_relevance | 0.297 | 0.710 | -0.980 | 1.000 |
| coping_potential | 0.674 | 0.251 | 0.050 | 1.000 |
| norm_congruence | 0.396 | 0.579 | -1.000 | 1.000 |
| self_relevance | 0.822 | 0.171 | 0.000 | 1.000 |

## Valence Sign Confusion (LLM vs oracle)

TP=453 FP=87 TN=195 FN=15 accuracy=0.86

## Latency

Mean: 5903.8 ms | P95: 8431.6 ms | Total: 4427.8 s

## Decision

P1d — BOTH systematic bias and high variance detected. Zero-shot LLM appraisal is unreliable. Recommended action: document limitation; consider pluggable fine-tuned appraisal (P1d).

---

Thresholds: bias > 0.1 → P1b/P1d; std > 0.3 → P1c/P1d.
