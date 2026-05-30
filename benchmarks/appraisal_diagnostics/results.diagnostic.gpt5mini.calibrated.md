# Appraisal Diagnostics — WP-1a Report

N events: 150 | seed: 42 | dry_run: False

## Residuals (LLM - oracle)

| Dimension | Bias (mean) | 95% CI | Std | MAE | Pearson r |
|---|---|---|---|---|---|
| Valence | 0.044 | [-0.005, 0.094] | 0.308 | 0.235 | 0.873 |
| Arousal | -0.118 | [-0.142, -0.093] | 0.152 | 0.160 | 0.555 |

## SEC Dimension Descriptives (LLM output, no oracle)

| Dimension | Mean | Std | Min | Max |
|---|---|---|---|---|
| novelty | 0.321 | 0.324 | -0.500 | 0.900 |
| goal_relevance | 0.168 | 0.670 | -1.000 | 0.900 |
| coping_potential | 0.595 | 0.262 | 0.100 | 0.950 |
| norm_congruence | 0.233 | 0.403 | -0.900 | 0.800 |
| self_relevance | 0.918 | 0.137 | 0.500 | 1.000 |

## Valence Sign Confusion (LLM vs oracle)

TP=93 FP=8 TN=45 FN=4 accuracy=0.92

## Latency

Mean: 7141.1 ms | P95: 11203.2 ms | Total: 1071.2 s

## Decision

P1d — BOTH systematic bias and high variance detected. Zero-shot LLM appraisal is unreliable. Recommended action: document limitation; consider pluggable fine-tuned appraisal (P1d).

---

Thresholds: bias > 0.1 → P1b/P1d; std > 0.3 → P1c/P1d.
