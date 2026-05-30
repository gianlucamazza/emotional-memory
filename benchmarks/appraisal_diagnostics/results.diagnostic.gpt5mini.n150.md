# Appraisal Diagnostics — WP-1a Report

N events: 150 | seed: 42 | dry_run: False

## Residuals (LLM - oracle)

| Dimension | Bias (mean) | 95% CI | Std | MAE | Pearson r |
|---|---|---|---|---|---|
| Valence | 0.169 | [0.123, 0.215] | 0.286 | 0.236 | 0.883 |
| Arousal | -0.115 | [-0.141, -0.089] | 0.162 | 0.163 | 0.483 |

## SEC Dimension Descriptives (LLM output, no oracle)

| Dimension | Mean | Std | Min | Max |
|---|---|---|---|---|
| novelty | 0.270 | 0.448 | -0.800 | 0.900 |
| goal_relevance | 0.310 | 0.730 | -1.000 | 1.000 |
| coping_potential | 0.655 | 0.268 | 0.100 | 0.950 |
| norm_congruence | 0.395 | 0.588 | -1.000 | 0.950 |
| self_relevance | 0.840 | 0.142 | 0.200 | 1.000 |

## Valence Sign Confusion (LLM vs oracle)

TP=95 FP=9 TN=44 FN=2 accuracy=0.93

## Latency

Mean: 5829.7 ms | P95: 7709.3 ms | Total: 874.5 s

## Decision

P1b — Systematic bias detected (|mean residual| > threshold). Recommended action: fix LLM appraisal prompt to reduce directional error.

---

Thresholds: bias > 0.1 → P1b/P1d; std > 0.3 → P1c/P1d.
