# Realistic Replay Benchmark

This report summarizes the comparative multi-session benchmark that uses
persisted affective state and memory carry-over across scripted sessions.

Headline metric: `top1_accuracy`. `hit@k` remains secondary support. 95% CI via percentile bootstrap (n=2000, seed=0).

| System | Queries | top1 [95% CI] | hit@k [95% CI] | Min candidates | Non-trivial queries [95% CI] | Stateful sessions |
|---|---:|---:|---:|---:|---:|---:|
| `aft` | 20 | 0.85 [0.70, 1.00] | 0.95 [0.85, 1.00] | 6 | 1.00 [1.00, 1.00] | 0.95 |
| `naive_cosine` | 20 | 0.75 [0.55, 0.95] | 0.85 [0.70, 1.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `recency` | 20 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |

## Per Scenario

### `aft`

| Scenario | Queries | top1 [95% CI] | hit@k [95% CI] | Min candidates | Non-trivial queries [95% CI] | Stateful sessions |
|---|---:|---:|---:|---:|---:|---:|
| `customer_repair_arc` | 2 | 0.50 [0.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 0.50 |
| `team_conflict_repair` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 1.00 |
| `family_health_reassurance` | 2 | 0.50 [0.00, 1.00] | 0.50 [0.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 1.00 |
| `creative_block_breakthrough` | 2 | 0.50 [0.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 1.00 |
| `founder_pitch_rebound` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 1.00 |
| `mentorship_revision_arc` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 1.00 |
| `housing_search_reversal` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 1.00 |
| `volunteer_burnout_reset` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 1.00 |
| `scholarship_reversal_arc` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 1.00 |
| `choir_audition_rebound` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 1.00 |

### `naive_cosine`

| Scenario | Queries | top1 [95% CI] | hit@k [95% CI] | Min candidates | Non-trivial queries [95% CI] | Stateful sessions |
|---|---:|---:|---:|---:|---:|---:|
| `customer_repair_arc` | 2 | 0.50 [0.00, 1.00] | 0.50 [0.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `team_conflict_repair` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `family_health_reassurance` | 2 | 0.50 [0.00, 1.00] | 0.50 [0.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `creative_block_breakthrough` | 2 | 0.00 [0.00, 0.00] | 0.50 [0.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `founder_pitch_rebound` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `mentorship_revision_arc` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `housing_search_reversal` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `volunteer_burnout_reset` | 2 | 0.50 [0.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `scholarship_reversal_arc` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `choir_audition_rebound` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |

### `recency`

| Scenario | Queries | top1 [95% CI] | hit@k [95% CI] | Min candidates | Non-trivial queries [95% CI] | Stateful sessions |
|---|---:|---:|---:|---:|---:|---:|
| `customer_repair_arc` | 2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `team_conflict_repair` | 2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `family_health_reassurance` | 2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `creative_block_breakthrough` | 2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `founder_pitch_rebound` | 2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `mentorship_revision_arc` | 2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `housing_search_reversal` | 2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `volunteer_burnout_reset` | 2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `scholarship_reversal_arc` | 2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |
| `choir_audition_rebound` | 2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] | 0.00 |

## By Challenge Type

### `aft`

| Challenge Type | Queries | top1 [95% CI] | hit@k [95% CI] | Min candidates | Non-trivial queries [95% CI] |
|---|---:|---:|---:|---:|---:|
| `affective_arc` | 4 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] |
| `recency_confound` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] |
| `same_topic_distractor` | 6 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] |
| `semantic_confound` | 8 | 0.62 [0.25, 0.88] | 0.88 [0.62, 1.00] | 6 | 1.00 [1.00, 1.00] |

### `naive_cosine`

| Challenge Type | Queries | top1 [95% CI] | hit@k [95% CI] | Min candidates | Non-trivial queries [95% CI] |
|---|---:|---:|---:|---:|---:|
| `affective_arc` | 4 | 0.75 [0.25, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] |
| `recency_confound` | 2 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] |
| `same_topic_distractor` | 6 | 0.83 [0.50, 1.00] | 1.00 [1.00, 1.00] | 6 | 1.00 [1.00, 1.00] |
| `semantic_confound` | 8 | 0.62 [0.25, 0.88] | 0.62 [0.25, 0.88] | 6 | 1.00 [1.00, 1.00] |

### `recency`

| Challenge Type | Queries | top1 [95% CI] | hit@k [95% CI] | Min candidates | Non-trivial queries [95% CI] |
|---|---:|---:|---:|---:|---:|
| `affective_arc` | 4 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] |
| `recency_confound` | 2 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] |
| `same_topic_distractor` | 6 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] |
| `semantic_confound` | 8 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 6 | 1.00 [1.00, 1.00] |

## Pairwise vs naive_cosine

Two-sided tests: paired bootstrap p-value and exact McNemar p-value.
H0: no difference. CI excludes 0 ↔ difference is credible at 95% level.

| System | Metric | Δ [95% CI] | p (bootstrap) | p (McNemar) | N | Discordant |
|---|---|---:|---:|---:|---:|---:|
| `aft` | top1 | 0.10 [0.00, 0.25] | 0.2505 | 0.5000 | 20 | 2 |
| `aft` | hit@k | 0.10 [0.00, 0.25] | 0.2520 | 0.5000 | 20 | 2 |
| `recency` | top1 | -0.75 [-0.90, -0.55] | 0.0000 | 0.0001 | 20 | 15 |
| `recency` | hit@k | -0.85 [-1.00, -0.70] | 0.0000 | 0.0000 | 20 | 17 |
