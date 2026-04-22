# Realistic Replay Benchmark

This report summarizes the comparative multi-session benchmark that uses
persisted affective state and memory carry-over across scripted sessions.

Headline metric: `top1_accuracy`. `hit@k` remains secondary support.

| System | Queries | top1 | hit@k | Min candidates | Non-trivial queries | Stateful sessions |
|---|---:|---:|---:|---:|---:|---:|
| `aft` | 20 | 0.60 | 0.65 | 6 | 1.00 | 0.95 |
| `naive_cosine` | 20 | 0.55 | 0.55 | 6 | 1.00 | 0.00 |
| `recency` | 20 | 0.00 | 0.00 | 6 | 1.00 | 0.00 |

## Per Scenario

### `aft`

| Scenario | Queries | top1 | hit@k | Min candidates | Non-trivial queries | Stateful sessions |
|---|---:|---:|---:|---:|---:|---:|
| `customer_repair_arc` | 2 | 0.50 | 0.50 | 6 | 1.00 | 0.50 |
| `team_conflict_repair` | 2 | 1.00 | 1.00 | 6 | 1.00 | 1.00 |
| `family_health_reassurance` | 2 | 0.50 | 0.50 | 6 | 1.00 | 1.00 |
| `creative_block_breakthrough` | 2 | 0.50 | 0.50 | 6 | 1.00 | 1.00 |
| `founder_pitch_rebound` | 2 | 1.00 | 1.00 | 6 | 1.00 | 1.00 |
| `mentorship_revision_arc` | 2 | 1.00 | 1.00 | 6 | 1.00 | 1.00 |
| `housing_search_reversal` | 2 | 1.00 | 1.00 | 6 | 1.00 | 1.00 |
| `volunteer_burnout_reset` | 2 | 0.50 | 0.50 | 6 | 1.00 | 1.00 |
| `scholarship_reversal_arc` | 2 | 0.00 | 0.00 | 6 | 1.00 | 1.00 |
| `choir_audition_rebound` | 2 | 0.00 | 0.50 | 6 | 1.00 | 1.00 |

### `naive_cosine`

| Scenario | Queries | top1 | hit@k | Min candidates | Non-trivial queries | Stateful sessions |
|---|---:|---:|---:|---:|---:|---:|
| `customer_repair_arc` | 2 | 0.50 | 0.50 | 6 | 1.00 | 0.00 |
| `team_conflict_repair` | 2 | 1.00 | 1.00 | 6 | 1.00 | 0.00 |
| `family_health_reassurance` | 2 | 0.50 | 0.50 | 6 | 1.00 | 0.00 |
| `creative_block_breakthrough` | 2 | 0.50 | 0.50 | 6 | 1.00 | 0.00 |
| `founder_pitch_rebound` | 2 | 0.50 | 0.50 | 6 | 1.00 | 0.00 |
| `mentorship_revision_arc` | 2 | 1.00 | 1.00 | 6 | 1.00 | 0.00 |
| `housing_search_reversal` | 2 | 0.50 | 0.50 | 6 | 1.00 | 0.00 |
| `volunteer_burnout_reset` | 2 | 0.50 | 0.50 | 6 | 1.00 | 0.00 |
| `scholarship_reversal_arc` | 2 | 0.00 | 0.00 | 6 | 1.00 | 0.00 |
| `choir_audition_rebound` | 2 | 0.50 | 0.50 | 6 | 1.00 | 0.00 |

### `recency`

| Scenario | Queries | top1 | hit@k | Min candidates | Non-trivial queries | Stateful sessions |
|---|---:|---:|---:|---:|---:|---:|
| `customer_repair_arc` | 2 | 0.00 | 0.00 | 6 | 1.00 | 0.00 |
| `team_conflict_repair` | 2 | 0.00 | 0.00 | 6 | 1.00 | 0.00 |
| `family_health_reassurance` | 2 | 0.00 | 0.00 | 6 | 1.00 | 0.00 |
| `creative_block_breakthrough` | 2 | 0.00 | 0.00 | 6 | 1.00 | 0.00 |
| `founder_pitch_rebound` | 2 | 0.00 | 0.00 | 6 | 1.00 | 0.00 |
| `mentorship_revision_arc` | 2 | 0.00 | 0.00 | 6 | 1.00 | 0.00 |
| `housing_search_reversal` | 2 | 0.00 | 0.00 | 6 | 1.00 | 0.00 |
| `volunteer_burnout_reset` | 2 | 0.00 | 0.00 | 6 | 1.00 | 0.00 |
| `scholarship_reversal_arc` | 2 | 0.00 | 0.00 | 6 | 1.00 | 0.00 |
| `choir_audition_rebound` | 2 | 0.00 | 0.00 | 6 | 1.00 | 0.00 |

## By Challenge Type

### `aft`

| Challenge Type | Queries | top1 | hit@k | Min candidates | Non-trivial queries |
|---|---:|---:|---:|---:|---:|
| `affective_arc` | 4 | 1.00 | 1.00 | 6 | 1.00 |
| `recency_confound` | 2 | 0.50 | 0.50 | 6 | 1.00 |
| `same_topic_distractor` | 6 | 1.00 | 1.00 | 6 | 1.00 |
| `semantic_confound` | 8 | 0.12 | 0.25 | 6 | 1.00 |

### `naive_cosine`

| Challenge Type | Queries | top1 | hit@k | Min candidates | Non-trivial queries |
|---|---:|---:|---:|---:|---:|
| `affective_arc` | 4 | 0.75 | 0.75 | 6 | 1.00 |
| `recency_confound` | 2 | 0.50 | 0.50 | 6 | 1.00 |
| `same_topic_distractor` | 6 | 0.83 | 0.83 | 6 | 1.00 |
| `semantic_confound` | 8 | 0.25 | 0.25 | 6 | 1.00 |

### `recency`

| Challenge Type | Queries | top1 | hit@k | Min candidates | Non-trivial queries |
|---|---:|---:|---:|---:|---:|
| `affective_arc` | 4 | 0.00 | 0.00 | 6 | 1.00 |
| `recency_confound` | 2 | 0.00 | 0.00 | 6 | 1.00 |
| `same_topic_distractor` | 6 | 0.00 | 0.00 | 6 | 1.00 |
| `semantic_confound` | 8 | 0.00 | 0.00 | 6 | 1.00 |
