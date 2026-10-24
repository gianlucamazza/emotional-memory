# Realistic Replay Benchmark

This report summarizes the comparative multi-session benchmark that uses
persisted affective state and memory carry-over across scripted sessions.

Headline metric: `top1_accuracy`. `hit@k` remains secondary support. 95% CI via percentile bootstrap (n=2000, seed=7).

| System | Queries | top1 [95% CI] | hit@k [95% CI] | Min candidates | Non-trivial queries [95% CI] | Stateful sessions |
|---|---:|---:|---:|---:|---:|---:|
| `cosine` | 36 | 0.69 [0.53, 0.83] | 0.81 [0.67, 0.92] | 3 | 0.67 [0.50, 0.83] | 0.14 |
| `gated` | 36 | 0.72 [0.58, 0.86] | 0.86 [0.75, 0.97] | 3 | 0.67 [0.50, 0.83] | 0.14 |
| `always_on` | 36 | 0.69 [0.53, 0.83] | 0.81 [0.67, 0.92] | 3 | 0.67 [0.50, 0.83] | 0.14 |

## Binary relevance ranking metrics

MRR@k is truncated at each query's k; no relevant result gives zero.

| System | Recall@k | nDCG@k | MRR@k |
|---|---:|---:|---:|
| cosine | 0.8056 | 0.7645 | 0.7500 |
| gated | 0.8611 | 0.8099 | 0.7917 |
| always_on | 0.8056 | 0.7645 | 0.7500 |

## State divergence

62 of 248 paired event snapshots differ from cosine. Difference is not quality.

## Reproducibility

Source digest: `83292fc14d744b0474aa11a11cebb52b944624aa75e395a2a7fa31984cfe5906`.
Full configuration, dataset hash and environment are in JSON/protocol.

## Per Scenario

### `cosine`

| Scenario | Queries | top1 [95% CI] | hit@k [95% CI] | Min candidates | Non-trivial queries [95% CI] | Stateful sessions |
|---|---:|---:|---:|---:|---:|---:|
| `kitchen_renovation_pivot` | 3 | 0.67 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `grant_review_reversal` | 3 | 0.33 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `bakery_opening_delay` | 3 | 0.67 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `orchestra_chair_challenge` | 3 | 0.67 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `community_garden_conflict` | 3 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `parent_teacher_conference` | 3 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `ferry_strike_homecoming` | 3 | 0.67 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `archive_digitization_scare` | 3 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `foster_cat_adoption` | 3 | 0.33 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.50 |
| `mountain_rescue_training` | 3 | 0.67 [0.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.50 |
| `immigration_paperwork_scare` | 3 | 0.67 [0.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `startup_outage_postmortem` | 3 | 0.67 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |

### `gated`

| Scenario | Queries | top1 [95% CI] | hit@k [95% CI] | Min candidates | Non-trivial queries [95% CI] | Stateful sessions |
|---|---:|---:|---:|---:|---:|---:|
| `kitchen_renovation_pivot` | 3 | 0.67 [0.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `grant_review_reversal` | 3 | 0.33 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `bakery_opening_delay` | 3 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `orchestra_chair_challenge` | 3 | 0.67 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `community_garden_conflict` | 3 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `parent_teacher_conference` | 3 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `ferry_strike_homecoming` | 3 | 0.67 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `archive_digitization_scare` | 3 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `foster_cat_adoption` | 3 | 0.33 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.50 |
| `mountain_rescue_training` | 3 | 0.67 [0.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.50 |
| `immigration_paperwork_scare` | 3 | 0.67 [0.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `startup_outage_postmortem` | 3 | 0.67 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |

### `always_on`

| Scenario | Queries | top1 [95% CI] | hit@k [95% CI] | Min candidates | Non-trivial queries [95% CI] | Stateful sessions |
|---|---:|---:|---:|---:|---:|---:|
| `kitchen_renovation_pivot` | 3 | 0.67 [0.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `grant_review_reversal` | 3 | 0.33 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `bakery_opening_delay` | 3 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `orchestra_chair_challenge` | 3 | 0.67 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `community_garden_conflict` | 3 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `parent_teacher_conference` | 3 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `ferry_strike_homecoming` | 3 | 0.67 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `archive_digitization_scare` | 3 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `foster_cat_adoption` | 3 | 0.33 [0.00, 1.00] | 0.33 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.50 |
| `mountain_rescue_training` | 3 | 0.67 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.50 |
| `immigration_paperwork_scare` | 3 | 0.67 [0.00, 1.00] | 1.00 [1.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |
| `startup_outage_postmortem` | 3 | 0.33 [0.00, 1.00] | 0.67 [0.00, 1.00] | 3 | 0.67 [0.00, 1.00] | 0.00 |

## By Challenge Type

### `cosine`

| Challenge Type | Queries | top1 [95% CI] | hit@k [95% CI] | Min candidates | Non-trivial queries [95% CI] |
|---|---:|---:|---:|---:|---:|
| `affective_arc` | 5 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4 | 0.00 [0.00, 0.00] |
| `momentum_alignment` | 3 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4 | 0.00 [0.00, 0.00] |
| `recency_confound` | 12 | 0.75 [0.50, 1.00] | 1.00 [1.00, 1.00] | 3 | 1.00 [1.00, 1.00] |
| `same_topic_distractor` | 5 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4 | 0.20 [0.00, 0.60] |
| `semantic_confound` | 11 | 0.27 [0.00, 0.55] | 0.36 [0.09, 0.64] | 5 | 1.00 [1.00, 1.00] |

### `gated`

| Challenge Type | Queries | top1 [95% CI] | hit@k [95% CI] | Min candidates | Non-trivial queries [95% CI] |
|---|---:|---:|---:|---:|---:|
| `affective_arc` | 5 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4 | 0.00 [0.00, 0.00] |
| `momentum_alignment` | 3 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4 | 0.00 [0.00, 0.00] |
| `recency_confound` | 12 | 0.75 [0.50, 1.00] | 1.00 [1.00, 1.00] | 3 | 1.00 [1.00, 1.00] |
| `same_topic_distractor` | 5 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4 | 0.20 [0.00, 0.60] |
| `semantic_confound` | 11 | 0.36 [0.09, 0.64] | 0.55 [0.27, 0.82] | 5 | 1.00 [1.00, 1.00] |

### `always_on`

| Challenge Type | Queries | top1 [95% CI] | hit@k [95% CI] | Min candidates | Non-trivial queries [95% CI] |
|---|---:|---:|---:|---:|---:|
| `affective_arc` | 5 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4 | 0.00 [0.00, 0.00] |
| `momentum_alignment` | 3 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4 | 0.00 [0.00, 0.00] |
| `recency_confound` | 12 | 0.67 [0.42, 0.92] | 0.83 [0.58, 1.00] | 3 | 1.00 [1.00, 1.00] |
| `same_topic_distractor` | 5 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 4 | 0.20 [0.00, 0.60] |
| `semantic_confound` | 11 | 0.36 [0.09, 0.64] | 0.55 [0.27, 0.82] | 5 | 1.00 [1.00, 1.00] |

## Pairwise vs cosine

Two-sided tests: paired bootstrap p-value and exact McNemar p-value.
H0: no difference. CI excludes 0 ↔ difference is credible at 95% level.

| System | Metric | Δ [95% CI] | p (bootstrap) | p (McNemar) | d (Hedges g) | N | Discordant |
|---|---|---:|---:|---:|---:|---:|---:|
| `gated` | top1 | 0.03 [0.00, 0.08] | 0.6340 | 1.0000 | 0.1631 | 36 | 1 |
| `gated` | hit@k | 0.06 [0.00, 0.14] | 0.2735 | 0.5000 | 0.2340 | 36 | 2 |
| `always_on` | top1 | 0.00 [-0.08, 0.08] | 1.0000 | 1.0000 | 0.0000 | 36 | 2 |
| `always_on` | hit@k | 0.00 [-0.11, 0.11] | 1.0000 | 1.0000 | 0.0000 | 36 | 4 |
