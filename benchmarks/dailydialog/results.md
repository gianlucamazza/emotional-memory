# Hk1 — DailyDialog Affect-Conditioned Retrieval Benchmark

**Personas:** 120  **Embedder:** `multilingual-e5-small`  **Bootstrap:** n=10000, seed=0

## Aggregate Results

| System | N queries | top1_accuracy | hit@k |
|---|---|---|---|
| aft | 396 | 0.212 | 0.366 |
| naive_cosine | 396 | 0.220 | 0.389 |

## By Query Type

### aft

| Query type | N | top1_accuracy | hit@k |
|---|---|---|---|
| emotion_state_recall | 120 | 0.217 | 0.375 |
| affect_conditioned_content | 120 | 0.175 | 0.325 |
| affective_trajectory | 39 | 0.385 | 0.487 |
| cross_session_control | 117 | 0.188 | 0.359 |

### naive_cosine

| Query type | N | top1_accuracy | hit@k |
|---|---|---|---|
| emotion_state_recall | 120 | 0.225 | 0.392 |
| affect_conditioned_content | 120 | 0.217 | 0.408 |
| affective_trajectory | 39 | 0.282 | 0.385 |
| cross_session_control | 117 | 0.197 | 0.368 |

## Pairwise Comparisons (AFT vs baseline, Holm-corrected)

### AFT vs naive_cosine

| Key | Δ | CI | p_one | p_holm | d | Verdict |
|---|---|---|---|---|---|---|
| aggregate | -0.008 | -0.008 [-0.056, 0.043] | 0.596 | 1.000 | -0.015 | FAIL Δ=-0.008 p_holm=1.000 |
| emotion_state_recall | -0.008 | -0.008 [-0.092, 0.075] | 0.538 | 1.000 | -0.017 | FAIL Δ=-0.008 p_holm=1.000 |
| affect_conditioned_content | -0.042 | -0.042 [-0.125, 0.042] | 0.808 | 1.000 | -0.088 | FAIL Δ=-0.042 p_holm=1.000 |
| affective_trajectory | +0.103 | 0.103 [-0.077, 0.282] | 0.147 | 0.734 | 0.186 | FAIL Δ=+0.103 p_holm=0.734 |
| cross_session_control | -0.009 | -0.009 [-0.094, 0.077] | 0.540 | 1.000 | -0.017 | FAIL Δ=-0.009 p_holm=1.000 |

## Hk1 Decision

Aggregate: FAIL Δ=-0.008 p_holm=1.000

Types passing Holm: 0/3 directional types

**Hk1 verdict: FAIL**

Headline metric: `top1_accuracy`. See `benchmarks/preregistration_addendum_k_dailydialog.md` for decision rule.
