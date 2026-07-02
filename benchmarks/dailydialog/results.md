# Hk1 — DailyDialog Affect-Conditioned Retrieval Benchmark

**Personas:** 5  **Embedder:** `multilingual-e5-small`  **Bootstrap:** n=10000, seed=0

## Aggregate Results

| System | N queries | top1_accuracy | hit@k |
|---|---|---|---|
| aft | 16 | 0.312 | 0.375 |
| naive_cosine | 16 | 0.250 | 0.375 |

## By Query Type

### aft

| Query type | N | top1_accuracy | hit@k |
|---|---|---|---|
| emotion_state_recall | 5 | 0.200 | 0.200 |
| affect_conditioned_content | 5 | 0.400 | 0.400 |
| affective_trajectory | 1 | 1.000 | 1.000 |
| cross_session_control | 5 | 0.200 | 0.400 |

### naive_cosine

| Query type | N | top1_accuracy | hit@k |
|---|---|---|---|
| emotion_state_recall | 5 | 0.200 | 0.200 |
| affect_conditioned_content | 5 | 0.200 | 0.400 |
| affective_trajectory | 1 | 1.000 | 1.000 |
| cross_session_control | 5 | 0.200 | 0.400 |

## Pairwise Comparisons (AFT vs baseline, Holm-corrected)

### AFT vs naive_cosine

| Key | Δ | CI | p_one | p_holm | d | Verdict |
|---|---|---|---|---|---|---|
| aggregate | +0.062 | 0.062 [-0.125, 0.250] | 0.385 | 1.000 | 0.141 | FAIL Δ=+0.062 p_holm=1.000 |
| emotion_state_recall | +0.000 | 0.000 [0.000, 0.000] | 0.500 | 1.000 | nan | FAIL Δ=+0.000 p_holm=1.000 |
| affect_conditioned_content | +0.200 | 0.200 [0.000, 0.600] | 0.293 | 1.000 | 0.447 | FAIL Δ=+0.200 p_holm=1.000 |
| affective_trajectory | +0.000 | 0.000 [0.000, 0.000] | 0.500 | 1.000 | nan | FAIL Δ=+0.000 p_holm=1.000 |
| cross_session_control | +0.000 | 0.000 [-0.600, 0.600] | 0.500 | 1.000 | 0.000 | FAIL Δ=+0.000 p_holm=1.000 |

## Hk1 Decision

Aggregate: FAIL Δ=+0.062 p_holm=1.000

Types passing Holm: 0/3 directional types

**Hk1 verdict: FAIL**

Headline metric: `top1_accuracy`. See `benchmarks/preregistration_addendum_k_dailydialog.md` for decision rule.
