# Addendum T2A — Retrieve-time Query Appraisal on DailyDialog (Ht2a)

**Personas:** 120  **Embedder:** `multilingual-e5-small`  **Bootstrap:** n=10000, seed=0

## Aggregate

| Arm | N | top1_accuracy | hit@k |
|---|---|---|---|
| naive_cosine | 396 | 0.220 | 0.389 |
| aft | 396 | 0.202 | 0.364 |
| aft_query_appraised | 396 | 0.212 | 0.386 |

## Contrasts

### aft_query_appraised vs naive_cosine (Ht2a, Holm family)

| Key | Δ | CI | p_one | p_holm | Verdict |
|---|---|---|---|---|---|
| aggregate | -0.008 | [-0.056, +0.040] | 0.602 | 1.000 | FAIL Δ=-0.008 p=1.000 |
| emotion_state_recall | +0.008 | [-0.083, +0.100] | 0.465 | 1.000 | FAIL Δ=+0.008 p=1.000 |
| affect_conditioned_content | -0.017 | [-0.092, +0.058] | 0.626 | 1.000 | FAIL Δ=-0.017 p=1.000 |
| affective_trajectory | +0.077 | [-0.103, +0.256] | 0.237 | 1.000 | FAIL Δ=+0.077 p=1.000 |
| cross_session_control | -0.043 | [-0.128, +0.043] | 0.823 | 1.000 | FAIL Δ=-0.043 p=1.000 |

### aft_query_appraised vs aft (Ht2a-ref)

| Key | Δ | CI | p_one | p_holm | Verdict |
|---|---|---|---|---|---|
| aggregate | +0.010 | [-0.013, +0.033] | 0.224 | nan | FAIL Δ=+0.010 p=0.224 |

### aft vs naive_cosine (Hk1 reproduction)

| Key | Δ | CI | p_one | p_holm | Verdict |
|---|---|---|---|---|---|
| aggregate | -0.018 | [-0.066, +0.030] | 0.753 | nan | FAIL Δ=-0.018 p=0.753 |

## Diagnostic — appraised query affect vs target-session oracle PAD

N=396  valence r=0.688  arousal r=0.736

## Ht2a Decision

Aggregate (appraised vs cosine): FAIL Δ=-0.008 p=1.000
Directional types passing Holm: 0/3

**Ht2a verdict: FAIL**

Decision rule: see `benchmarks/preregistration_addendum_t2a_naturalistic_query_appraisal.md`.
