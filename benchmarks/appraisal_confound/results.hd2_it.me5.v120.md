# Appraisal Confound Study

Dataset: realistic_recall_v2_it v1.0  (30 scenarios, 120 queries)  
Embedder: `multilingual-e5-small`  n_bootstrap: 10000  seed: 42

## System Results

| System | N | top1_acc | 95% CI |
|---|---:|---:|---|
| `aft_noAppraisal` | 120 | 0.333 | [0.250, 0.417] |
| `aft_keyword` | 120 | 0.067 | [0.025, 0.117] |
| `naive_cosine` | 120 | 0.275 | [0.200, 0.358] |

## Hypothesis Tests

### Hd2_IT — ✗ FAIL

**aft_noAppraisal.top1 > naive_cosine.top1, Δ > 0.10 (Addendum D generalization)**

Δ = 0.06 [-0.04, 0.16]  p_two_sided = 0.2760  Cohen's d = 0.105

*AFT without appraisal does not show a practically significant advantage over naive cosine at Δ > 0.10 threshold.*

### Ha2 — ✗ FAIL

**aft_keyword.top1 > naive_cosine.top1 (Δ > 0.05, one-tailed alpha=0.05)**

Δ = -0.21 [-0.30, -0.12]  p_one_sided = 0.0000  Cohen's d = -0.403

*No significant advantage of AFT+keyword over naive cosine. Architecture benefit not confirmed on this dataset.*

### Hb2 — ✗ FAIL

**|aft_keyword - aft_noAppraisal| < 0.05 (equivalence)**

Δ = -0.27 [-0.37, -0.17]  p_two_sided = 0.0000  Cohen's d = -0.488

*Keyword appraisal meaningfully changes retrieval vs preset affect. Appraisal inference is not neutral.*
