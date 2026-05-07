# Appraisal Confound Study

Dataset: realistic_recall_v2_es v1.0  (30 scenarios, 120 queries)  
Embedder: `multilingual-e5-small`  n_bootstrap: 10000  seed: 42

## System Results

| System | N | top1_acc | 95% CI |
|---|---:|---:|---|
| `aft_noAppraisal` | 120 | 0.267 | [0.192, 0.350] |
| `aft_keyword` | 120 | 0.083 | [0.033, 0.133] |
| `naive_cosine` | 120 | 0.267 | [0.192, 0.350] |

## Hypothesis Tests

### Hd2_ES — ✗ FAIL

**aft_noAppraisal.top1 > naive_cosine.top1, Δ > 0.10 (Addendum D generalization)**

Δ = 0.00 [-0.10, 0.10]  p_two_sided = 1.0000  Cohen's d = 0.000

*AFT without appraisal does not show a practically significant advantage over naive cosine at Δ > 0.10 threshold.*

### Ha2 — ✗ FAIL

**aft_keyword.top1 > naive_cosine.top1 (Δ > 0.05, one-tailed alpha=0.05)**

Δ = -0.18 [-0.28, -0.09]  p_one_sided = 0.0000  Cohen's d = -0.365

*No significant advantage of AFT+keyword over naive cosine. Architecture benefit not confirmed on this dataset.*

### Hb2 — ✗ FAIL

**|aft_keyword - aft_noAppraisal| < 0.05 (equivalence)**

Δ = -0.18 [-0.28, -0.09]  p_two_sided = 0.0003  Cohen's d = -0.343

*Keyword appraisal meaningfully changes retrieval vs preset affect. Appraisal inference is not neutral.*
