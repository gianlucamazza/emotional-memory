# Appraisal Confound Study

Dataset: realistic_recall_v2_es v1.0  (20 scenarios, 80 queries)  
Embedder: `multilingual-e5-small`  n_bootstrap: 10000  seed: 42

## System Results

| System | N | top1_acc | 95% CI |
|---|---:|---:|---|
| `aft_noAppraisal` | 80 | 0.325 | [0.225, 0.425] |
| `aft_keyword` | 80 | 0.013 | [0.000, 0.037] |
| `naive_cosine` | 80 | 0.212 | [0.125, 0.300] |

## Hypothesis Tests

### Hd2_ES — ✗ FAIL

**aft_noAppraisal.top1 > naive_cosine.top1, Δ > 0.10 (Addendum D generalization)**

Δ = 0.11 [-0.01, 0.24]  p_two_sided = 0.1095  Cohen's d = 0.189

*AFT without appraisal does not show a practically significant advantage over naive cosine at Δ > 0.10 threshold.*

### Ha2 — ✗ FAIL

**aft_keyword.top1 > naive_cosine.top1 (Δ > 0.05, one-tailed alpha=0.05)**

Δ = -0.20 [-0.30, -0.11]  p_one_sided = 0.0001  Cohen's d = -0.462

*No significant advantage of AFT+keyword over naive cosine. Architecture benefit not confirmed on this dataset.*

### Hb2 — ✗ FAIL

**|aft_keyword - aft_noAppraisal| < 0.05 (equivalence)**

Δ = -0.31 [-0.41, -0.21]  p_two_sided = 0.0000  Cohen's d = -0.670

*Keyword appraisal meaningfully changes retrieval vs preset affect. Appraisal inference is not neutral.*
