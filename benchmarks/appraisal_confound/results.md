# Appraisal Confound Study

Dataset: realistic_recall_v1 v1.4  (50 scenarios, 100 queries)  
Embedder: `sbert-bge`  n_bootstrap: 10000  seed: 42

## System Results

| System | N | top1_acc | 95% CI |
|---|---:|---:|---|
| `aft_noAppraisal` | 100 | 0.780 | [0.700, 0.860] |
| `aft_keyword` | 100 | 0.160 | [0.090, 0.240] |
| `naive_cosine` | 100 | 0.550 | [0.450, 0.650] |

## Hypothesis Tests

### Ha2 — ✗ FAIL

**aft_keyword.top1 > naive_cosine.top1 (Δ > 0.05, one-tailed α=0.05)**

Δ = -0.39 [-0.49, -0.29]  p_one_sided = 0.0000  Cohen's d = -0.736

*No significant advantage of AFT+keyword over naive cosine. Architecture benefit not confirmed on this dataset.*

### Hb2 — ✗ FAIL

**|aft_keyword - aft_noAppraisal| < 0.05 (equivalence)**

Δ = -0.62 [-0.71, -0.52]  p_two_sided = 0.0000  Cohen's d = -1.271

*Keyword appraisal meaningfully changes retrieval vs preset affect. Appraisal inference is not neutral.*
