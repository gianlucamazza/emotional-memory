# Appraisal Confound Study

Dataset: realistic_recall_v2_es v1.0  (20 scenarios, 80 queries)  
Embedder: `sbert-bge`  n_bootstrap: 10000  seed: 42

## System Results

| System | N | top1_acc | 95% CI |
|---|---:|---:|---|
| `aft_noAppraisal` | 80 | 0.362 | [0.263, 0.475] |
| `aft_keyword` | 80 | 0.037 | [0.000, 0.087] |
| `naive_cosine` | 80 | 0.225 | [0.138, 0.325] |

## Hypothesis Tests

### Hd2_ES — ✓ PASS

**aft_noAppraisal.top1 > naive_cosine.top1, Δ > 0.10 (Addendum D generalization)**

Δ = 0.14 [0.01, 0.26]  p_two_sided = 0.0450  Cohen's d = 0.233

*AFT architecture (no appraisal, preset affect) reliably outperforms naive cosine.*

### Ha2 — ✗ FAIL

**aft_keyword.top1 > naive_cosine.top1 (Δ > 0.05, one-tailed alpha=0.05)**

Δ = -0.19 [-0.29, -0.10]  p_one_sided = 0.0001  Cohen's d = -0.442

*No significant advantage of AFT+keyword over naive cosine. Architecture benefit not confirmed on this dataset.*

### Hb2 — ✗ FAIL

**|aft_keyword - aft_noAppraisal| < 0.05 (equivalence)**

Δ = -0.33 [-0.44, -0.21]  p_two_sided = 0.0000  Cohen's d = -0.622

*Keyword appraisal meaningfully changes retrieval vs preset affect. Appraisal inference is not neutral.*
