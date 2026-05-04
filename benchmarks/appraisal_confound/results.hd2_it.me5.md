# Appraisal Confound Study

Dataset: realistic_recall_v2_it v1.0  (20 scenarios, 80 queries)  
Embedder: `multilingual-e5-small`  n_bootstrap: 10000  seed: 42

## System Results

| System | N | top1_acc | 95% CI |
|---|---:|---:|---|
| `aft_noAppraisal` | 80 | 0.438 | [0.325, 0.550] |
| `aft_keyword` | 80 | 0.025 | [0.000, 0.062] |
| `naive_cosine` | 80 | 0.275 | [0.175, 0.375] |

## Hypothesis Tests

### Hd2_IT — ✓ PASS

**aft_noAppraisal.top1 > naive_cosine.top1, Δ > 0.10 (Addendum D generalization)**

Δ = 0.16 [0.04, 0.29]  p_two_sided = 0.0118  Cohen's d = 0.289

*AFT architecture (no appraisal, preset affect) reliably outperforms naive cosine.*

### Ha2 — ✗ FAIL

**aft_keyword.top1 > naive_cosine.top1 (Δ > 0.05, one-tailed alpha=0.05)**

Δ = -0.25 [-0.35, -0.15]  p_one_sided = 0.0000  Cohen's d = -0.539

*No significant advantage of AFT+keyword over naive cosine. Architecture benefit not confirmed on this dataset.*

### Hb2 — ✗ FAIL

**|aft_keyword - aft_noAppraisal| < 0.05 (equivalence)**

Δ = -0.41 [-0.53, -0.30]  p_two_sided = 0.0000  Cohen's d = -0.793

*Keyword appraisal meaningfully changes retrieval vs preset affect. Appraisal inference is not neutral.*
