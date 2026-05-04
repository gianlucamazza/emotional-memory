# Appraisal Confound Study

Dataset: realistic_recall_v2 v2.0  (50 scenarios, 200 queries)  
Embedder: `sbert-bge`  n_bootstrap: 10000  seed: 42

## System Results

| System | N | top1_acc | 95% CI |
|---|---:|---:|---|
| `aft_noAppraisal` | 200 | 0.595 | [0.525, 0.660] |
| `aft_keyword` | 200 | 0.215 | [0.160, 0.275] |
| `naive_cosine` | 200 | 0.470 | [0.400, 0.540] |

## Hypothesis Tests

### Hd2 — ✓ PASS

**aft_noAppraisal.top1 > naive_cosine.top1, Δ > 0.10 (Addendum D generalization)**

Δ = 0.12 [0.07, 0.18]  p_two_sided = 0.0000  Cohen's d = 0.286

*AFT architecture (no appraisal, preset affect) reliably outperforms naive cosine.*

### Ha2 — ✗ FAIL

**aft_keyword.top1 > naive_cosine.top1 (Δ > 0.05, one-tailed alpha=0.05)**

Δ = -0.26 [-0.33, -0.18]  p_one_sided = 0.0000  Cohen's d = -0.472

*No significant advantage of AFT+keyword over naive cosine. Architecture benefit not confirmed on this dataset.*

### Hb2 — ✗ FAIL

**|aft_keyword - aft_noAppraisal| < 0.05 (equivalence)**

Δ = -0.38 [-0.46, -0.30]  p_two_sided = 0.0000  Cohen's d = -0.654

*Keyword appraisal meaningfully changes retrieval vs preset affect. Appraisal inference is not neutral.*
