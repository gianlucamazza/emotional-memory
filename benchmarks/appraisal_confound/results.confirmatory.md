# Appraisal Confound Study

Dataset: realistic_recall_v1 v1.4  (50 scenarios, 100 queries)  
Embedder: `sbert-bge`

## System Results

| System | N | top1_acc | 95% CI |
|---|---:|---:|---|
| `aft_noAppraisal` | 100 | 0.780 | [0.700, 0.860] |
| `aft_keyword` | 100 | 0.160 | [0.090, 0.230] |
| `naive_cosine` | 100 | 0.550 | [0.460, 0.650] |

## Hypothesis Tests

### Hd1 — ✓ PASS

**aft_noAppraisal.top1 > naive_cosine.top1, Δ > 0.10 (Addendum D confirmatory)**

Δ = 0.23 [0.14, 0.32]  Cohen's d = 0.515

*AFT architecture (no appraisal, preset affect) reliably outperforms naive cosine.*

### Ha2 — ✗ FAIL

**aft_keyword.top1 > naive_cosine.top1 (Δ > 0, CI excludes 0)**

Δ = -0.39 [-0.50, -0.29]  Cohen's d = -0.736

*No significant advantage of AFT+keyword over naive cosine. Architecture benefit not confirmed on this dataset.*

### Hb2 — ✗ FAIL

**|aft_keyword - aft_noAppraisal| < 0.05 (equivalence)**

Δ = -0.62 [-0.72, -0.52]  Cohen's d = -1.271

*Keyword appraisal meaningfully changes retrieval vs preset affect. Appraisal inference is not neutral.*
