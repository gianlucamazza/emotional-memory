# Addendum G — Dual-Path LLM Appraisal (Hg1)

Dataset: `realistic_recall_v4_noAF` v1.0.0  (40 scenarios, 160 queries)
Embedder: `sbert-bge`  n_bootstrap: 10000  seed: 0

> **Note:** All memory events were encoded without preset valence/arousal. Affect signal comes exclusively from the LLM appraisal engine.

## System Results

| System | N | top1_acc | 95% CI |
|---|---:|---:|---|
| `aft_llm_dual` | 160 | 0.800 | [0.738, 0.863] |
| `naive_cosine` | 160 | 0.887 | [0.838, 0.931] |
| `aft_neutral` | 160 | 0.744 | [0.675, 0.812] |
| `aft_llm_sync` | 160 | 0.287 | [0.219, 0.362] |

## Hypothesis Tests

### Hg1 (confirmatory) — ✗ FAIL

**aft_llm_dual.top1 > naive_cosine.top1 (Δ > 0.05, one-tailed alpha=0.05)**

Δ = -0.09 [-0.14, -0.03]  p_one_sided = 0.0018  Cohen's d = -0.242

*No significant advantage of aft_llm_dual over naive cosine at Δ>0.05. LLM appraisal does not provide net positive on this dataset at this effect size.*

### Hg2 (exploratory) — ✓ PASS

**aft_llm_dual.top1 > aft_neutral.top1 (exploratory)**

Δ = 0.06 [0.00, 0.11]  p_one_sided = 0.0295  Cohen's d = 0.157

*LLM-inferred affect outperforms neutral baseline — appraisal adds signal.*

### Hg3 (exploratory) — ✓ PASS

**aft_llm_dual.top1 > aft_llm_sync.top1 (exploratory)**

Δ = 0.51 [0.42, 0.59]  p_one_sided = 0.0000  Cohen's d = 0.953

*Dual-path (deferred) LLM appraisal outperforms synchronous LLM appraisal.*
