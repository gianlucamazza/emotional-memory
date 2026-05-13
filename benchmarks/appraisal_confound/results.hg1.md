# Addendum G — Dual-Path LLM Appraisal (Hg1)

Dataset: `realistic_recall_v3_noAF` v1.0.0  (50 scenarios, 200 queries)
Embedder: `sbert-bge`  n_bootstrap: 10000  seed: 0

> **Note:** All memory events were encoded without preset valence/arousal. Affect signal comes exclusively from the LLM appraisal engine.

## System Results

| System | N | top1_acc | 95% CI |
|---|---:|---:|---|
| `aft_llm_dual` | 200 | 0.315 | [0.250, 0.380] |
| `naive_cosine` | 200 | 0.325 | [0.260, 0.390] |
| `aft_neutral` | 200 | 0.315 | [0.250, 0.380] |
| `aft_llm_sync` | 200 | 0.130 | [0.085, 0.180] |

## Hypothesis Tests

### Hg1 (confirmatory) — ✗ FAIL

**aft_llm_dual.top1 > naive_cosine.top1 (Δ > 0.05, one-tailed alpha=0.05)**

Δ = -0.01 [-0.06, 0.04]  p_one_sided = 0.3669  Cohen's d = -0.032

*No significant advantage of aft_llm_dual over naive cosine at Δ>0.05. LLM appraisal does not provide net positive on this dataset at this effect size.*

### Hg2 (exploratory) — ✗ FAIL

**aft_llm_dual.top1 > aft_neutral.top1 (exploratory)**

Δ = 0.00 [-0.06, 0.06]  p_one_sided = 0.5000  Cohen's d = 0.000

*LLM appraisal does not reliably outperform neutral affect baseline.*

### Hg3 (exploratory) — ✓ PASS

**aft_llm_dual.top1 > aft_llm_sync.top1 (exploratory)**

Δ = 0.18 [0.11, 0.26]  p_one_sided = 0.0000  Cohen's d = 0.342

*Dual-path (deferred) LLM appraisal outperforms synchronous LLM appraisal.*
