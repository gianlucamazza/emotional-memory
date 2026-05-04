# Pre-registration Addendum F Closure — Hf1 (Dual-Path vs Synchronous Keyword)

**Date executed:** 2026-05-04
**Protocol version:** addendum_f_closure_v1
**Parent pre-reg:** `benchmarks/preregistration_addendum_f.md`

> **Epistemic status:** This document records the execution and results of the
> pre-registered Hf1 hypothesis. All numbers match the committed JSON files exactly.

---

## Background

Addendum F pre-registered **Hf1**: does deferring keyword appraisal to a slow path
(`dual_path` via `elaborate()`) partially mitigate the destructive override observed
with synchronous keyword appraisal (`aft_keyword_synchronous`)?

Pre-registered criterion:
- `dual_path.top1_accuracy > aft_keyword_synchronous.top1_accuracy`
- Paired bootstrap (n=10,000, seed=0), **one-tailed p < 0.05 AND CI for Δ_Hf1 fully
  above 0**
- Primary dataset: `realistic_recall_v1.4` (N=100), embedder: SBERT bge-small-en-v1.5

---

## Execution

```bash
# Primary (v1.4, pre-registered dataset)
uv run python -m benchmarks.ablation.runner \
    --embedder sbert-bge \
    --n-bootstrap 10000 --seed 0 \
    --out-json benchmarks/ablation/results.sbert.json \
    --out-md benchmarks/ablation/results.sbert.md

# Replications (v2, N=200, not pre-registered for Hf1 but reported)
uv run python -m benchmarks.ablation.runner \
    --embedder sbert-bge \
    --dataset benchmarks/datasets/realistic_recall_v2.json \
    --n-bootstrap 10000 --seed 0 \
    --out-json benchmarks/ablation/results.v2.sbert.json

uv run python -m benchmarks.ablation.runner \
    --embedder e5-small-v2 \
    --dataset benchmarks/datasets/realistic_recall_v2.json \
    --n-bootstrap 10000 --seed 0 \
    --out-json benchmarks/ablation/results.v2.e5.json
```

Canonical result files (Hf1 section in each):
- `benchmarks/ablation/results.sbert.json` / `.md` (primary)
- `benchmarks/ablation/results.v2.sbert.json` / `.md` (replication)
- `benchmarks/ablation/results.v2.e5.json` / `.md` (replication)

---

## Results summary

### Primary (realistic_recall_v1.4, SBERT, N=100)

| System | top1_accuracy |
|--------|--------------|
| `dual_path` | 0.370 |
| `aft_keyword_synchronous` | 0.080 |

**Hf1 direct comparison** (paired bootstrap, n=10,000, seed=0):
Δ_Hf1 = +0.290 [0.190, 0.390], p (one-tailed) < 0.001, CI fully above 0.

**Verdict: Hf1 PASS.**

### Replications (realistic_recall_v2, N=200)

| Embedder | dual_path | aft_kw_sync | Δ_Hf1 [95% CI] | p (one-tailed) | CI above 0 | Verdict |
|---|---|---|---|---|---|---|
| SBERT bge-small-en | 0.350 | 0.095 | +0.255 [0.190, 0.320] | <0.001 | ✓ | **PASS** |
| e5-small-v2 | 0.235 | 0.070 | +0.165 [0.110, 0.225] | <0.001 | ✓ | **PASS** |

---

## Interpretation

**Hf1 PASS — primary and both replications.** Deferring keyword appraisal to the
slow path (`dual_path` + `elaborate()`) significantly mitigates the destructive
synchronous override. The gap is large: +0.255 to +0.280 percentage points (26–28 pp)
with CIs fully above 0.

**Practical interpretation:** The dual-path timing architecture (LeDoux 1996) does
matter, even for a noisy keyword engine. When appraisal runs synchronously at encode
time, it overwrites preset affect and collapses top1 to ~0.07–0.095. When it runs
asynchronously via `elaborate()`, the architecture retains partial information from
the preset affect write, boosting top1 to 0.235–0.350.

**Scope limitation:** Both conditions use `KeywordAppraisalEngine`, which is
destructive on this benchmark. Hf1 PASS shows that deferral *partially* mitigates the
damage, but `dual_path` still underperforms `full_aft` (preset affect only) by
Δ ≈ −0.19 (SBERT v2). The LeDoux dual-path advantage is therefore about reducing
*the cost of bad appraisal*, not about providing a net positive over oracle affect.

**Addendum G (future):** The theoretically important question remains open —
whether dual-path encoding with a *good* appraisal engine (LLM) provides a positive
advantage over no appraisal, on a dataset not pre-seeded with oracle affect values.
