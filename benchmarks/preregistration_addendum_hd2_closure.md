# Pre-registration Addendum — Hd2 Closure (Addendum D Generalization)

**Date executed:** 2026-05-04
**Protocol version:** addendum_hd2_closure_v1
**Parent pre-reg:** `benchmarks/preregistration_addendum_v3.md` (Hd2, lines 78–84)

> **Epistemic status:** This document records the execution and results of the
> pre-registered Hd2 (generalization) and Hd2_IT (cross-language secondary slice).
> The canonical result files were written at execution time; this document provides
> the interpretive closure. All numbers match the committed JSON files exactly.

---

## Background

Addendum D (pre-reg v3) pre-registered **Hd1** (primary, on realistic_recall_v1) and
**Hd2** (generalization, on realistic_recall_v2). Hd1 was confirmed (PASS, Δ=0.23,
d=0.515, commit `benchmarks/appraisal_confound/results.confirmatory.json`).

Hd2 tests whether the same architecture advantage (`aft_noAppraisal > naive_cosine`,
Δ > 0.10) holds on the harder v2 dataset. Per the pre-reg: "if Hd1 holds but Hd2
fails, the architecture advantage is scoped to realistic_recall_v1."

The secondary Hd2_IT slice (Italian, multilingual-e5-small) was not formally
pre-registered but is a direct extension consistent with Addendum B+H intent.

---

## Execution

```bash
make bench-hd2-sbert      # realistic_recall_v2, SBERT
make bench-hd2-it-me5     # realistic_recall_v2_it, multilingual-e5-small (secondary)
```

Canonical result files:
- `benchmarks/appraisal_confound/results.hd2.sbert.json` (EN, primary)
- `benchmarks/appraisal_confound/results.hd2_it.me5.json` (IT, secondary)

Parameters: seed=42, n_bootstrap=10000, one-tailed α=0.05, Δ > 0.10 threshold.
Per-study N: Hd2 EN N=200 (50 scenarios × 4 queries); Hd2_IT N=80 (20 scenarios × 4 queries).

---

## Results summary

| Study | Dataset | Embedder | Verdict | Δ (top1) | p_two_sided | Cohen's d |
|---|---|---|---|---|---|---|
| Hd1 (primary, v1) | realistic_recall_v1 | SBERT | **PASS** | +0.230 | <0.001 | 0.515 |
| Hd2 (generalization, v2) | realistic_recall_v2 | SBERT | **PASS** | +0.125 | <0.001 | 0.286 |
| Hd2_IT (cross-language, v2_it) | realistic_recall_v2_it | me5 | **PASS** | +0.163 | 0.012 | 0.289 |

---

## Interpretation

**Hd2 PASS.** The architecture advantage established in Hd1 generalizes to
realistic_recall_v2. The effect size is smaller (Δ=0.125 vs 0.230) — expected
given v2's harder challenge types (lexical distraction, temporal ordering) compress
absolute accuracy for all systems. The Δ > 0.10 pre-registered threshold is met.

**Hd2_IT PASS (secondary, cross-language).** With a multilingual embedder on the
Italian slice, the advantage is Δ=0.163 (d=0.289), comparable in magnitude to
EN. This extends the AFT architecture claim to non-English language + multilingual
embedder settings, consistent with Addendum H (hit@k significance on Italian).

**Coherence with S3 results.** S3 found Ha/Hb FAIL (mood, resonance individually
not significant). The Hd2 PASS establishes that the *system-level* advantage is
real; S3 shows it is not attributable to any single layer. Both findings are
consistent: AFT's advantage is an emergent system property, not the sum of
isolatable layer contributions.

**v1 vs v2 comparison.** The gap between Hd1 (Δ=0.23) and Hd2 (Δ=0.125) is not
a regression — it reflects the increased difficulty of the v2 benchmark. The
proportional advantage (AFT/naive ratio) is similar across both datasets.
