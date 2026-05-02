# Pre-registration Addendum H — Multilingual Cross-Embedder Analysis (Italian)

**Date written:** 2026-05-02
**Protocol version:** addendum_h_v1
**Parent pre-reg:** `benchmarks/preregistration_addendum_v2.md` (Addendum B — multilingual slice)

> **Epistemic status:** This is a **post-hoc analysis document**, not a strict
> pre-registration. The data files `results.v2_it.sbert.json` and
> `results.v2_it.me5.json` were committed prior to this write-up. This document
> formalises the protocol and statistical tests that were applied to those runs,
> records the embedder-comparison hypothesis explicitly, and provides an
> audit-trail for the cross-embedder consistency claim. All numbers reported here
> match the committed result files exactly.

---

## Background

Addendum B (pre-reg v2) scheduled a multilingual slice of the realistic replay
benchmark (`realistic_recall_v2_it`) using the Italian corpus. The primary goal
was to verify that AFT provides a ranking advantage over `naive_cosine` in a
non-English language.

The Italian SBERT slice (embedder: `all-MiniLM-L6-v2`, EN-centric) was committed
in v0.7.0 (commit `2d8b397`) and showed:

- AFT advantage on `hit@k` is statistically significant (Δ=+0.15, p=0.0005)
- `top1_accuracy` advantage is not significant (Δ=+0.09, p=0.074)
- Absolute `top1` (0.24) is much lower than English v2 (0.70) — attributable
  to the EN-centric backbone, which cannot leverage Italian semantics effectively

A subsequent commit ran the same benchmark with `intfloat/multilingual-e5-small`
(commit `242143d`), a 100-language model, to test whether the AFT advantage is
robust to the choice of embedder. The results (`results.v2_it.me5.json`) are the
subject of this addendum.

---

## Research question

**Does the AFT retrieval advantage observed in the Italian SBERT slice persist
when the embedder is replaced with a multilingual model (`multilingual-e5-small`)?**

Sub-questions:
1. Does the AFT vs `naive_cosine` Δ on `hit@k` remain positive and statistically
   significant under `multilingual-e5-small`?
2. Does the AFT vs `naive_cosine` Δ on `top1_accuracy` remain statistically
   non-significant (as in SBERT)?
3. Do both systems improve in absolute accuracy with `multilingual-e5-small`
   compared to SBERT (consistent with the hypothesis that SBERT is the bottleneck)?

---

## Hypothesis formalised

**Ha1 (cross-embedder robustness, hit@k):** The AFT advantage on `hit@k` in the
Italian corpus is not an artefact of the EN-centric SBERT backbone. It persists
under `multilingual-e5-small`.

**Ha2 (top1 pattern persistence):** `top1_accuracy` advantage remains not
significant under `multilingual-e5-small` (consistent with the SBERT result).

**Ha3 (absolute lift):** Both `aft` and `naive_cosine` show higher absolute
`top1_accuracy` and `hit@k` under `multilingual-e5-small` than under SBERT.

---

## Dataset and protocol

| Field | Value |
|---|---|
| Dataset | `realistic_recall_v2_it.json` |
| Scenarios | 20 Italian scripted multi-session scenarios |
| Total queries | N = 80 (4 per scenario) |
| Top-K | k = 2 |
| Minimum candidate pool | 7 per query |
| Non-trivial queries | 69 / 80 (86.25%) |
| Challenge types | affective_arc, momentum_alignment, recency_confound, same_topic_distractor, semantic_confound (16 each) |
| Systems | `aft`, `naive_cosine`, `recency` |
| Embedder A (SBERT baseline) | `all-MiniLM-L6-v2` (EN-centric) |
| Embedder B (this addendum) | `intfloat/multilingual-e5-small` (100+ langs) |
| Bootstrap | n = 2000 paired, seed = 0, percentile CI |
| Significance threshold | p < 0.05 (two-tailed); p < 0.05 one-tailed for Ha1 |

---

## Results

### Headline table

| Embedder | System | top1 [95% CI] | hit@k [95% CI] |
|---|---|---|---|
| SBERT all-MiniLM-L6-v2 (EN-centric) | `aft` | 0.24 [0.15, 0.34] | 0.34 [0.24, 0.44] |
| SBERT all-MiniLM-L6-v2 | `naive_cosine` | 0.15 [0.07, 0.24] | 0.19 [0.11, 0.28] |
| multilingual-e5-small | `aft` | **0.29** [0.20, 0.39] | **0.42** [0.31, 0.54] |
| multilingual-e5-small | `naive_cosine` | 0.21 [0.12, 0.30] | 0.26 [0.17, 0.36] |

### Pairwise comparisons (AFT vs naive_cosine)

| Embedder | Metric | Δ | 95% CI | p (bootstrap) | p (McNemar) | d |
|---|---|---|---|---|---|---|
| SBERT | top1 | +0.09 | [0.00, 0.18] | 0.074 | 0.092 | 0.22 |
| SBERT | hit@k | **+0.15** | **[0.08, 0.24]** | **0.0005** | **0.0005** | 0.41 |
| me5-small | top1 | +0.08 | [-0.02, 0.18] | 0.154 | 0.210 | 0.17 |
| me5-small | hit@k | **+0.16** | **[0.06, 0.26]** | **0.001** | **0.004** | 0.35 |

---

## Hypothesis verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Ha1 — hit@k advantage persists cross-embedder | **PASS** | me5 hit@k Δ=+0.16, p=0.001; CI fully above 0 |
| Ha2 — top1 remains NS cross-embedder | **PASS** | me5 top1 Δ=+0.08, p=0.154; CI crosses 0 |
| Ha3 — absolute lift with multilingual embedder | **PASS** | aft: top1 0.24→0.29, hit@k 0.34→0.42; naive: top1 0.15→0.21, hit@k 0.19→0.26 |

---

## Interpretation

The AFT ranking advantage on `hit@k` is not an artefact of the EN-centric SBERT
backbone. The delta (≈+0.15–0.16 across both embedders) is strikingly consistent,
and both p-values are below the significance threshold.

The absolute accuracy of both systems improves when the multilingual embedder is
used, confirming that the low Italian SBERT numbers reflect embedder limitations
rather than AFT design failures. The Δ remains stable, which means the affective
layer adds value over and above what the backbone provides.

`top1_accuracy` remains non-significant under both embedders. This is consistent
with the hypothesis that the AFT advantage is primarily expressed in the top-2
recall window (hit@k) rather than strict top-1 precision.

### What this study cannot show

- Whether the cross-embedder robustness holds for non-Italian non-English languages.
- Whether an LLM-backed appraisal engine would change the picture (keyword
  Italian rules are limited; Addendum G direction for future work).
- Whether the AFT advantage scales with larger/more capable multilingual models
  (e.g., `multilingual-e5-large`, `BGE-M3`).

---

## Source files

| File | Description |
|---|---|
| `benchmarks/realistic/results.v2_it.sbert.json` | SBERT raw results (N=80) |
| `benchmarks/realistic/results.v2_it.sbert.md` | SBERT summary report |
| `benchmarks/realistic/results.v2_it.me5.json` | me5 raw results (N=80) |
| `benchmarks/realistic/results.v2_it.me5.md` | me5 summary report |
| `benchmarks/datasets/realistic_recall_v2_it.json` | Italian dataset |
| `benchmarks/realistic/runner.py` | Benchmark runner (`--embedder multilingual-e5-small`) |
