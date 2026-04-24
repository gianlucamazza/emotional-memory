# Pre-Registration Addendum — v2 Studies

**Status:** FROZEN. This document is an addendum to `benchmarks/preregistration.md`.
It pre-registers study designs that were not included in the original pre-registration.
Commit timestamp establishes precedence: any results produced from files committed
after this document's commit are CONFIRMATORY for the hypotheses stated here.

**Parent pre-registration:** `benchmarks/preregistration.md` (committed 2026-04-24).

---

## Addendum A — Appraisal Confound Study

### Motivation

The primary comparative result (AFT top1_accuracy > naive_cosine on
realistic_recall_v1.4, Δ=+0.20, DISCOVERY) may be partially explained by the
LLM appraisal prompt injecting richer affect signals rather than by the AFT
retrieval architecture itself. This study isolates the contribution.

### Systems

- `aft_noAppraisal`: AFT with manually preset affect from scenario data,
  no appraisal engine. Isolates the 6-signal retrieval architecture.
- `aft_keyword`: AFT + `KeywordAppraisalEngine` (deterministic, no LLM).
  Represents the minimum viable appraisal signal.
- `naive_cosine`: pure cosine baseline (same as Study S2).

No LLM-backed appraisal condition in this study — that comparison is deferred
to a future study to avoid LLM-as-judge confounds in retrieval evaluation.

### Dataset

`benchmarks/datasets/realistic_recall_v1.json` (v1.4: 50 scenarios, 100
queries, 4 challenge types). Same dataset as Study S2 for comparability.

### Primary hypothesis (Ha2)

`aft_keyword.top1_accuracy > naive_cosine.top1_accuracy`

- Test: paired bootstrap difference (N=10,000 samples, seed=42)
- α=0.05, one-tailed (AFT keyword > naive_cosine)
- Effect threshold: Δ > 0.05 (5 pp) for practical significance
- Holm correction not applied (single primary hypothesis in this study)

### Secondary hypothesis (Hb2)

`aft_keyword.top1_accuracy ≈ aft_noAppraisal.top1_accuracy`

- Test: equivalence test or paired bootstrap, CI within ±0.05
- If Hb2 holds (no difference): the preset affect and keyword appraisal are
  functionally equivalent on this dataset; architecture is the driver.
- If Hb2 fails (keyword < noAppraisal): preset affect helps more than
  keyword inference — or keyword appraisal introduces noise.

### Execution

```bash
make bench-appraisal-confound
```

Output: `benchmarks/appraisal_confound/results.{json,md}`.

No LLM API key required.

---

## Addendum B — Realistic Recall v2 (extended)

### Changes from Study S2 (preregistration.md)

The original S2 pre-registers N≥200 queries and 5 challenge types including
`momentum_alignment`. This addendum adds:

1. **Cross-embedder requirement**: results must be reported on two embedder classes:
   - Class A: dense (default `sbert-bge` = `bge-small-en-v1.5`)
   - Class B: a second dense embedder (e.g. `e5-small-v2` or `gtr-t5-base`)
   Hypotheses Ha–Hb from S2 must hold on *both* embedder classes for a strong claim.

2. **Multilingual slice**: ≥20 scenarios in Italian (primary) or Spanish.
   Reported separately; aggregate claims remain English-scoped until this
   slice shows equivalent performance.

3. **Failure case logging**: for each embedder class, scenarios where AFT
   underperforms naive_cosine by ≥0.1 (absolute top1_acc) are logged as
   failure cases and must appear in the paper's limitations section.

### Updated primary hypothesis (H3-v2)

Same as H3 in `benchmarks/preregistration.md` but replicated on both embedder
classes:

`aft.top1_accuracy > naive_cosine.top1_accuracy` on Class A AND Class B.

If the result holds only on Class A, the claim is scoped to dense-embedding
settings.

---

## Addendum C — Human Evaluation Pilot (updated criteria)

### Changes from preregistration.md §M1.2

1. **Minimum raters**: ≥5 (increased from ≥3). Three raters is insufficient
   for Krippendorff's α reliability when a disagreement occurs on a single item.

2. **Failure case field**: the rating template must include a `failure_mode`
   field (freetext, optional) for any item a rater scores 1 on any dimension.
   Items with `failure_mode` filled by ≥2 raters are flagged as failure cases.

3. **Publishability criterion**: Krippendorff's α ≥ 0.67 ordinal on the
   "relevance" dimension is required for the pilot to be reported as valid.
   Below this threshold, the pilot is exploratory only and cannot be cited as
   evidence.

4. **Conditions**: each item must include both AFT and baseline outputs
   (within-rater design, counterbalanced order). Raters are blind to which
   output is AFT and which is baseline.

---

## Provenance note

This addendum is written before executing any of the studies it describes.
Results for Addendum A, B, and C will be CONFIRMATORY under this pre-registration,
provided they are committed after this file's git timestamp.

Results from before this file's commit date are DISCOVERY.
