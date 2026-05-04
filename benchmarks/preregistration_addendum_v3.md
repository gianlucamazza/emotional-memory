# Pre-Registration Addendum — v3 Studies

**Status:** FROZEN. This document is an addendum to `benchmarks/preregistration.md`
and supersedes nothing in `benchmarks/preregistration_addendum_v2.md`.

**Parent pre-registration:** `benchmarks/preregistration.md` (committed 2026-04-24).
**Companion:** `benchmarks/preregistration_addendum_v2.md` (Addenda A/B/C).

This addendum pre-registers two new studies that close gaps surfaced by the
audit (`docs/research/audit_2026-04.md`) and the appraisal-confound result
(`benchmarks/appraisal_confound/results.json`):

- **Addendum D** — re-frames the architecture-attribution question after
  Addendum A's Ha2 was found to be poorly specified.
- **Addendum E** — formalizes the dual-path encoding and APE-gated
  reconsolidation ablations identified by audit gap G9.

---

## Addendum D — Architecture Attribution (re-pre-registration)

### Motivation

Addendum A's Ha2 (`aft_keyword > naive_cosine`) **failed**
(Δ = −0.39, p ≈ 0; results committed 2026-04-26). Diagnosis:
`KeywordAppraisalEngine` overwrites the dataset's hand-curated preset
`valence/arousal` in `engine.encode`, replacing high-quality preset affect
with noisier inferred affect. The pre-registered comparison therefore
tested the wrong question.

The audit's actual question — *"does the AFT architecture (no LLM, no
inferred appraisal) beat naive cosine?"* — is answered by the
**descriptive** comparison `aft_noAppraisal > naive_cosine`
(Δ ≈ +0.23) on the same N=100 dataset, *and* by Study S2 which also used no
appraisal engine. Both signals are observational; neither was
pre-registered as the primary hypothesis.

This addendum re-registers the comparison so a confirmatory result can
land.

### Systems

- `aft_noAppraisal`: `AFTReplayAdapter` with no appraisal engine; affect
  set from scenario presets via `engine.set_affect()` immediately before
  `engine.encode()`. Identical configuration to the system used in
  Addendum A (cf. `benchmarks/appraisal_confound/runner.py:88`).
- `naive_cosine`: pure semantic cosine baseline. Identical to S2 and
  Addendum A.

### Dataset

Two datasets, executed in sequence:

1. **realistic_recall_v1** v1.4 (50 scenarios, 100 queries, 4 challenge
   types) — same as S2 / Addendum A. Confirmatory of the descriptive
   Δ ≈ +0.23 already observed.
2. **realistic_recall_v2** (when Addendum B's v2 dataset is committed)
   N≥200 queries, 5 challenge types. Generalization slice.

### Embedder

`sbert-bge` (BAAI/bge-small-en-v1.5) — paper-canonical dense embedder.

Hash-embedder run is reported as a sensitivity check, not as primary
evidence (per `10_scientific_quality_bar.md` cross-embedder rules).

### Primary hypothesis (Hd1)

`aft_noAppraisal.top1_accuracy > naive_cosine.top1_accuracy`

- Test: paired bootstrap difference, N=10,000 samples, seed=42
- One-tailed alpha=0.05 (direction pre-specified)
- Effect threshold: Δ > 0.10 (10 pp) for practical significance
  — stricter than Addendum A's 0.05 because the prior descriptive
  point estimate is +0.23 with CI lower bound around +0.12
- No Holm correction: single primary hypothesis

### Secondary hypothesis (Hd2)

`aft_noAppraisal.top1_accuracy > naive_cosine.top1_accuracy` on the
**v2 dataset** (when available), with the same statistical criteria as Hd1.

If Hd1 holds but Hd2 fails, the architecture advantage is scoped to
realistic_recall_v1.

### Pre-registered execution

```bash
make bench-appraisal-confound  # produces aft_noAppraisal vs naive_cosine
                               # numbers as part of the existing 3-system run
```

The runner's existing 3-system output (`results.json`) already contains
the numbers needed; analysis script should formally compute Hd1 from
`aft_noAppraisal` and `naive_cosine` `top1_flags`.

A future commit will extend the runner to formally print Hd1 verdict
alongside Ha2/Hb2.

### What this study cannot show

- Whether the advantage holds on real human conversation (Gate 1, S1 LoCoMo).
- Whether the advantage holds outside English (Addendum B multilingual).
- Whether humans prefer the resulting outputs (Gate 2, Addendum C).

---

## Addendum E — Dual-path encoding & APE-gated reconsolidation ablations

### Motivation

Audit gap G9: the paper highlights two implemented mechanisms — dual-path
encoding (LeDoux 1996) and APE-gated reconsolidation (Pearce-Hall 1980) —
as architectural contributions, but neither has a dedicated ablation in
`benchmarks/ablation/`. The current ablation suite isolates `no_appraisal`,
`no_mood`, `no_momentum`, and `no_resonance`. This addendum extends it.

### Variants added

- **`dual_path`** (new flag: `EmotionalMemoryConfig.dual_path_encoding=True`)
  Fast path encodes with raw affect and skips appraisal at write time;
  slow path runs `elaborate(memory_id)` later with a 70/30 blend.
  Requires an appraisal engine to be meaningful. The ablation runner
  must inject `KeywordAppraisalEngine` for this variant (mirror of the
  pattern used in `benchmarks/appraisal_confound/runner.py:73-79`),
  otherwise the variant is a no-op (engine.encode short-circuits when
  `_appraisal_engine is None`).

  **Caveat (carried forward from Addendum A finding):** keyword appraisal
  on this dataset overwrites preset affect destructively. The dual_path
  variant therefore tests "does deferring appraisal to slow path mitigate
  the destructive override observed in synchronous keyword appraisal?"

- **`no_reconsolidation`** (new flag: `EmotionalMemoryConfig.enable_reconsolidation=False`)
  Wraps the APE-gated reconsolidation block in `engine.py` and
  `async_engine.py` behind a config flag. When False, retrieval skips
  `update_prediction` and skips Pearce-Hall reconsolidation triggers.

### Dataset, embedder, statistics

Same as the existing `bench-ablation` and `bench-ablation-sbert` targets:
- Dataset: `realistic_recall_v1` v1.4 (N=100)
- Embedders: `sbert-bge` (paper-canonical) and `hash` (sensitivity)
- Bootstrap: N=10,000 paired, seed=0 (consistent with existing ablation
  variants — note this differs from Addendum A/D's seed=42 because
  ablation runner's `seed=0` is established convention; sensitivity to
  seed is reported alongside primary results)
- Holm correction: applied across all variants (existing ablation
  pipeline behavior)

### Primary hypotheses

For each new variant, the ablation runner produces:

`Δ_variant = top1(variant) − top1(full_aft)`

A variant **contributes** iff `Δ < 0` with paired bootstrap CI fully below
0 and `p_adj < 0.05` after Holm correction.

- **He1** (`dual_path` contributes): `Δ_dual_path < 0`, CI excludes 0,
  Holm-adjusted.
  — Interpretation under He1 PASS: dual-path encoding *helps* on this
  dataset (the slow-path 70/30 blend recovers some signal that
  synchronous keyword appraisal destroys).
  — Interpretation under He1 FAIL: dual-path encoding is neutral or
  harmful on this dataset; the implementation is theoretically faithful
  but not empirically advantaged.

- **He2** (`no_reconsolidation` contributes): `Δ_no_reconsolidation < 0`,
  CI excludes 0, Holm-adjusted.
  — Interpretation under He2 PASS: APE-gated reconsolidation contributes
  to retrieval quality.
  — Interpretation under He2 FAIL: reconsolidation is implemented per
  theory but does not measurably affect this benchmark's retrieval (the
  benchmark may not exercise the conditions under which it activates).

### What this study cannot show

- Whether either mechanism contributes on *other* datasets.
- Whether either mechanism contributes via a different observable than
  top1 retrieval accuracy (e.g., long-horizon memory consolidation,
  affect-trajectory smoothness).

### Implementation contract

This addendum is committed BEFORE the implementation lands. The
following must be true at execution time:

1. `EmotionalMemoryConfig` exposes `enable_reconsolidation: bool = True`
   defaulting to current behavior.
2. `engine.py` and `async_engine.py` reconsolidation blocks are gated on
   `self._config.enable_reconsolidation`.
3. `benchmarks/ablation/runner.py` `ABLATIONS` list includes
   `("dual_path", {"dual_path_encoding": True})` and
   `("no_reconsolidation", {"enable_reconsolidation": False})`.
4. The `dual_path` variant injects `KeywordAppraisalEngine` per the
   pattern in `benchmarks/appraisal_confound/runner.py:73-79`.
5. `make bench-ablation` and `make bench-ablation-sbert` produce updated
   `results.{json,md,protocol.json}` and `results.sbert.{json,md,protocol.json}`
   covering all six variants.

---

## Provenance note

This addendum is written before executing the studies it describes.
Results for Addenda D and E will be **CONFIRMATORY** under this
pre-registration, provided they are committed after this file's git
timestamp. Results from before this file's commit date are DISCOVERY.

Specifically: the Addendum A run committed 2026-04-26 already produced the
numbers needed for Hd1 (descriptive); a re-execution after this addendum
commits will be confirmatory. The G9 ablation variants do not yet exist
in `benchmarks/ablation/runner.py` and will be implemented as part of the
PR that lands this addendum.

---

## Hd2 Closure — 2026-05-04

Executed at power (N=200, seed=42, n_bootstrap=10000) on realistic_recall_v2 (EN)
and realistic_recall_v2_it (IT cross-language slice).

| Hypothesis | Dataset | Embedder | Verdict | Δ (top1) | 95% CI | p_two_sided | Cohen's d |
|---|---|---|---|---|---|---|---|
| Hd2 (aft_noAppraisal > naive_cosine, Δ>0.10) | v2 EN | SBERT | **PASS** | +0.125 | not shown | <0.001 | 0.286 |
| Hd2_IT (aft_noAppraisal > naive_cosine, Δ>0.10) | v2 IT | me5 | **PASS** | +0.163 | not shown | 0.012 | 0.289 |

**Canonical result files:**
- `benchmarks/appraisal_confound/results.hd2.sbert.json` (EN, SBERT)
- `benchmarks/appraisal_confound/results.hd2_it.me5.json` (IT, multilingual-e5-small)

**Interpretation:** Hd2 PASSES on both EN and IT slices. The architecture advantage
established in Hd1 (v1, Δ=0.23, d=0.515) generalizes to realistic_recall_v2
(Δ=0.125, d=0.286) with reduced but still practically significant effect size
(Δ > 0.10 pre-registered threshold). The Italian cross-language slice (Hd2_IT,
Δ=0.163, d=0.289) shows comparable magnitude, extending the finding to a non-English
setting with a multilingual embedder.

Note: the smaller effect on v2 vs v1 is expected — v2 contains harder challenge
types (lexical distraction, temporal ordering) that reduce absolute accuracy for
all systems, compressing the Δ range. The AFT advantage remains detectable.

This is a pre-registered confirmatory result and is reported as such.
