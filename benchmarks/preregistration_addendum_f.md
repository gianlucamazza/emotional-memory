# Pre-registration Addendum F — Synchronous-Keyword Baseline for Dual-Path Ablation

**Date committed:** 2026-04-27
**Precedes:** First execution of `aft_keyword_synchronous` benchmark variant.
**Protocol version:** addendum_f_v1
**Parent pre-reg:** `benchmarks/preregistration_addendum_v3.md` (Addendum E, He1)

---

## Background

Addendum E (pre-reg v3) introduced the `dual_path` ablation variant (He1) and
documented a caveat: He1 compares *deferred keyword appraisal* against *no
appraisal* (full_aft uses preset affect with no appraisal engine). The result was
He1 FAIL (dual_path top1=0.35 vs full_aft top1=0.70, Δ=−0.35).

He1 FAIL is expected given that `KeywordAppraisalEngine` overwrites hand-curated
preset valence/arousal destructively on `realistic_recall_v1.4` (same finding as
G3/Addendum A). He1 therefore answers "does deferred bad appraisal help vs no
appraisal?" rather than the more theoretically interesting question.

**The theoretically interesting question** (LeDoux 1996 dual-path hypothesis):
*Does deferring appraisal to a slow path partially mitigate the destructive
override observed in synchronous keyword appraisal?*

Answering this requires a comparison between:
- `dual_path` (deferred keyword appraisal via `elaborate()`)
- `aft_keyword_synchronous` (synchronous keyword appraisal at encode time)

Both variants use `KeywordAppraisalEngine`; they differ only in *when* the
appraisal runs relative to the preset-affect write.

---

## New variant

**`aft_keyword_synchronous`**: `AFTReplayAdapter` with `KeywordAppraisalEngine`
injected at session initialisation, running synchronously at `encode()` time
(standard pipeline, no `elaborate()` call). Config: all AFT flags at default
(`dual_path_encoding=False`).

This variant corresponds to the `aft_keyword` condition from the
appraisal-confound study (G3/Addendum A), confirmed result there: top1=0.16.
Expected range on v1.4 ablation (N=100, seed=0): 0.10–0.20 (same destructive
override mechanism, slightly different call path).

---

## Pre-registered hypothesis

**Hf1 (confirmatory, one-tailed):**

```
dual_path.top1_accuracy > aft_keyword_synchronous.top1_accuracy
```

- Direction: `Δ_Hf1 = dual_path.top1 − aft_keyword_synchronous.top1 > 0`
- Metric: top1_accuracy on `realistic_recall_v1.4` (N=100, seed=0, SBERT)
- Test: paired bootstrap (n=10,000, seed=0), one-tailed p < 0.05 **and**
  bootstrap CI for Δ_Hf1 fully above 0
- Holm family: all pairwise ablation comparisons vs `full_aft` (extended from
  6 to 7 with `aft_keyword_synchronous`); Hf1 is reported in the supplementary
  pairwise section and tested at the corrected threshold

### Interpretations

- **Hf1 PASS:** Deferring keyword appraisal to the slow path partially mitigates
  the destructive synchronous override. LeDoux dual-path architecture is beneficial
  compared to synchronous keyword appraisal, even if both are inferior to no appraisal
  (preset affect only).
- **Hf1 FAIL:** Timing of keyword appraisal does not affect the degree of
  destructive override on this dataset. The mechanism by which preset valence/arousal
  is overwritten is the same in both paths. Interpretation: the caveat in He1 is
  moot on this dataset; Addendum F does not recover the dual-path advantage.

---

## Dataset, embedder, statistics

- Dataset: `realistic_recall_v1.4` (N=100) — same as Addendum E
- Embedder: `sbert-bge` (`bge-small-en-v1.5`) — paper-canonical
- Bootstrap: n=10,000 paired, seed=0
- Holm correction: applied across all 7 ablation variants vs `full_aft` (H_full,
  Ha, Hb, Hc, Hd, He2, He1, Hf1) — Holm family grows from 6 to 7

## Implementation contract

This addendum is committed BEFORE the implementation and benchmark run.
The following must be true at execution time:

1. `AFTKeywordSynchronousReplayAdapter` in `benchmarks/ablation/runner.py`:
   - Subclass of `AFTReplayAdapter`
   - Injects `KeywordAppraisalEngine` in `begin_session` (same pattern as
     `AFTDualPathReplayAdapter`)
   - Does NOT call `engine.elaborate()` — synchronous path
   - Uses default `EmotionalMemoryConfig` (no `dual_path_encoding=True`)
2. `ABLATIONS` list includes `("aft_keyword_synchronous", {}, AFTKeywordSynchronousReplayAdapter)`
3. `_ADAPTER_OVERRIDES` maps `"aft_keyword_synchronous"` to the new class
4. `results.sbert.protocol.json` shows 8 variants (was 7)
5. `results.sbert.md` includes a supplementary section "Hf1 pairwise:
   dual_path vs aft_keyword_synchronous" with Δ, CI, and verdict

## What this study cannot show

- Whether deferral mitigates non-destructive appraisal (requires a dataset without
  hand-curated preset affect).
- Whether the dual-path architecture helps on a dataset where LLM appraisal is
  non-destructive (requires `LLMAppraisalEngine` conditions).
- Addendum G (future): test dual_path with `LLMAppraisalEngine` on a dataset
  without preset affect injection.
