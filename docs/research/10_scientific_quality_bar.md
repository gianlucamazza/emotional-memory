# Scientific Quality Bar

This document defines the minimum evidence threshold required before the
project can claim "scientifically validated emotional memory system" rather
than "strong theory-driven prototype with early controlled evidence".

It is the project's own commitment to its standards, written before the
evidence exists, and frozen in git for provenance.

---

## Current honest claim (as of 2026-04-24)

> **"Strong theory-driven prototype with early controlled evidence."**
>
> The library implements Affective Field Theory (AFT) with explicit theoretical
> grounding (Russell, Hebb, Spinoza, Scherer, Collins & Loftus). It passes 126
> psychological invariant tests, shows a controlled comparative advantage on a
> synthetic affect-aware retrieval probe (v1.4, DISCOVERY), and has been
> deployed to a public demo. No peer review, no external benchmark results
> committed, no human rater data collected.

This is the correct public claim until all three mandatory gates below are passed.

---

## Three mandatory gates before upgrading the claim

### Gate 1 — External benchmark (ecological validity)

**Criterion:** LoCoMo Study S1 (pre-reg committed 2026-04-24) produces
committed results with H1 or H2 surviving Holm correction on the full 1986
QA pairs.

**What it proves:** AFT is useful on real human conversational data — not just
a synthetic benchmark designed around the model's assumptions.

**Status:** Running (as of 2026-04-24). Results pending.

---

### Gate 2 — Human evaluation (perceived utility)

**Criterion:** ≥5 raters × ≥20 items per condition. Krippendorff's α ≥ 0.67
(ordinal) on at least one primary dimension (relevance or affective plausibility).
Wilcoxon signed-rank test AFT > baseline, p < 0.05 on the primary dimension.

**What it proves:** The emotional-memory layer improves the quality of a
conversation as *perceived by people*, not just as measured by automated metrics.

**Status:** Packet ready (`benchmarks/human_eval/`). Raters not recruited.

**Minimum rater requirements:**
- ≥5 raters (3 is insufficient for α reliability with disagreement resolution)
- Blind to condition (system label hidden)
- Failure cases must be logged: any item scored 1 by ≥3 raters is a failure case
  and must appear in the paper's limitations section
- Krippendorff α is a publishability criterion, not an optional statistic

---

### Gate 3 — Architecture attribution (appraisal confound resolved)

**Criterion:** Study `benchmarks/appraisal_confound/` shows that
`aft_keyword` (AFTReplayAdapter + KeywordAppraisalEngine, no LLM) beats
`naive_cosine` with Δ > 0 and bootstrap CI excluding 0.

**What it proves:** The AFT retrieval advantage comes from the architecture
(6-signal scoring, resonance, momentum, mood congruence), not from an LLM
appraisal prompt injecting high-quality affect signals into the retrieval.

Without this, the claim "AFT improves retrieval" could be explained away as
"LLM appraisal creates better embeddings". The gate makes that explanation
testable and either falsifies or confirms it.

**Status:** Executed 2026-04-26. Results in
`benchmarks/appraisal_confound/results.json` (SBERT, N=100, n\_bootstrap=10 000,
seed=42, one-tailed alpha=0.05).

**Finding:** Pre-registered Ha2 (aft\_keyword > naive\_cosine) **FAILS** —
Δ = −0.39, p ≈ 0. Keyword appraisal overrides hand-curated preset
valence/arousal and degrades retrieval. Hb2 also fails (Δ = −0.62).

**Architecture attribution (descriptive):** aft\_noAppraisal (no appraisal
engine) vs naive\_cosine — Δ ≈ +0.23, CI [+0.12, +0.33]. Combined with S2
(which also used no appraisal engine), this is the cleanest evidence that
the architecture drives the retrieval advantage, not appraisal signal
injection. Gate 3 is **partially closed**: the architecture-only advantage
is confirmed descriptively; the pre-registered formal test did not pass
because the hypothesis was framed around the wrong comparison.

**Next step to fully close Gate 3:** a new confirmatory pre-registration
framing `aft_noAppraisal > naive_cosine` as the primary hypothesis.

---

## Additional quality requirements for the paper

These are not blocking gates but must be satisfied before journal / main-track
conference submission (workshop submission can proceed at Gate 1 + partial Gate 2):

### Cross-embedder validation

Results must be reported on ≥2 meaningfully different embedder classes:
- Class A: dense sentence embedding (e.g. `bge-small-en-v1.5`, `e5-small`)
- Class B: sparse / hash (current `TokenHashEmbedder`)

If the advantage disappears on Class B, the claim must be scoped to
"dense-embedding settings".

### Multilingual slice

Realistic recall v2 must include ≥1 non-English slice (Italian or Spanish
preferred; at least 20 scenarios). If performance degrades significantly, the
paper's scope must be stated as "English conversational text".

### Failure case disclosure

Every submitted version of the paper must include a failure cases section with
at minimum:
- Cases where AFT retrieval score < naive_cosine (adversarial to AFT)
- Human eval items that scored 1 from ≥3 raters
- Category or challenge type with the largest negative gap

Omitting failure cases is not acceptable, even if overall aggregate is positive.

---

## Claim upgrade path

| Evidence level | Correct public claim |
|---|---|
| Current (only controlled synthetic) | "Strong theory-driven prototype with early controlled evidence" |
| Gate 1 passed | + "with confirmed external benchmark advantage on LoCoMo" |
| Gate 1 + Gate 3 passed | + "with architecture attribution (advantage persists without LLM appraisal)" |
| All 3 gates + cross-embedder | "Scientifically grounded emotional memory system" |
| All 3 gates + multilingual + peer review | "Scientifically validated" |

---

## Files this document governs

- `benchmarks/preregistration.md` — Study S1 (Gate 1)
- `benchmarks/preregistration_addendum_v2.md` — Gate 3 + v2 studies
- `benchmarks/human_eval/` — Gate 2
- `benchmarks/appraisal_confound/` — Gate 3
- `docs/research/09_current_evidence.md` — live evidence matrix (updated per study)
- `README.md` — public-facing claim must match the claim level in this table
