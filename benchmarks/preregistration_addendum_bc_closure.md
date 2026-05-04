# Pre-registration Closure — Addendum B + C

**Date written:** 2026-05-05
**Protocol version:** addendum_bc_closure_v1
**Parent pre-reg:** `benchmarks/preregistration_addendum_v2.md`

> **Epistemic status:** This document formalizes the closure status of
> Addendum B (realistic recall v2 cross-embedder + multilingual) and
> Addendum C (human evaluation pilot updated criteria). Neither addendum
> is executed in full as originally scoped; both have been either
> subsumed by later addenda or remain pending execution.

---

## Addendum B — Realistic Recall v2 (cross-embedder + multilingual)

### Pre-registered scope

Addendum B extended Study S2 with two requirements:
1. **Cross-embedder**: results reported on Class A (SBERT bge-small-en)
   and Class B (e5-small-v2 or equivalent).
2. **Multilingual slice**: ≥20 scenarios in Italian (primary) or Spanish.

### Closure status: SUBSUMED by Addendum H + S2 confirmatory run

The cross-embedder requirement (point 1) is fully addressed by the S2
confirmatory run (2026-05-04):
- SBERT bge-small-en: AFT top1=0.53 vs naive 0.33, Δ=+0.205 [0.15,0.27], p<0.001
- e5-small-v2: AFT top1=0.50 vs naive 0.34, Δ=+0.155 [0.09,0.22], p<0.001

The multilingual requirement (point 2) is addressed by Addendum H (Italian,
2026-04-27) and the Hd2_ES extension (Spanish, 2026-05-04):
- Italian (v2_it, me5): AFT Δ=+0.163, p=0.012 (PASS, Addendum H)
- Spanish (v2_es, SBERT): AFT Δ=+0.138, p=0.045 (PASS, exploratory)
- Spanish (v2_es, me5): AFT Δ=+0.113, p=0.110 (FAIL, exploratory)

Failure case logging (point 3): per-challenge Holm analysis in
`benchmarks/realistic/challenge_subset_pairwise_v2.json` shows that H5
(recency_confound) SBERT and H6 (momentum_alignment) e5 are the two
failure sub-types.

**Verdict: Addendum B is considered CLOSED by subsumed execution.** No
separate `bench-addendum-b` run is required.

Canonical result files:
- `benchmarks/realistic/results.v2.sbert.json` / `.md` (Class A)
- `benchmarks/realistic/results.v2.e5.json` / `.md` (Class B)
- `benchmarks/realistic/results.v2_it.me5.json` / `.md` (IT)
- `benchmarks/realistic/results.v2_es.sbert.json` / `.md` (ES SBERT)
- `benchmarks/realistic/results.v2_es.me5.json` / `.md` (ES me5)
- `benchmarks/realistic/challenge_subset_pairwise_v2.json` (Holm family)

---

## Addendum C — Human Evaluation Pilot (updated criteria)

### Pre-registered scope

Addendum C updated the human-eval pilot criteria from `preregistration.md §M1.2`:
1. Minimum raters: ≥5 (upgraded from ≥3)
2. Failure case field in rating template
3. Publishability criterion: Krippendorff's α ≥ 0.67 on "relevance"
4. Within-rater blinded design with counterbalanced order

### Closure status: PENDING EXECUTION

The human-eval pipeline is operationally ready (`benchmarks/human_eval/`):
packets, rating templates, alpha computation, and summary generation are
all implemented. The pipeline rejects placeholder `ratings.jsonl` and
validates rater count.

However, **no actual ratings have been collected** from human participants.
The pipeline satisfies the procedural criteria of Addendum C; the empirical
criteria (≥5 raters, α ≥ 0.67) cannot be evaluated without execution.

**Verdict: Addendum C remains PENDING.** Closure requires recruiting ≥5
raters, completing the rating round, and committing a `ratings.jsonl` with
valid entries. Until then, the claim `models_human_emotional_memory` stays
at status `not_established`.

Pipeline entry point: `make bench-human-eval` (or `uv run python -m
benchmarks.human_eval.runner`). Rater material template: `benchmarks/human_eval/`.
