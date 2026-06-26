# Pre-registration Addendum U — Hu1/Hu2: Circularity audit of realistic_recall_v2

**Status:** EXECUTED (2026-06-27) — Hu1 **PASS** (62.5% favorable; neutral Δ ns). See
`preregistration_addendum_u_circularity_audit_closure.md`.
**Date (pre-reg):** 2026-06-27
**Embedder:** `sbert-bge` (bge-small-en-v1.5) — matches the headline +0.205 claim
**Dataset:** `realistic_recall_v2` (50 scenarios, 200 queries, oracle affect)
**No LLM** — deterministic, retrieval-only audit.
**Issue/limitation:** `docs/research/08_limitations.md` §2.4 — "the proportion of
AFT-favorable vs neutral scenarios has **not** been audited."

## Motivation

`08_limitations` §2.4 admits the v2 affect labels were "hand-crafted by the author
with AFT theory in mind" and that the AFT-favorable vs neutral split has never been
quantified. The headline advantage (top1 Δ=+0.205 SBERT) and the Addendum R downstream
PASS therefore rest on a benchmark that may be AFT-favorable **by construction**. This
audit measures that: it partitions the 200 queries by whether affect _can_ help, and
re-reports the AFT−cosine advantage on the subset where affect is **not** discriminative.

## Operational definitions (computed from data + the headline embedder — NOT from the author's `challenge_type`)

For each query:

- **cosine-solvable** — the gold memory (`expected_memory_ids`) is the top-1 of the
  `naive_cosine` baseline (semantics alone already wins → affect cannot add value).
  Taken directly from naive_cosine's per-query `top1_hit`.
- **affect-separating** — in (valence, arousal) space, the gold memory is _strictly_
  the closest candidate to the query `state` (Euclidean distance) among all scenario
  events; ties → not separating. Computed from committed fields, independent of
  `challenge_type`.
- **2×2 partition.** The **affect-only-can-help** cell = `not cosine-solvable AND
affect-separating`; the other three cells are **neutral** for affect (cosine already
  solves it, or affect does not separate the gold).

## Hypotheses / quantities

- **Descriptive (primary).** The 2×2 cell counts (the never-done audit) and the
  AFT-favorable fraction of the 200 queries.
- **Hu1 (directional).** The AFT−cosine top1 advantage is _concentrated_ in the
  affect-favorable cell.
- **Hu2 (author-labeling validity, descriptive).** Contingency of data-driven
  `affect-separating` vs the author's `challenge_type` (expectation: the four
  "affective" types separate; `recency_confound` does not).

## Statistical analysis plan (pre-declared)

- Per-query top1 for `aft` and `naive_cosine` (same adapters, embedder, top_k, and
  query-`state` injection as the canonical realistic runner).
- Δ = mean(aft top1) − mean(cosine top1), per cell, with 95% paired bootstrap CI,
  `n_bootstrap=2000`, `seed=42` (`benchmarks/common/statistics.paired_bootstrap_diff`).
- N = 200 (full dataset).

## Decision rule

- **Hu1 PASS** (advantage is largely a design artifact) iff `Δ_favorable > Δ_neutral`
  **and** the **neutral-subset** Δ 95% CI **includes 0** (where affect does not
  discriminate, AFT does not beat cosine).
- **Hu1 FAIL** (advantage is more general than the circularity worry) iff AFT also
  beats cosine on the neutral subset (its Δ CI excludes 0 and is positive).
- Both outcomes are reported honestly; the audit is descriptive first, the directional
  call second. No post-hoc reframing.

## Execution

```bash
make bench-circularity-audit                 # full deterministic run (no API key)
uv run python -m benchmarks.circularity_audit.runner --limit-scenarios 2   # quick check
```

**Pre-registration integrity:** committed before the runner is executed; the closure
(`..._closure.md`) reports the realized 2×2 counts, per-cell Δ/CI, and the Hu1/Hu2 verdicts.
