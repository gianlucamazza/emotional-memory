# Pre-registration Addendum T — Ht1/Ht2: Retrieve-time query appraisal vs oracle state-injection

**Status:** PENDING*EXECUTION
**Date (pre-reg):** 2026-06-27
**Embedder:** `sbert-bge` (bge-small-en-v1.5) — matches the headline +0.205 claim
**Dataset:** `realistic_recall_v2` (50 scenarios, 200 queries, oracle affect)
**LLM:** direct-VAD appraisal of the query text (`DIRECT_VAD_SCHEMA`, Addendum V), resolved
from `EMOTIONAL_MEMORY_LLM*\*` (`.env`pins`gpt-5-mini`). ~200 calls (one per query).

## Motivation

This is the only untested lever on the **state-injection boundary** (A2) that scopes every
AFT result. The benchmark's advantage comes from injecting each query's oracle `state`
(valence/arousal) as the engine's `core_affect` before retrieval (`AFTReplayAdapter.retrieve`
→ `engine.set_affect`). Production has no oracle state — it must **appraise the query text**.
Every recovery attempt that kept the oracle state but reshaped the channel has FAILED:
weight tuning (Hj1), routing (Hl), gating (Hq — "the bottleneck is the channel, not the gate"),
mapping recalibration (Hp). The untested mechanism: **derive the query affect by appraising
the query itself** and inject _that_ as the state. Addendum V gives a stronger query estimator
(direct-VAD) to do it with.

The question: **can retrieve-time query appraisal substitute for oracle state-injection?**
The oracle arm IS the established upper bound (= the +0.205 headline).

## Arms (same dataset, embedder, top_k; only the query-affect source differs)

1. `cosine` — `NaiveCosineReplayAdapter` (no affect). Baseline.
2. `aft_oracle` — `AFTReplayAdapter` with the dataset's oracle `query.state` injected
   (= the headline; the upper bound).
3. `aft_query_appraised` — `AFTReplayAdapter` where the query's affect is obtained by
   **appraising the query text** with `DIRECT_VAD_SCHEMA` → `CoreAffect`, and _that_ valence/
   arousal is injected as the state (production-reachable; no oracle).

## Hypotheses / quantities

- **Ht1 (primary).** `aft_query_appraised` top1 > `cosine` top1 (Δ > 0, CI excludes 0).
- **Ht2.** Recovery fraction `= (appraised − cosine) / (oracle − cosine)` — how much of the
  oracle advantage query appraisal recovers.
- **Diagnostic D.** Pearson r between the appraised query affect (valence, arousal) and the
  oracle `query.state` — explains any recovery gap.
- **Secondary.** Ht1 restricted to the affect-favorable subset (Addendum U partition:
  not-cosine-solvable AND affect-separating), where recovery, if any, should concentrate.

## Statistical analysis plan (pre-declared)

- Per-query top1 for all three arms; paired bootstrap on the appraised−cosine and
  oracle−cosine differences, `n_bootstrap=2000`, `seed=42`
  (`benchmarks/common/statistics.paired_bootstrap_diff`). N = 200.

## Decision rule

- **Ht1 PASS** → retrieve-time query appraisal **substitutes (at least partially) for oracle
  state-injection**: the headline advantage is production-reachable without preset affect.
  Report the recovery fraction; a partial recovery is a positive, nuanced result.
- **Ht1 FAIL** → AFT is **oracle-bound**: even appraising the query cannot reach its oracle
  advantage; the last mechanism on the state-injection boundary is closed (a definitive,
  honest capstone).
- Result stands as measured; no post-hoc reframing.

## Execution

```bash
make bench-query-appraisal                                           # full run (requires API key)
uv run python -m benchmarks.query_appraisal.runner --limit-scenarios 2   # quick check
```

**Pre-registration integrity:** committed before execution; the closure
(`..._closure.md`) reports realized top1 per arm, Δ/CI, recovery fraction, diagnostic r,
the favorable-subset breakdown, and the Ht1 verdict.
