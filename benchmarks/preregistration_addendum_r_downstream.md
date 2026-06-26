# Pre-registration Addendum R — Hr1: Downstream answer quality (encode→retrieve→generate→judge)

**Status:** EXECUTED (2026-06-26) — Hr1 **PASS**, Hr2 **PASS**. See
`preregistration_addendum_r_downstream_closure.md`.
**Date (pre-reg):** 2026-06-26
**Embedder:** `sbert-bge` (bge-small-en-v1.5)
**Dataset:** `realistic_recall_v2` (50 scenarios, 200 queries; **oracle affect**)
**LLM:** resolved from `EMOTIONAL_MEMORY_LLM*\*`(project`.env`pins`gpt-5-mini`);
the judge uses the same model.
**Issue:** #61 (A3). Closes the "future work" item recorded in
`docs/research/problem_register_2026-06.md`§A3 and`08_limitations.md` §2.2/§2.4.

## Motivation

AFT's retrieval-ranking advantage is real **only** in the affect-discriminative
regime: on `realistic_recall_v2` AFT beats `naive_cosine` at top-1 by
Δ=+0.205 [0.150, 0.265], d=0.49 (Hd2 PASS, SBERT). The open question (A3) is
whether that _ranking_ gain converts to **downstream value** once a generator
consumes the retrieved memories — i.e. does better ranking yield better answers?

This has never been measured **in the regime where AFT wins**. The existing
LoCoMo benchmark already answers the downstream question in the _oracle-free,
naturalistic_ regime and AFT **loses** there (judge acc 0.279 vs 0.441,
Δ=−0.159, Gate 1 FAIL). Addendum R isolates the complementary case: the
affect-discriminative regime, with the full encode→retrieve→generate→judge loop.

Scope boundary (carried from A2): like all Hd\* results, this uses **oracle
affect** (preset valence/arousal at encode) and query-time state injection. A
positive Hr1 therefore bounds to "given affect-discriminative inputs, AFT's
ranking edge converts to answer-quality edge", **not** an automatic-appraisal or
production claim.

## Systems (identical generator; only retrieval differs)

| System         | Retrieval                                          | Affect                                         |
| -------------- | -------------------------------------------------- | ---------------------------------------------- |
| `aft`          | full 6-signal AFT (`AFTReplayAdapter`)             | oracle valence/arousal at encode + query state |
| `naive_cosine` | embedding cosine only (`NaiveCosineReplayAdapter`) | none                                           |

Both share: same embedder, same `top_k` (dataset `default_top_k`, override via
`--top-k`), same answer-generation prompt (`_ANSWER_SYSTEM`), same LLM,
`temperature=0`, same judge prompt. This isolates the contribution of the
retrieval stage.

## Protocol

1. **encode** — per scenario, per session: store every event with its oracle
   `valence`/`arousal` (AFT consumes them; cosine ignores them).
2. **retrieve** — per query: top-k items, AFT additionally receives the query's
   `state` (valence/arousal) when present.
3. **generate** — build context from the retrieved items, prompt the LLM for a
   concise answer (`_ANSWER_SYSTEM`, temperature 0).
4. **judge** — LLM-as-judge (the LoCoMo judge prompt) labels the generated
   answer CORRECT/WRONG against the **gold answer = concatenated `content` of the
   query's `expected_memory_ids`**.

Per query we also log the retrieval `top1_hit` (gold memory ranked first) so the
report can show the ranking→answer conversion side by side.

## Hypotheses

- **Hr1 (primary).** `Δ(judge_correct) = mean(AFT) − mean(cosine) > 0` on the
  full 200-query set.
- **Hr2 (secondary).** `Δ(token_F1) > 0`.

## Statistical analysis plan (pre-declared)

- Pairing: by `query_id` across the two systems (identical query set).
- Paired bootstrap on per-query differences, `n_bootstrap=2000`, `seed=42`, 95%
  CI, two-sided p (`benchmarks/common/statistics.paired_bootstrap_diff`).
- McNemar exact on discordant judge pairs (`mcnemar_exact`).
- Multiple comparisons: Holm-Bonferroni across {Hr1 judge, Hr2 F1}.
- N = 200 queries (full dataset; no subsetting in the confirmatory run).

## Decision rule

- **Hr1 PASS** iff Δ > 0 **and** 95% CI lower bound > 0 **and** `p_holm < 0.05`.
- Otherwise **FAIL** → the new `downstream_value` claim is recorded as
  `not_established` (Δ≈0 / CI crosses 0) or `falsified` (Δ<0, CI excludes 0),
  per the matrix legend. No post-hoc reframing; the result stands as measured.

## Execution

```bash
make bench-a3                       # full confirmatory run (requires API key)
# dry-run (no LLM, pipeline check):
uv run python -m benchmarks.downstream.runner --no-judge --limit-scenarios 2
```

**Pre-registration integrity:** this file is committed **before** the runner is
executed against the LLM judge. The closure (`..._closure.md`) reports the
realized numbers and the resulting claim status.
