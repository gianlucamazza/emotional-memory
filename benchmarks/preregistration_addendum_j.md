# Pre-registration Addendum J — LoCoMo Per-Task Pareto Study

**Date written:** 2026-05-06
**Protocol version:** addendum_j_v1
**Parent pre-regs:** `benchmarks/preregistration.md` (S1 — LoCoMo primary run)
**Closes:** #26 (protocol; execution deferred pending LLM budget)

> **Epistemic status:** This is a **prospective pre-registration**. No Pareto-sweep
> data has been collected as of this writing. The S1 dataset and per-category
> results (committed in `benchmarks/locomo/results.json`) pre-date and motivate
> this design, but are NOT used to derive the weight grid below — they are used
> only for cost estimation and power context.
>
> **This document must not be modified after execution begins.** The weight grid
> (§Hj1 Weight grid), sampling protocol (§Sampling), and exclusion criteria are
> frozen at commit time.

---

## Motivation

S1 run (2026-04-27) confirmed Gate 1 FAIL: AFT underperforms naive_rag on all
four LoCoMo QA categories:

| Category | n | AFT F1 | naive_rag F1 | Δ |
|---|---:|---:|---:|---:|
| multi_hop | 282 | 0.172 | 0.227 | −0.055 |
| temporal | 321 | 0.049 | 0.077 | −0.028 |
| open_domain | 96 | 0.092 | 0.103 | −0.011 |
| single_hop | 841 | 0.221 | 0.379 | −0.158 |

The S1 run used fixed default weights
`base_weights = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]`
(`[semantic, mood_congruence, affect_proximity, momentum_alignment, recency, resonance_boost]`).

**Research question:** Is there a weight configuration under which AFT matches
or beats naive_rag on at least one LoCoMo category, and what is the accuracy
cost on other categories?

This is an exploratory Pareto study. There are no confirmatory claims.
A Pareto-favourable result would motivate per-task configuration tuning as
a future feature; a null result would close this line of inquiry.

---

## Systems

- **`aft_config_{W}`** — `AFTLoCoMoAdapter` instantiated with
  `EmotionalMemoryConfig(retrieval=RetrievalConfig(base_weights=W))`.
  Same embedder (`bge-small-en-v1.5`) and `KeywordAppraisalEngine` as S1.
- **`naive_rag`** — same as S1. One run only (not repeated per weight config;
  its per-QA predictions from S1 are reused if checkpointing allows, otherwise
  re-run once).

---

## Weight grid (frozen)

Ten weight configurations, covering a structured sweep of the two signals
most likely to drive AFT-specific advantage (mood congruence, resonance) vs
the semantic baseline:

| Config ID | semantic | mood | affect_prox | momentum | recency | resonance | Description |
|---|---:|---:|---:|---:|---:|---:|---|
| W0 | 0.35 | 0.25 | 0.15 | 0.10 | 0.10 | 0.05 | S1 default (baseline) |
| W1 | 0.60 | 0.15 | 0.10 | 0.05 | 0.05 | 0.05 | High semantic |
| W2 | 0.50 | 0.30 | 0.10 | 0.05 | 0.05 | 0.00 | High semantic + mood, no resonance |
| W3 | 0.40 | 0.35 | 0.10 | 0.05 | 0.05 | 0.05 | Elevated mood |
| W4 | 0.35 | 0.20 | 0.10 | 0.05 | 0.25 | 0.05 | High recency |
| W5 | 0.35 | 0.10 | 0.10 | 0.05 | 0.35 | 0.05 | Very high recency (temporal QA) |
| W6 | 0.35 | 0.05 | 0.05 | 0.05 | 0.45 | 0.05 | Recency-dominant |
| W7 | 0.70 | 0.10 | 0.08 | 0.04 | 0.04 | 0.04 | Very high semantic |
| W8 | 0.50 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | Uniform non-semantic |
| W9 | 0.45 | 0.20 | 0.12 | 0.08 | 0.10 | 0.05 | Midpoint S1 + W7 |

All rows sum to 1.0 within floating-point tolerance. Configs W4–W6 are
motivated by the temporal category gap (largest Δ in absolute terms on a
per-QA basis given n=321); W1/W7 test whether reducing affective noise
closes the single_hop gap.

No post-hoc addition of weight configs is permitted after the first execution.

---

## Sampling protocol (frozen)

**Held-out sweep subset:**

To reduce LLM cost, the Pareto sweep runs on a stratified subsample of
LoCoMo:

- **N = 50 QA per category** (categories: multi_hop, temporal, open_domain,
  single_hop). Total: **200 QA pairs**.
- Sample is drawn **before any weight-config run** (sampling seed = 42).
- Sample is fixed and identical across all 10 weight configs. Re-use naive_rag
  predictions for the same subset from S1 if available; otherwise re-run
  naive_rag on the subset once.
- The adversarial (category-5) QA pairs are excluded from this study —
  S1 already handled adversarial scoring as a separate track.

**Implementation note:** The LoCoMo runner does not natively support
stratified subsampling. Before execution, add a `--sample-per-category N
--sample-seed K` argument to `benchmarks/locomo/runner.py` (or implement
in a new `benchmarks/locomo/pareto_runner.py`). The sampling logic must be
committed before running.

---

## Primary analysis (Hj1)

**Hj1 (exploratory, non-confirmatory):** For at least one weight config
W ∈ {W1…W9} and at least one category C ∈ {multi_hop, temporal, open_domain,
single_hop}, `aft_config_W.F1(C) ≥ naive_rag.F1(C)` on the 50-QA-per-category
subsample.

- **Test:** point-estimate comparison only (no hypothesis test claimed as
  confirmatory — the sweep is exploratory search, not a pre-specified
  directional hypothesis).
- **Reporting:** full (W × C) matrix of F1 deltas relative to W0 (S1 default)
  and relative to naive_rag. Pareto-optimal configs highlighted.
- **Pareto definition:** a config W dominates W0 on category C if
  `aft_W.F1(C) > aft_W0.F1(C)` by at least +0.01 without reducing
  aggregate F1 below `aft_W0.F1(all) − 0.05`.

If Hj1 passes (≥1 Pareto-favourable config found):
- Report the Pareto-optimal weight profile.
- Recommend a full-N (1986 QA) replication run for the best config.
- Do NOT claim superiority to naive_rag without a pre-registered full-N
  replication.

If Hj1 fails (no config improves any category without aggregate regression):
- Document as honest negative. Closes the per-task tuning line of inquiry
  for `base_weights`. Note that architectural changes (e.g., per-category
  weight routing) may still be worth exploring as a future study.

---

## Cost estimate

LLM calls per weight config run on 200 QA:
- Answer generation: 200 calls × ~1k tokens/call
- Judge scoring: 200 calls × ~0.5k tokens/call

Total per config (at gpt-5-mini pricing ~$0.30/1M tokens):
≈ 200 × 1500 tokens / 1,000,000 × $0.30 ≈ **$0.09 per config**

Total for 10 configs (excluding naive_rag re-run):
≈ **$0.90** (plus ~$0.09 for naive_rag baseline on the 200-QA subset)

**Total estimated cost: ~$1.00.** Well within a $2–5 budget.

---

## Statistical analysis plan

- All results reported as mean token-F1 per category, Δ vs W0, Δ vs naive_rag.
- No bootstrap CI on per-config per-category metrics (n=50 is too small for
  reliable paired bootstrap; report as point estimates).
- Bootstrap 95% CI reported at **aggregate level only** (n=200) for each
  config, so relative ranking of configs has error bounds.
- No Holm correction — this is entirely exploratory.

---

## Reporting rules

Per `benchmarks/preregistration.md` §Reporting rules:

1. All 10 weight configs reported, not cherry-picked. (W0 = baseline always shown.)
2. Hj1 verdict (pass/fail) reported regardless of outcome.
3. Closure document: `benchmarks/preregistration_addendum_j_closure.md`.
4. Any exploratory analysis beyond the grid (e.g., random search) must be
   clearly labeled "post-hoc, outside pre-reg".

---

## Prerequisites before execution

- [ ] Implement `--sample-per-category` + `--sample-seed` in
      `benchmarks/locomo/runner.py` (or a new `pareto_runner.py`)
- [ ] Commit the sampling code before first run
- [ ] Configure LLM env (`EMOTIONAL_MEMORY_LLM_API_KEY`, model)
- [ ] Dry-run on 2 QA to verify the 10-config loop runs end-to-end
- [ ] Record budget approval and run start timestamp in closure doc

---

## Files to create at execution

- `benchmarks/locomo/pareto_results.json` — (W × C) F1 matrix + aggregate
- `benchmarks/locomo/pareto_results.md` — human-readable table
- `benchmarks/preregistration_addendum_j_closure.md` — verdict + numbers
- Update `docs/research/claim_validation_matrix.json` row
  `locomo_external_qa_negative.not_yet_shown` with Pareto outcome
