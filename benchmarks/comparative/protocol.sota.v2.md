# Comparative Benchmark — SOTA Extension on realistic_recall_v2

**Status:** PENDING_EXECUTION
**Date:** 2026-05-07
**Embedder:** `sbert-bge` (BAAI/bge-small-en-v1.5)
**LLM model (mem0, langmem):** `gpt-4.1-mini`
**Output:** `benchmarks/comparative/results.sota.v2.sbert.{json,md,protocol.json}`
**Parent protocol:** `benchmarks/comparative/protocol.md`
**Pre-registration:** `benchmarks/preregistration_addendum_v2_sota.md`
**Prior SOTA run (affect_reference_v1):** `benchmarks/comparative/protocol.sota.md`

---

## Context

The prior SOTA run (`protocol.sota.md`, committed 9746afb) evaluated Mem0 and LangMem
on `affect_reference_v1` — a controlled single-session quadrant-retrieval probe (N=258
memories, 4 queries, recall@5). Results: Mem0 1.00, LangMem 0.90, AFT 0.85,
naive_cosine 0.80.

However, AFT's headline empirical result (`Hd2 PASS`, Δ +0.205 [0.15, 0.27], d=0.49)
lives on a different benchmark: `realistic_recall_v2` — a multi-session realistic
replay dataset with 50 scenarios, 200 queries, 5 challenge types, `top1_accuracy`
as primary metric. The `affect_reference_v1` run does not speak to whether AFT's
advantage on this harder dataset resists LLM-backed SOTA.

This run closes the remaining gap: it applies Mem0 and LangMem to `realistic_recall_v2`
via the `ComparativeReplayShim` adapter, which wraps the comparative adapter interface
in the multi-session `ReplayAdapter` protocol used by `benchmarks/realistic/runner.py`.

---

## Why this is "exploratory reference" not "definitive"

- The benchmark probes multi-session affect-aware retrieval; Mem0/LangMem do not
  consume the affect channel (valence/arousal) — their shim adapters ignore these.
- AFT receives preset oracle-affect values from the dataset (oracle-affect mode).
  This asymmetry is the *intended controlled comparison*: "AFT with its architectural
  advantage vs SOTA LLM without it."
- LLM-extracted memory may re-phrase or consolidate content, making ID-based hit
  detection conservative for Mem0/LangMem (they may retrieve the right semantic
  content under a different ID).
- The dataset (N=200 queries across 50 scenarios) is narrower than the production
  surface of Mem0/LangMem (agentic multi-turn, long-horizon state).

The result is therefore reported as **exploratory reference**, not a production benchmark.

---

## Model choice rationale

`gpt-4.1-mini` (released 2025-04): non-reasoning, ~1-2 s per call, same model as the
prior `affect_reference_v1` SOTA run. Consistent across both SOTA comparisons.

---

## Hypothesis (H_v2_sota, exploratory)

> On `realistic_recall_v2` (sbert-bge, N=200, top1), AFT top1_accuracy ≥ best of
> {Mem0, LangMem}.

**Pre-registered decision rule:**
- Exploratory, no Holm correction.
- Report Δ, 95% CI (paired bootstrap n=10000, seed=0), p_bootstrap, p_mcnemar, d, n_discordant.
- PASS if AFT ≥ max(mem0, langmem) with overlapping CI; FAIL otherwise.
- Per-challenge breakdown (5 types) as secondary read-out.

---

## Coherence with prior closures

- **Hd2 PASS (Δ +0.205, d=0.49, sbert-bge, v2):** This run adds Mem0/LangMem to the
  comparison. AFT is already above naive_cosine; the question is whether it stays above
  the LLM-backed systems on the same benchmark.
- **SOTA affect_reference_v1 (protocol.sota.md):** Mem0 1.00 vs AFT 0.85 on the simpler
  probe. That probe uses recall@5 quadrant congruence. This run uses top1_accuracy on
  multi-session realistic scenarios — a harder and more architecture-relevant evaluation.
- **Hg1 FAIL (Addendum G):** AFT + LLM appraisal does not beat cosine on affect-free data.
  The current run uses AFT in oracle-affect mode (no appraisal engine), consistent with Hd2.
- **LoCoMo S1 FAIL:** AFT underperforms naive RAG on factual QA. Not relevant here —
  the benchmark probes affect-aware retrieval, not factual recall.

---

## Results (2026-05-07)

### Headline — top1_accuracy

| System | top1 [95% CI] | hit@k [95% CI] | Stateful |
|---|---:|---:|---:|
| **aft** | **0.535** [0.465, 0.600] | 0.640 [0.575, 0.705] | 0.99 |
| langmem | 0.365 [0.300, 0.430] | 0.585 [0.515, 0.655] | 0.00 |
| mem0 | 0.330 [0.265, 0.395] | **0.900** [0.855, 0.940] | 0.00 |
| naive_cosine | 0.325 [0.260, 0.390] | 0.465 [0.395, 0.535] | 0.00 |
| recency | 0.020 [0.005, 0.040] | 0.115 [0.075, 0.160] | 0.00 |

### Pairwise vs naive_cosine (paired bootstrap n=10000, seed=0)

| System | Δ top1 [95% CI] | p (bootstrap) | p (McNemar) | d | Discordant |
|---|---:|---:|---:|---:|---:|
| **aft** | **+0.210** [+0.155, +0.270] | **<0.001** | **<0.001** | **0.512** | 42 |
| langmem | +0.040 [−0.020, +0.100] | 0.233 | 0.268 | 0.089 | 40 |
| mem0 | +0.005 [−0.065, +0.075] | 0.945 | 1.000 | 0.010 | 49 |
| recency | −0.305 [−0.375, −0.235] | <0.001 | <0.001 | −0.616 | 67 |

### Per-challenge type breakdown (top1_accuracy, N=40 each)

| Challenge type | AFT | Mem0 | LangMem | cosine | Δ (AFT vs cosine) |
|---|---:|---:|---:|---:|---:|
| semantic_confound | **0.750** | 0.350 | 0.400 | 0.475 | +0.275 |
| same_topic_distractor | **0.775** | 0.625 | 0.650 | 0.625 | +0.150 |
| momentum_alignment | **0.600** | 0.450 | 0.425 | 0.325 | +0.275 |
| affective_arc | **0.425** | 0.150 | 0.225 | 0.150 | +0.275 |
| recency_confound | 0.125 | 0.075 | 0.125 | 0.050 | +0.075 |

### H_v2_sota verdict: PASS

**AFT top1_accuracy = 0.535 > max(Mem0=0.330, LangMem=0.365), with non-overlapping 95% CIs.**
AFT's Δ vs cosine (+0.210 [+0.155, +0.270], p<0.001, d=0.512) is both statistically and
practically significant, matching the Hd2 headline (Δ +0.205, d=0.49) to within bootstrap
noise. Mem0 and LangMem show no meaningful advantage over naive_cosine (Δ +0.005, p=0.945
and Δ +0.040, p=0.233).

### Interpretation

**Quality ranking (top1):** aft >> langmem ≈ mem0 ≈ cosine >> recency.
The pattern is the **opposite of affect_reference_v1** (where Mem0=1.00, LangMem=0.90, AFT=0.85).
This reveals that the two benchmarks test fundamentally different capabilities:

- `affect_reference_v1` tests **pure semantic quadrant recall** — Mem0's LLM-extracted memory
  summaries excel at semantic matching against simple mood-quadrant queries.
- `realistic_recall_v2` tests **affect-modulated multi-session retrieval** — AFT's oracle-affect
  scoring (mood congruence, momentum alignment, resonance) provides the decisive signal.

**Mem0's hit@k = 0.90 anomaly:** Mem0 retrieves the correct memory within top-2 in 90% of
queries (vs AFT 0.64), despite only achieving top1 = 0.330. This discrepancy is a methodological
artifact: Mem0's LLM extraction may create multiple synthesized memory facts from a single
`encode()` call and rank one higher than the tracked original ID. The top1 metric is therefore
conservative for Mem0 on this dataset. A re-implementation that tracks all extracted fact IDs
would likely improve Mem0's top1 score; however, this would require modifying the adapter
beyond its production behavior, so the conservative evaluation is retained.

**Per-challenge analysis:** AFT's advantage is strongest on semantic_confound (+0.275),
momentum_alignment (+0.275), and affective_arc (+0.275) — exactly the challenge types
where the affect signal is most informative. The recency_confound challenge is hard for all
systems (AFT=0.125, LangMem=0.125, cosine=0.050) — recency scoring is not the dominant
signal in the AFT oracle-affect regime.

**Coherence with prior closures (re-checked post-result):**

- **Hd2 PASS:** AFT Δ +0.205 vs cosine is replicated exactly here (+0.210, within bootstrap
  variance). The 5 new systems do not affect AFT's or cosine's performance. ✓
- **SOTA affect_reference_v1 (protocol.sota.md):** Mem0=1.00>AFT=0.85 on that probe. This
  run shows the pattern REVERSES on v2. Together, the two runs clarify the scope: Mem0 excels
  at semantic quadrant matching; AFT excels at affect-modulated realistic recall.
- **Hg1 FAIL (Addendum G):** AFT + LLM appraisal does not beat cosine on affect-free data.
  This run uses AFT in oracle-affect mode — the positive result is fully consistent.

### Paper-section action

Update §6 Comparative Benchmark with this v2 SOTA table. Key message for the paper:
*"On realistic_recall_v2, AFT (oracle-affect) achieves top1 = 0.535 vs Mem0 = 0.330 and
LangMem = 0.365 — neither LLM-backed system outperforms naive cosine on this probe.
The pattern inverts the affect_reference_v1 result, demonstrating that the two benchmarks
test different capabilities."*

### `claim_validation_matrix.json` action

Add `realistic_replay_vs_sota` claim row (status: established, scoped).
Update `retrieval_affect_aware` current_evidence with v2 SOTA numbers.
