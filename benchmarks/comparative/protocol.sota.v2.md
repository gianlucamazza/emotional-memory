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

## Results (populated post-run)

<!-- This section is populated after execution. -->
