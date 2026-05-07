# Comparative Benchmark — SOTA Extension (Mem0, LangMem)

**Status:** EXECUTING
**Date:** 2026-05-07
**Embedder:** `sbert` (`sentence-transformers/all-MiniLM-L6-v2`)
**LLM model (mem0, langmem):** `gpt-4.1-mini`
**Output:** `benchmarks/comparative/results.sota.sbert.{csv,md,protocol.json}`
**Parent protocol:** `benchmarks/comparative/protocol.md`

---

## Context

The comparative benchmark in `results.sbert.{csv,md}` (committed `6c05ba2`)
evaluates AFT, naive_cosine, and recency on `affect_reference_v1` (N=258
examples, 4 queries × 5 top-k = 20 ranked items). Mem0 and LangMem adapters
exist (`adapters/mem0_adapter.py`, `adapters/langmem_adapter.py`) but were
not previously executed against the benchmark — the prior commits report
only the deterministic systems.

This SOTA extension closes the W3 gap identified in the internal review
(`il-nostro-studio-sta-iridescent-river.md` §3): "no end-to-end comparison
with external SOTA". It runs the existing benchmark with the additional two
LLM-backed systems (Mem0, LangMem) under the protocol of `protocol.md`.

## Why this is "exploratory reference" not "definitive"

Per `protocol.md` §What this benchmark does not support claiming:
- The benchmark probes mood-congruent retrieval behaviour, not QA quality.
- AFT receives explicit query affect; Mem0 and LangMem do not consume the
  affect channel of the dataset (their adapters ignore `valence`/`arousal`
  on encode and retrieve).
- The dataset (N=258 affect-labelled examples) is narrower than the
  intended product surface of Mem0/LangMem (multi-session conversational
  recall, agentic state).

The result is therefore reported as **exploratory reference**, not a
production benchmark. It answers: "On the same affect-aware probe AFT was
designed for, where do production LLM-backed memory systems land relative
to the deterministic baselines?".

## Model choice rationale

The first attempt (2026-05-07 ~01:46, 25 min runtime, killed at mem0
encode 89/258) used `gpt-5-mini` with `EMOTIONAL_MEMORY_LLM_REASONING_EFFORT=minimal`.
At ~15-20 s per LLM call (reasoning model), the projected total runtime
was 3-4 hours. Audit trail at `benchmarks/comparative/.partial_run_25m.log`.

For this re-run we picked `gpt-4.1-mini` (released 2025-04, the current
production-tier OpenAI mini model):

- ~1-2 s per call (non-reasoning, similar latency class to `gpt-4o-mini`)
- Better instruction following than `gpt-4o-mini` per the OpenAI 2025-04
  release notes — closer to what users would deploy in production
- Estimated runtime: ~40 min for full 258-example encode + retrieve on
  both Mem0 and LangMem

The model used by Mem0/LangMem under this protocol is reported in the
output `protocol.json` and in the closure document below. The `aft` system
in this run does NOT use any LLM (deterministic adapter); the LLM model
choice only affects Mem0 and LangMem.

## Hypothesis (H_sota, exploratory)

> On `affect_reference_v1` with the AFT-aware protocol (top_k=5, 4
> quadrant queries), AFT's recall@k is at least as good as the best of
> Mem0 and LangMem.

**Pre-registered decision rule:**
- This is exploratory, not confirmatory. We do not Holm-correct.
- Report point estimate, 95% CI (paired bootstrap, n=2000, seed=0), and
  pairwise paired tests (bootstrap + McNemar) per the existing
  `protocol.md` reporting requirements.
- Sample size is fixed by the dataset (N=20 ranked items). No new examples
  are added.

**What a "PASS" would mean:** AFT is competitive with production LLM-backed
memory on the controlled affective probe. This is a weak signal — it does
not establish AFT superiority, only non-inferiority on a narrow task.

**What a "FAIL" would mean:** Mem0 or LangMem outperforms AFT on the
affect-aware probe. This would suggest that LLM-based memory extraction
+ vector search captures the affective ranking signal at least as well as
AFT's hand-crafted scorer, weakening the case for the AFT architecture
beyond educational/research value.

## Coherence with prior closures

- **Hd1/Hd2 (oracle preset affect):** AFT advantage on `realistic_recall_v2`
  is established under preset valence/arousal. This SOTA run uses the same
  preset-affect protocol on a different probe (affect_reference_v1).
  Outcome here speaks to the same regime, not to LLM-inferred affect.
- **Hg1 FAIL (Addendum G, 2026-05-07):** With LLM appraisal, AFT does not
  beat naive cosine. The SOTA run is run with AFT in oracle-affect mode
  (consistent with the pre-existing `results.sbert.md`); it is not a
  test of LLM-inferred affect.
- **LoCoMo (S1) FAIL:** AFT underperforms naive RAG on factual QA. SOTA
  comparison here is on affect-aware retrieval (a different task), not
  factual QA. Outcomes do not directly compare.

## Reporting

Closure document and result interpretation will be appended below in
§Results once the run completes. Files produced:

- `benchmarks/comparative/results.sota.sbert.csv` — raw per-system
- `benchmarks/comparative/results.sota.sbert.md` — human-readable summary
- `benchmarks/comparative/results.sota.sbert.protocol.json` — execution
  metadata (seeds, models, dataset hash, embedder)
