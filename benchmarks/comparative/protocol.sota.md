# Comparative Benchmark — SOTA Extension (Mem0, LangMem)

**Status:** COMPLETE
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

Files produced:

- `benchmarks/comparative/results.sota.sbert.csv` — raw per-system
- `benchmarks/comparative/results.sota.sbert.md` — human-readable summary
- `benchmarks/comparative/results.sota.sbert.protocol.json` — execution
  metadata (seeds, models, dataset hash, embedder)

---

## Results (2026-05-07)

### Recall@5 (mood-congruent quadrant)

| System | Recall@5 [95% CI] | Encode ms/item | Retrieve p50 ms | Retrieve p95 ms |
|---|---:|---:|---:|---:|
| **mem0** | **1.00** [1.00, 1.00] | 1130.04 | 239.78 | 269.93 |
| langmem | 0.90 [0.75, 1.00] | 150.01 | 160.47 | 164.75 |
| aft | 0.85 [0.65, 1.00] | 45.29 | 45.16 | 47.07 |
| naive_cosine | 0.80 [0.60, 0.95] | 32.47 | 68.19 | 83.15 |
| recency | 0.25 [0.10, 0.45] | 0.01 | 0.02 | 0.04 |

### Pairwise vs naive_cosine (paired bootstrap n=2000, seed=0)

| System | Δ [95% CI] | p (bootstrap) | p (McNemar) | Discordant |
|---|---:|---:|---:|---:|
| mem0 | +0.20 [0.05, 0.40] | **0.045** | 0.125 | 4 |
| langmem | +0.10 [0.00, 0.25] | 0.254 | 0.500 | 2 |
| aft | +0.05 [-0.15, 0.30] | 0.825 | 1.000 | 5 |
| recency | -0.55 [-0.75, -0.35] | <0.001 | 0.001 | 11 |

### Interpretation

**Quality ranking (recall@5):** mem0 > langmem > aft > naive_cosine > recency.
On `affect_reference_v1`, the LLM-backed memory systems (Mem0, LangMem)
score equal to or higher than AFT. Mem0 achieves a perfect recall@5 = 1.00
on all 4 quadrant queries; the bootstrap pairwise vs naive_cosine reaches
nominal p = 0.045 (McNemar p = 0.125 — borderline given N = 20 items and
ceiling-bound CI).

**H_sota verdict (exploratory, non-confirmatory):** AFT is **not strictly
non-inferior** to Mem0 on this probe (Δ = -0.15, AFT below). AFT remains
non-inferior to LangMem (Δ = -0.05, within the CI of the AFT vs cosine
contrast). The hypothesis "AFT ≥ best of {Mem0, LangMem}" is not supported
on this dataset.

**Latency ranking (encode):** recency < naive_cosine < aft < langmem < mem0.
AFT is ~25× faster on encode than Mem0 (45 ms vs 1130 ms/item) and ~3×
faster than LangMem. On retrieve, AFT is the fastest non-deterministic
system at p50 = 45 ms (vs 240 ms Mem0, 160 ms LangMem).

**Cost dimension (not measured but architectural):** Mem0 and LangMem
issue at least one LLM call per encode and per retrieve. AFT issues zero
LLM calls at runtime (the appraisal engine is optional and was disabled
for this run, AFT operating in oracle-affect mode per `protocol.md`). On
a 1M-event corpus the LLM-call differential dominates total cost.

**Honest trade-off framing:** Mem0 dominates AFT in this probe's quality
metric; AFT dominates Mem0 in encode/retrieve latency and runtime cost.
LangMem sits between them on both axes.

### Coherence with prior closures (re-checked post-result)

- **Hd1/Hd2 (oracle preset, realistic_recall_v2):** AFT advantage on the
  realistic replay benchmark (Δ=+0.21 vs cosine, d=0.49, N=200) is on a
  **different probe** (multi-session realistic recall, 5 challenge types).
  This SOTA result on `affect_reference_v1` (single-quadrant retrieval,
  N=20 items) does not contradict it but does narrow the scope: Mem0/
  LangMem are not in the v2 evaluation, so we cannot infer whether they
  would also outperform AFT there. Recommend a future SOTA replication
  on v2.
- **Hg1 FAIL (Addendum G):** With LLM appraisal, AFT does not beat
  cosine. The current SOTA result is consistent: external LLM-backed
  systems do beat cosine, but AFT (without LLM at runtime) does not match
  them on quality.
- **LoCoMo S1 FAIL:** AFT underperforms naive RAG on factual QA. The
  current SOTA result extends the picture: on affect-aware retrieval too,
  AFT is competitive with cosine but below LLM-backed memory.

### Paper-section action

Add a paragraph in §6 (Benchmark) reporting the SOTA comparison with
honest trade-off framing. Update the abstract's "comparative benchmark
against a semantic-only baseline" wording to reflect that LLM-backed
SOTA was tested and outperforms AFT on this probe at higher latency/cost.

### `claim_validation_matrix.json` action

Update `retrieval_affect_aware` `current_evidence` to mention the SOTA
comparison: AFT recall@5 = 0.85 vs Mem0 1.00 vs LangMem 0.90 on the
controlled probe (encode latency 45 ms vs 1130 ms vs 150 ms). The
existing claim "retrieval is affect-aware" remains valid; the new
qualifier is that affect-aware ranking under hand-crafted scoring does
not match LLM-extracted memory on this metric, but is ~25× cheaper.
