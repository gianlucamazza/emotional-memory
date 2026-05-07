# Pre-registration: SOTA Replication on realistic_recall_v2

**Status:** PENDING_EXECUTION
**Date (pre-reg):** 2026-05-07
**Embedder:** `sbert-bge` (BAAI/bge-small-en-v1.5)
**LLM model (mem0, langmem):** `gpt-4.1-mini`
**Dataset:** `benchmarks/datasets/realistic_recall_v2.json` (N=200 queries, 50 scenarios)
**Parent closure:** `benchmarks/preregistration_addendum_hd2_closure.md` (Hd2 PASS, Δ +0.205, d=0.49)

---

## Motivation

Hd2 established that AFT (oracle-affect mode, sbert-bge) achieves top1_accuracy = 0.530
vs naive_cosine = 0.325 on `realistic_recall_v2` (Δ +0.205 [0.150, 0.265], d=0.49,
N=200, p<0.001). This is the headline result of the paper.

The SOTA extension run (`benchmarks/comparative/protocol.sota.md`, committed 2cd2c91)
showed that on `affect_reference_v1` (a simpler controlled probe), Mem0 outperforms
AFT (1.00 vs 0.85, Δ +0.15) and LangMem is within noise (0.90 vs 0.85). This partial
result leaves open whether LLM-backed memory systems also dominate AFT on the harder
multi-session realistic replay benchmark where the headline effect lives.

This addendum answers: **does AFT's Δ +0.21 advantage over cosine survive against
production LLM-backed memory systems (Mem0, LangMem) on the same benchmark?**

---

## Hypothesis (H_v2_sota, exploratory)

> On `realistic_recall_v2` (sbert-bge, N=200, top_k=2 default), AFT `top1_accuracy`
> is ≥ best of {Mem0, LangMem}.

**Nature:** Exploratory reference, NOT confirmatory.

This is consistent with the framing in `protocol.sota.md` §"Why this is exploratory
reference not definitive" and with the prior SOTA run.

---

## Protocol

**Runner:** `benchmarks.realistic.runner` with `--systems aft,naive_cosine,recency,mem0,langmem`

**Adapter:** `benchmarks/realistic/adapters/sota_shim.py` `ComparativeReplayShim` wrapping
`Mem0Adapter` / `LangMemAdapter` from the comparative adapter library.

**Critical methodological asymmetry:** AFT receives preset `valence`/`arousal` values
from the dataset on both encode and retrieve (oracle-affect mode, consistent with Hd2).
Mem0 and LangMem ignore affect entirely — their adapters do not consume the affect channel.
This asymmetry is the same as in the `affect_reference_v1` SOTA run and must be reported
explicitly in any paper section that cites this result.

**Memory isolation:** All systems accumulate memories across all 50 scenarios within a
single benchmark run (no per-scenario reset), consistent with how AFT and naive_cosine
operate in the existing Hd2 run. The `alias_to_actual` mapping in the runner is per-scenario,
so the hit metric is evaluated correctly against scenario-specific expected IDs.

---

## Statistical handling

- Primary metric: `top1_accuracy` (same as Hd2, `default_top_k=2`)
- Secondary: `hit@k` (top_k=2), encode_ms_avg, retrieve_p50/p95 ms
- Per-challenge breakdown (5 types: semantic_confound, affective_arc, recency_confound,
  same_topic_distractor, momentum_alignment)
- Bootstrap CI: paired percentile, n=10000, seed=0 (consistent with all v2 closures)
- Pairwise comparisons vs naive_cosine: Δ, 95% CI, p_bootstrap, p_mcnemar, d Cohen,
  n_discordant
- **No Holm correction** (exploratory single-family, consistent with `protocol.sota.md`)

---

## Decision rule

**PASS:** AFT top1 ≥ max(mem0.top1, langmem.top1), with overlapping 95% CIs.
AFT's headline advantage resists the LLM-backed SOTA comparison.

**FAIL:** AFT top1 < max(mem0.top1, langmem.top1) with non-overlapping or strongly
overlapping CIs favouring the SOTA system. The headline effect exists vs cosine but
does not establish superiority over LLM-backed memory.

**MIXED:** Per-challenge heterogeneity — AFT wins on affect-specific challenges
(affective_arc, momentum_alignment) but loses on semantic-heavy challenges
(semantic_confound, same_topic_distractor).

Verdict is exploratory; no paper withdrawal or major claim reversal is triggered by
a FAIL. A FAIL narrows the scope: "AFT is competitive with cosine baseline; LLM-backed
systems achieve higher quality at substantially higher latency/cost."

---

## Coherence with prior closures

- **Hd2 PASS (oracle-affect, sbert-bge, v2):** The baseline for this comparison. AFT
  top1 = 0.530, naive_cosine = 0.325. Mem0/LangMem are on top of this dataset first time.
- **Hg1 FAIL (Addendum G, LLM-appraisal, v3_noAF):** AFT + LLMAppraisalEngine does not
  beat cosine. The current run uses AFT in oracle-affect mode (no appraisal engine), so
  the Hg1 FAIL does not apply here. Oracle-affect regime is the controlled test bed.
- **SOTA on affect_reference_v1 (protocol.sota.md):** Mem0 1.00, LangMem 0.90, AFT 0.85
  on the simpler probe. This addendum extends the comparison to the harder multi-session
  benchmark where AFT's architecture-specific advantage (mood, momentum, resonance) should
  theoretically matter more.
- **Hi3 PASS (resonance, d=0.25):** Resonance layer shows marginal benefit on sbert-bge
  within AFT ablation. Does not speak to Mem0/LangMem comparison.

---

## Output files

- `benchmarks/comparative/results.sota.v2.sbert.json` — full run output (produced by runner)
- `benchmarks/comparative/results.sota.v2.sbert.md` — human-readable summary
- `benchmarks/comparative/results.sota.v2.sbert.protocol.json` — execution metadata
- `benchmarks/comparative/protocol.sota.v2.md` — protocol document (this sister file)

Closure: append **Results** section to `benchmarks/comparative/protocol.sota.v2.md` and
update `docs/research/claim_validation_matrix.json`.

---

## Execution command

```bash
EMOTIONAL_MEMORY_LLM_MODEL=gpt-4.1-mini \
uv run python -m benchmarks.realistic.runner \
  --embedder sbert-bge \
  --dataset benchmarks/datasets/realistic_recall_v2.json \
  --systems aft,naive_cosine,recency,mem0,langmem \
  --n-bootstrap 10000 \
  --seed 0 \
  --out-json benchmarks/comparative/results.sota.v2.sbert.json \
  --out-md benchmarks/comparative/results.sota.v2.sbert.md \
  --out-protocol benchmarks/comparative/results.sota.v2.sbert.protocol.json
```
