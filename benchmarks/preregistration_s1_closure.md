# Pre-registration Closure — Study S1 (LoCoMo External Benchmark)

**Date executed:** 2026-04-27
**Protocol version:** s1_closure_v1
**Parent pre-reg:** `benchmarks/preregistration.md` (Study S1, lines 15–103)

> **Epistemic status:** This document formalizes the closure of Study S1.
> All numbers match the committed result files exactly.

---

## Background

Study S1 pre-registered AFT vs naive_RAG on the LoCoMo10 external benchmark
(10 conversations, ~1540 QA pairs). This was the first test of AFT on a
publicly available external benchmark not constructed by the author.

Pre-registered hypotheses (confirmatory):
- **H1 (one-tailed):** AFT token-F1 > naive_RAG token-F1 on full LoCoMo10.
- **H2 (one-tailed):** AFT LLM-judge accuracy > naive_RAG judge accuracy.
- Holm correction across H1 and H2 jointly.
- **Gate 1:** both H1 and H2 must pass Holm correction for a strong claim.

---

## Execution

```bash
set -a && source .env && set +a
make bench-locomo
```

Canonical result files:
- `benchmarks/locomo/results.json`
- `benchmarks/locomo/results.md`
- `benchmarks/locomo/results.protocol.json`

Parameters: seed=0, n_bootstrap=2000, model=gpt-5-mini, N=1986 QA pairs
(scored=1540 after exclusions per §Exclusion criteria).

---

## Results summary

| Metric | AFT | naive_RAG | Δ | p_boot (one-tailed) | p_adj_holm | Verdict |
|---|---|---|---|---|---|---|
| H1: token-F1 | 0.168 | 0.271 | -0.101 | 1.000 | 1.000 | **FAIL** |
| H2: judge_acc | 0.279 | 0.441 | -0.159 | 1.000 | 1.000 | **FAIL** |

**Gate 1: NOT MET.** Both H1 and H2 fail (Δ < 0, p=1.000).

---

## Interpretation

**H1 FAIL, H2 FAIL — both metrics, Gate 1 not met.** AFT retrieval
underperforms naive RAG on factual open-domain conversational QA. The
direction of the gap is negative (AFT is worse, not just non-superior),
indicating that affective weighting actively hurts factual recall when
affect is not a relevant retrieval signal.

This is a pre-registered negative result and is reported as such per
Reporting rule 5. It is not evidence against AFT in affect-discriminative
settings — see the realistic_recall_v2 positive results — but it clearly
establishes scope: AFT's advantage is conditional on affective context
being discriminative. On purely factual QA it is a liability.

**Claim ceiling:** `locomo_external_qa_negative` in
`docs/research/claim_validation_matrix.json` is the canonical record.
`replayable_multi_session_help` is unaffected (different benchmark design).

**Next step:** per-task retrieval-weight tuning (lower mood weight on
factual QA) is identified as open work; no further S1-family study has
been pre-registered.
