# Closure — Addendum R (A3): Downstream answer quality (encode→retrieve→generate→judge)

**Status:** EXECUTED 2026-06-26 · **Hr1 PASS · Hr2 PASS**
**Pre-registration:** `preregistration_addendum_r_downstream.md` (committed before execution)
**Runner:** `benchmarks/downstream/runner.py` · **Artifact:** `benchmarks/downstream/results.{json,md}`
**Config:** `realistic_recall_v2` (N=200 queries), `sbert-bge`, judge+generator `gpt-5-mini`,
paired bootstrap n=2000, seed=42.

## Result

| Hyp     | Metric        |   AFT | cosine | Δ [95% CI]                  |      p | p_holm | McNemar | Verdict     |
| ------- | ------------- | ----: | -----: | --------------------------- | -----: | -----: | ------: | ----------- |
| **Hr1** | judge_correct | 0.595 |  0.440 | **+0.155 [+0.095, +0.220]** | <0.001 | <0.001 |  <0.001 | **✅ PASS** |
| **Hr2** | token-F1      | 0.493 |  0.341 | **+0.152 [+0.100, +0.205]** | <0.001 | <0.001 |  <0.001 | **✅ PASS** |

Retrieval-ranking reference (not a hypothesis): AFT top1 0.530 vs cosine 0.325,
Δ=+0.205 [+0.150, +0.265] — matches the published Hd2 ranking advantage.

### By challenge type (judge accuracy)

| challenge_type        |   N |   AFT | cosine |      Δ |
| --------------------- | --: | ----: | -----: | -----: |
| same_topic_distractor |  40 | 0.875 |  0.675 | +0.200 |
| momentum_alignment    |  40 | 0.575 |  0.375 | +0.200 |
| semantic_confound     |  40 | 0.775 |  0.625 | +0.150 |
| affective_arc         |  40 | 0.450 |  0.300 | +0.150 |
| recency_confound      |  40 | 0.300 |  0.225 | +0.075 |

The advantage is positive across all five challenge types; weakest on
`recency_confound` (where a recency baseline is hardest to beat).

## Interpretation

The retrieval-ranking edge **converts to downstream answer quality**: a +0.205
top-1 ranking advantage yields a +0.155 LLM-judged answer-accuracy advantage —
a near 1:1 conversion. This is the first end-to-end (encode→retrieve→generate→judge)
evidence that AFT's ranking gains are not cosmetic.

**Scope (critical).** This holds **only in the affect-discriminative regime** that
defines `realistic_recall_v2`: oracle valence/arousal at encode **and** query-time
state injection (the A2 boundary). It is **not** a production / automatic-appraisal
claim, and it does **not** contradict the oracle-free LoCoMo result, where AFT
_loses_ downstream (judge 0.279 vs 0.441). Both are true and complementary:

- **affect-discriminative + oracle affect** (this study) → AFT beats cosine downstream.
- **naturalistic + automatic affect** (LoCoMo) → AFT loses downstream.

## Claim-matrix impact

Adds `downstream_value` = `early_controlled_evidence`, `requires_oracle_affect=true`,
evidence level `4_realistic_tasks`. Allowed public wording: _"In the
affect-discriminative regime (oracle affect), AFT's retrieval-ranking advantage
converts to higher LLM-judged answer quality (+0.155, N=200); no downstream
benefit is shown in the naturalistic, automatic-appraisal regime (LoCoMo)."_
