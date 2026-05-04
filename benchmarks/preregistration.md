# Pre-Registration: AFT Scientific Validation Studies

**Date:** 2026-04-24
**Status:** OPEN — no benchmark data has been collected under this pre-registration.

This document freezes hypotheses, metrics, statistical analysis plans, and exclusion
criteria **before** any study data is collected. Once data collection begins for a
study, its section must not be modified. New sections may be appended for future
studies, provided they are added before data collection for those studies begins.

Commit hash at time of pre-registration: see `git log --follow benchmarks/preregistration.md`

---

## Study S1 — LoCoMo External Benchmark

### Background

LoCoMo (Maharana et al. 2024) is a long-context multi-session conversational memory
benchmark: 10 conversations of 35–60 turns each, ~1540 QA pairs, covering single-hop,
multi-hop, temporal, open-domain, and adversarial question types.

**This will be the first execution of LoCoMo under this pre-registration.** No LoCoMo
results exist in the repository at time of writing.

### Systems

- `aft` — `AFTLoCoMoAdapter` with `bge-small-en-v1.5` embedder +
  `KeywordAppraisalEngine` (no LLM at encode time). Defined in
  `benchmarks/locomo/adapters/aft.py`.
- `naive_rag` — `NaiveRAGLoCoMoAdapter` with `bge-small-en-v1.5`, pure cosine
  retrieval, no affective state. Defined in `benchmarks/locomo/adapters/naive_rag.py`.

### Hypotheses

**H1 (confirmatory, one-tailed):** AFT token-F1 > NaiveRAG token-F1 on the full
LoCoMo10 dataset.
- Direction: `aft_F1 − naive_F1 > 0`
- Metric: mean token-overlap F1 as defined in `benchmarks/locomo/scoring.py:25-37`
- Test: paired bootstrap (n_bootstrap=2000, seed=0); one-tailed p < 0.05
- Theoretical basis: affective weighting (mood congruence + resonance) should
  preferentially surface emotionally coherent conversational context, increasing
  the proportion of correct tokens in answer spans.

**H2 (confirmatory, one-tailed):** AFT LLM-judge accuracy > NaiveRAG LLM-judge
accuracy on the full LoCoMo10 dataset.
- Direction: `aft_judge_acc − naive_judge_acc > 0`
- Metric: proportion of QA pairs judged CORRECT by `gpt-5-mini` (temp=0) per
  `benchmarks/locomo/scoring.py:94-122`. Same model is used for both answer
  generation (per-system adapter) and judging, as configured by
  `EMOTIONAL_MEMORY_LLM_MODEL` at run time.
- Test: paired bootstrap (n_bootstrap=2000, seed=0) + McNemar two-sided;
  one-tailed bootstrap p < 0.05
- Holm-Bonferroni correction applied across H1 and H2 jointly.

**H2-exploratory (non-confirmatory):** AFT outperforms NaiveRAG more on multi-hop
and temporal question types than on single-hop types. This analysis will be conducted
after primary results and labeled as exploratory; it does not contribute to
confirmatory claims.

### Primary metric

Token-overlap F1 (H1). LLM judge accuracy (H2) is co-primary but secondary if
results diverge.

### Exclusion criteria

Excluded from scoring (not from analysis set; flagged in output):
1. QA pairs where the gold answer string is empty or None.
2. QA pairs where the LLM judge response does not parse to a valid JSON object with
   a `"label"` field. These are excluded from H2 judge_acc calculation only; they
   still contribute to H1 F1.
3. No conversation-level exclusions are pre-specified; all 10 conversations are
   included.

Do NOT exclude QA pairs post-hoc based on system performance (e.g., both systems
wrong, F1 outliers). All non-excluded pairs are used.

### Statistical analysis

- Paired bootstrap: n=2000, seed=0, percentile method, two-tailed CI, one-tailed p.
- McNemar exact two-tailed for binary outcomes (judge_acc).
- Holm-Bonferroni correction across H1 and H2 using the two-tailed p-values from
  bootstrap (even though directional tests are primary) for conservative reporting.
- Report: mean score, 95% CI, Δ (aft − naive), 95% CI of Δ, p (bootstrap one-tailed),
  p_adj (Holm), discordant pairs (McNemar).

### Execution

```bash
# .env provides: EMOTIONAL_MEMORY_LLM_API_KEY, EMOTIONAL_MEMORY_LLM_MODEL=gpt-5-mini,
#                EMOTIONAL_MEMORY_LLM_BASE_URL=https://api.openai.com/v1
set -a && source .env && set +a
make bench-locomo
```

**Frozen model:** `gpt-5-mini` (OpenAI) is used for both answer generation and LLM
judging under this pre-registration. Any change to this model invalidates H1/H2 for
comparability with other LoCoMo published evals. If re-runs with a different model
are needed for robustness, they are reported as *exploratory*.

Results written to `benchmarks/locomo/results.{json,md,protocol.json}`.

---

## Study S2 — Realistic Recall v2 (Expanded Dataset)

### Background

`realistic_recall_v1` has been expanded to v1.4: 50 scenarios, 100 queries,
sbert-bge embedder. Current aggregate: AFT top1=0.70 vs naive 0.50 (N=100).
`semantic_confound` subset (N=30): AFT top1=0.73 vs naive 0.47, Δ=+0.27
[0.10, 0.43], p_adj=0.006 — first per-challenge result to survive Holm correction.
S2 targets `realistic_recall_v2` for the pre-registered confirmatory run
at N≥200 (see dataset construction rules below).

**Power analysis:** to detect Δ=0.10 at α=0.05 (one-tailed), β=0.80 with McNemar
on binary paired outcomes, the required discordant pairs is approximately 43. With
an expected discordance rate of ~20%, this requires N≈215 queries. To be conservative
and account for per-challenge-type sub-analyses, target **N=200 queries across ≥50
scenarios** (4 queries/scenario average).

### Dataset construction rules (pre-specified)

Scenarios for `realistic_recall_v2.json` must satisfy all of:
1. Each scenario has exactly 1 target memory and ≥5 distractor memories.
2. Each scenario is labeled with exactly one challenge type (see list below).
3. Scenarios are stratified across challenge types with no type accounting for
   more than 35% of the total (prevents a single type from dominating overall
   results).
4. Scenarios are constructed before running any benchmark; no scenario is added
   or modified after the first benchmark run on the v2 dataset.

**Challenge types** (5 types, each must have N≥20 queries in v2):
- `semantic_confound` — distractors share topic/keywords with target but differ
  in affect
- `affective_arc` — correct memory is earlier in time but more emotionally salient
- `recency_confound` — most recent memory is NOT the correct answer; requires
  affective weighting to bypass recency bias
- `same_topic_distractor` — distractors share high cosine similarity to query
- `momentum_alignment` — query affect matches the affective momentum built across
  prior turns; naive cosine lacks this signal

### Hypotheses

**H3 (confirmatory, one-tailed):** AFT top1_accuracy > naive_cosine top1_accuracy
on the full v2 query set.
- Direction: `aft_top1 − naive_top1 > 0`
- Metric: top1_accuracy (proportion where rank-1 retrieved memory = target)
- Test: paired bootstrap (n=2000, seed=0) one-tailed p < 0.05 + McNemar two-tailed
  p < 0.05; both must be significant for confirmation.
- Theoretical basis: multi-signal scoring (mood congruence, momentum, resonance)
  should outperform pure cosine when emotional context discriminates memories.

**H4 (confirmatory, one-tailed, Holm-corrected):** AFT top1 > naive_cosine on
`affective_arc` challenge type. Theory: mood field privileges emotionally charged
older memories over neutral recent ones.

**H5 (confirmatory, one-tailed, Holm-corrected):** AFT top1 > naive_cosine on
`recency_confound` challenge type. Theory: recency weight is one of 6 signals;
affective salience can override it when mood/resonance signal is strong.

**H6 (confirmatory, one-tailed, Holm-corrected):** AFT top1 > naive_cosine on
`momentum_alignment` challenge type. Theory: `AffectiveMomentum` provides a unique
signal absent from naive cosine; should yield largest advantage on this type.

**H7–H8 (exploratory):** per-type results for `semantic_confound` and
`same_topic_distractor` are analyzed but not pre-registered as confirmatory. They
inform paper discussion.

**Holm family for S2:** H3, H4, H5, H6 are corrected jointly (4 tests).

### Exclusion criteria

Same as S1: no post-hoc scenario exclusions. Scenarios constructed before running.
Flag (do not exclude) scenarios where both systems score 0 (trivially hard) or both
score 1 (trivially easy) — report these counts separately.

### Statistical analysis

Same bootstrap/McNemar/Holm plan as S1. Additionally: report Hedges g (paired) for
each confirmatory comparison. Report N, discordant pairs per sub-analysis.

### Execution

```bash
make bench-realistic  # after realistic_recall_v2.json is committed
make bench-realistic-ablation  # for M2.2
```

---

## Study S3 — Layer Ablation v2 (Powered)

### Background

Ablation v1 on N=20 queries found no individually significant layer contribution after
Holm correction (`benchmarks/ablation/results.sbert.md`). The study was explicitly
underpowered. S3 re-runs on `realistic_recall_v2` (N≥200 queries).

### Hypotheses

**Ha (confirmatory, one-tailed):** Removing the MoodField layer (`no_mood`) reduces
top1_accuracy vs full AFT.
- Direction: Δ = `no_mood_top1 − full_top1 < 0`
- Theory: mood congruence is one of 6 retrieval signals; `no_mood` zeroes it out,
  degrading mood-dependent retrievals.
- Expectation: this is the most likely layer to show significance given v1 Hedges
  g = −0.312 even at N=20.

**Hb (confirmatory, one-tailed):** Removing the ResonanceLink layer (`no_resonance`)
reduces top1_accuracy vs full AFT.
- Direction: Δ = `no_resonance_top1 − full_top1 < 0`
- Theory: spreading activation over associative links enriches the retrieval set;
  without it, episodic associations are lost.
- Expectation: effect may be smaller than Mood; labeled confirmatory because the
  theory is unambiguous.

**Hc (invariant check):** Removing `no_appraisal` produces Δ ≈ 0. This is a
methodological invariant: the realistic benchmark injects affect directly via
`set_affect()` and does not configure an appraisal engine; the flag should be a
no-op. Failure here indicates a bug.

**Hd (exploratory):** `no_momentum` effect direction and magnitude. Theory predicts
small contribution (momentum is a 3-point velocity signal; smooth scenarios may not
stress it). Not pre-registered as confirmatory; result informs future design.

**Holm family for S3:** Ha and Hb are corrected jointly (2 tests, one-tailed
bootstrap p-values).

### Execution

```bash
make bench-ablation  # after realistic_recall_v2.json is committed
```

Results written to `benchmarks/ablation/results.v2.{json,md}`.

---

## Reporting rules (all studies)

1. **Confirmatory/exploratory labeling:** every result table must be labeled as
   (confirmatory) or (exploratory). Results labeled exploratory cannot be promoted
   to confirmatory claims post-hoc.
2. **No selective reporting:** all pre-registered hypotheses are reported regardless
   of outcome. Non-significant confirmatory results are reported as such, not omitted.
3. **No re-seeding:** seed=0 throughout. No alternative seeds are tested after
   results are known.
4. **No post-hoc subsetting:** results are not reported only for a subset of data
   found to be significant unless the subset was pre-specified here.
5. **Negative results are results:** if Ha or Hb are not confirmed, the interpretation
   is "insufficient evidence that [mood/resonance] contributes at N=200, Δ_min=0.10"
   — this is a publishable finding, reported as such.
6. **Paper claims:** the arXiv paper and any workshop submission will use only
   confirmatory results for primary claims. Exploratory results are presented as
   "preliminary" or in supplementary material.

---

## S3 Closure — 2026-05-04

Executed at power (N=200, realistic_recall_v2, seed=0, n_bootstrap=2000).

| Hypothesis | Verdict | Embedder | Δ (top1) | 95% CI | p_boot | p_adj_holm |
|---|---|---|---|---|---|---|
| Ha (no_mood < full) | **FAIL** | SBERT | -0.02 | [-0.05, 0.01] | 0.264 | 1.000 |
| Ha (no_mood < full) | **FAIL** | e5 | -0.005 | [-0.05, 0.04] | 0.915 | 1.000 |
| Hb (no_resonance < full) | **FAIL** | SBERT | +0.02 | [-0.01, 0.05] | 0.203 | 1.000 |
| Hb (no_resonance < full) | **FAIL** | e5 | +0.085 | [0.04, 0.13] | 0.000 | 0.000 |
| Hc (no_appraisal ≈ full) | **PASS** | SBERT | -0.01 | [-0.03, 0.00] | 0.283 | — |
| Hc (no_appraisal ≈ full) | **PASS** | e5 | +0.005 | [-0.03, 0.04] | 0.880 | — |
| Hd (no_momentum, exploratory) | NS | SBERT | +0.02 | [0.01, 0.04] | 0.067 | — |
| Hd (no_momentum, exploratory) | NS | e5 | 0.00 | [-0.03, 0.03] | 1.000 | — |

**Canonical result files:**
- `benchmarks/ablation/results.v2.sbert.json` (SBERT BAAI/bge-small-en-v1.5)
- `benchmarks/ablation/results.v2.e5.json` (intfloat/e5-small-v2)

**Interpretation:** Ha and Hb both FAIL on both embedders. Removing mood or
resonance in isolation does not reduce top1_accuracy at N=200. The resonance layer
shows an anomalous positive effect with e5 (Hb FAIL in the opposite direction,
Δ=+0.085, p<0.001), suggesting the spreading-activation mechanism may interfere
with e5's embedding geometry. Hc (no_appraisal invariant) PASSES on both: the
benchmark correctly does not configure an appraisal engine. The system-level
architecture advantage (Hd1/Hd2 vs naive cosine, Δ=0.13–0.23) is orthogonal to
this per-layer ablation: the AFT advantage appears to be a system-level emergent
property rather than attributable to any single layer in isolation.

This is a pre-registered negative result and is reported as such per Reporting
rule 5 above.

---

## S2 Closure — 2026-05-04

Executed at power (N=200, realistic_recall_v2, seed=0, n_bootstrap=2000).
Full closure: `benchmarks/preregistration_s2_closure.md`.

**H3 (overall, both embedders):** PASS.

| Hypothesis | Type | Embedder | Verdict | Δ (top1) | p_adj_holm |
|---|---|---|---|---|---|
| H3 (overall) | confirmatory | SBERT | **PASS** | +0.205 | <0.001 |
| H3 (overall) | confirmatory | e5 | **PASS** | +0.155 | <0.001 |
| H4 (affective_arc) | confirmatory | SBERT | **PASS** | +0.275 | 0.000 |
| H4 (affective_arc) | confirmatory | e5 | **PASS** | +0.275 | 0.008 |
| H5 (recency_confound) | confirmatory | SBERT | **FAIL** (p_adj=0.054) | +0.100 | 0.054 |
| H5 (recency_confound) | confirmatory | e5 | **PASS** | +0.200 | 0.040 |
| H6 (momentum_alignment) | confirmatory | SBERT | **PASS** | +0.275 | 0.000 |
| H6 (momentum_alignment) | confirmatory | e5 | **FAIL** (p_adj=0.811) | +0.025 | 0.811 |

**Canonical result files:**
- `benchmarks/realistic/results.v2.sbert.json` (overall, SBERT)
- `benchmarks/realistic/results.v2.e5.json` (overall, e5)
- `benchmarks/realistic/challenge_subset_pairwise_v2.json` (per-challenge Holm)

**Interpretation:** H3 (overall) and H4 (affective_arc) are confirmed on both
embedders. H5 and H6 are embedder-dependent: SBERT shows the theoretically predicted
advantages for recency_confound and momentum_alignment; e5-small-v2 shows the
advantage only for recency_confound (H5 PASS) while momentum_alignment (H6) fails
entirely. The architecture-level advantage is consistent (H3/H4); per-type advantages
are geometry-dependent. Hd2 (N=200) independently confirms the system-level
advantage; S2 and Hd2 jointly support the overall AFT claim.
