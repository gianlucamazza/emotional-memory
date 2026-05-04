# Pre-registration Addendum — Study A Closure (Appraisal Confound)

**Date executed:** 2026-05-04
**Protocol version:** addendum_a_closure_v1
**Parent pre-reg:** `benchmarks/preregistration_addendum_v2.md` (Addendum A, lines 12–60)

> **Epistemic status:** This document records the execution and results of the
> pre-registered Addendum A hypotheses Ha2 and Hb2. All numbers match the committed
> JSON files exactly.

---

## Background

Addendum A tests whether the AFT architecture advantage is attributable to the
retrieval architecture itself, or to the richer affect signals injected by an
appraisal engine (LLM or keyword) versus none.

The study compares three systems on `realistic_recall_v1` (v1.4, 50 scenarios,
N=100 queries):
- `aft_noAppraisal` — AFT with preset affect from scenario data; isolates architecture.
- `aft_keyword` — AFT + `KeywordAppraisalEngine`; minimum viable appraisal signal.
- `naive_cosine` — pure cosine baseline.

Ha2 asks whether keyword appraisal is sufficient to beat naive cosine (functional
architecture value). Hb2 asks whether keyword appraisal is equivalent to preset
affect (architectural interchangeability). Both are one-tailed, no Holm correction
(single primary hypothesis per pre-reg).

---

## Execution

```bash
make bench-appraisal-confound   # v1, sbert-bge, n_bootstrap=10000, seed=42
```

Canonical result files:
- `benchmarks/appraisal_confound/results.confirmatory.json` (Hd1 + Ha2 + Hb2, v1 primary)
- `benchmarks/appraisal_confound/results.md` (summary)

Parameters: seed=42, n_bootstrap=10,000, one-tailed α=0.05.

---

## Results summary

### Primary run (realistic_recall_v1, SBERT bge-small-en-v1.5, N=100)

| System | top1_accuracy | 95% CI |
|--------|--------------|--------|
| `aft_noAppraisal` | 0.780 | [0.700, 0.860] |
| `aft_keyword` | 0.160 | [0.090, 0.230] |
| `naive_cosine` | 0.550 | [0.460, 0.650] |

| Hypothesis | Verdict | Δ [95% CI] | Cohen's d |
|---|---|---|---|
| **Ha2** (aft_keyword > naive_cosine) | **FAIL** | −0.390 [−0.50, −0.29] | −0.736 |
| **Hb2** (aft_keyword ≈ aft_noAppraisal) | **FAIL** | −0.620 [−0.72, −0.52] | −1.271 |

### Replications (secondary, not pre-registered at Addendum A level)

Ha2/Hb2 are also computed in subsequent appraisal confound runs on other datasets
(reported for completeness — not confirmatory for this addendum):

| Dataset | Embedder | Ha2 Δ | Ha2 verdict | Hb2 Δ | Hb2 verdict |
|---------|----------|-------|-------------|-------|-------------|
| realistic_recall_v1 (primary) | SBERT | −0.390 | FAIL | −0.620 | FAIL |
| realistic_recall_v2 (EN) | SBERT | — | — | — | — |
| realistic_recall_v2_es | SBERT | −0.188 | FAIL | −0.325 | FAIL |
| realistic_recall_v2_es | me5 | −0.200 | FAIL | −0.312 | FAIL |

---

## Interpretation

**Ha2 FAIL.** Keyword appraisal hurts retrieval on this dataset. `aft_keyword` scores
0.160 vs `naive_cosine` at 0.550 (Δ = −0.39). The KeywordAppraisalEngine maps natural
language to affect in a way that is noisy enough to degrade the architecture's retrieval
signal. This confirms G3 (appraisal confound identified early) and the interpretation
that keyword appraisal is a destructive override on these scenarios.

**Hb2 FAIL.** Keyword appraisal is not equivalent to preset affect. The gap
(Δ = −0.62, d = −1.27) is large and highly significant. Preset affect from scenario
ground truth is far more informative than keyword-inferred affect. The keyword engine
introduces noise that fundamentally changes retrieval behavior.

**Interpretation context.** The FAIL verdicts for Ha2/Hb2 do not undermine the core
architecture claims (Hd1/Hd2 PASS). The `aft_noAppraisal` condition (preset ground-truth
affect) confirms the architecture advantage (Δ = +0.23 vs naive_cosine, Hd1 PASS).
The Ha2/Hb2 FAILs identify that the *quality of appraisal* is a critical bottleneck:
a perfect oracle (preset) confirms architecture value; a noisy keyword engine destroys it.

**Limitation: oracle-affect circularity.** The `aft_noAppraisal` advantage (Hd1/Hd2)
uses preset valence/arousal from the benchmark dataset, which was hand-crafted with AFT
theory in mind. This means Hd* numbers measure "oracle-affect AFT vs non-affect naive",
not "automatic-appraisal AFT vs naive". The architecture advantage under automatic
appraisal remains unconfirmed. This is disclosed in `docs/research/08_limitations.md`
and the `claim_validation_matrix.json` `not_yet_shown` fields.

**Reporting rule 5 (pre-reg):** Ha2 and Hb2 FAIL are reported as publishable null
findings — they identify the appraisal quality bottleneck that motivates Addendum G
(LLM appraisal dual-path future study).
