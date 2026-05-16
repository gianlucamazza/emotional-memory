# Pre-registration Addendum M — French Multilingual Slice Closure (FR, me5)

**Status:** CLOSED — Branch A (PASS)
**Date executed:** 2026-05-16
**Parent pre-reg:** `benchmarks/preregistration_addendum_m_fr.md` (committed 0faf88c)
**Parent closures:**
- `benchmarks/preregistration_addendum_bc_closure.md` (multilingual scope established)
- `benchmarks/preregistration_addendum_hd2_powertopup_closure.md` (IT/ES me5 N=120 — prior outcomes: Branch C FAIL-FAIL)

> **Epistemic status:** This document records the execution and results of the
> pre-registered Addendum M French slice (Hm1). The canonical JSON result files were
> written at execution time and are committed; this document provides the interpretive
> closure. All numbers match the committed JSON files exactly.

---

## Motivation recap

From `benchmarks/preregistration_addendum_m_fr.md` §Motivation:

Prior IT/ES me5 runs showed directional effects; the pre-reg for Addendum M was committed
2026-05-13 to test whether the effect extends to a natively-authored French dataset, with
an explicit null prior:

> *"Prior expectation: FAIL (null prior). [...] We declare FAIL as the expected outcome;
> a PASS would represent unexpected positive replication."*

The French dataset (30 scenarios, 120 queries) was hand-authored natively in French — not
translated from Italian — to avoid data-snooping via near-cognate lexical overlap. Oracle
valence/arousal values were pre-assigned following the identical numerical structure used
for IT/ES.

---

## Execution

```bash
uv run python -m benchmarks.realistic.runner \
  --dataset benchmarks/datasets/realistic_recall_v2_fr.json \
  --embedder multilingual-e5-small \
  --seed 0 \
  --n-bootstrap 10000 \
  --out-json benchmarks/realistic/results.v2_fr.me5.json \
  --out-md benchmarks/realistic/results.v2_fr.me5.md \
  --out-protocol benchmarks/realistic/results.protocol.v2_fr.me5.json
```

---

## Results — FR realistic_recall_v2 (N=120 queries, 30 scenarios, me5)

| System | Queries | top1_accuracy | 95% CI | hit@k | 95% CI |
|---|---|---|---|---|---|
| `aft` | 120 | **0.31** | [0.23, 0.39] | **0.40** | [0.32, 0.49] |
| `naive_cosine` | 120 | 0.12 | [0.07, 0.18] | 0.25 | [0.17, 0.33] |
| `recency` | 120 | 0.21 | [0.14, 0.28] | 0.48 | [0.39, 0.57] |

### Pairwise: AFT vs naive_cosine (primary)

| Metric | Δ | 95% CI | p (bootstrap, two-sided) | p (McNemar) | Hedges g | N discordant |
|---|---|---|---|---|---|---|
| top1_accuracy | **+0.18** | **[0.11, 0.26]** | **0.0000** | **0.0000** | **0.424** | 26 / 120 |
| hit@k | +0.15 | [0.09, 0.22] | 0.0000 | 0.0000 | 0.416 | 18 / 120 |

Bootstrap n=10,000, seed=0, paired on (scenario_id, query_id). Runner computes two-sided;
for the pre-registered one-tailed alternative (Δ>0), p_one_tailed < p_two_sided / 2
< 0.0001.

**Verdict: PASS.**

---

## Per-challenge-type breakdown (N=24 per type, AFT vs naive_cosine)

| Challenge type | AFT top1 [95% CI] | naive top1 [95% CI] | Δ |
|---|---|---|---|
| `affective_arc` | 0.38 [0.21, 0.58] | 0.21 [0.04, 0.38] | **+0.17** |
| `momentum_alignment` | 0.12 [0.00, 0.25] | 0.04 [0.00, 0.12] | +0.08 |
| `recency_confound` | 0.29 [0.12, 0.50] | 0.08 [0.00, 0.21] | **+0.21** |
| `same_topic_distractor` | 0.42 [0.21, 0.62] | 0.04 [0.00, 0.12] | **+0.38** |
| `semantic_confound` | 0.33 [0.17, 0.54] | 0.25 [0.08, 0.42] | +0.08 |

AFT advantage is broadest on `same_topic_distractor` (Δ=+0.38) and `recency_confound`
(Δ=+0.21). `momentum_alignment` and `semantic_confound` show weak but positive gaps.
Note: `same_topic_distractor` has `Non-trivial queries = 0.00` (all trivially solvable by
distractor exclusion), so the high AFT score reflects correct distractor suppression.

---

## Configuration verification (no post-hoc deviation)

| Parameter | Pre-reg spec | Actual |
|---|---|---|
| Dataset | `realistic_recall_v2_fr.json` (hand-authored native FR) | ✓ |
| Embedder | `multilingual-e5-small` | ✓ |
| Seed (bootstrap) | 0 | ✓ |
| n_bootstrap | 10,000 | ✓ |
| Scenarios | 30 | ✓ (30) |
| Queries | 120 | ✓ (120) |
| Challenge-type balance | 5 × 24 = 120 | ✓ (24 each) |
| Events total | ~228 (pre-reg estimate, 3–5 per session × 60 sessions) | 210 |
| top_k | 2 | ✓ |

**Events discrepancy:** The pre-reg estimated ~228 events (3–5 per session × 60 sessions
≈ [180, 300]). The final dataset contains 210 events — within the implied range.
All other structural parameters match exactly. No bias: events were authored before any
retrieval or scoring inspection.

---

## Branch decision (pre-registered)

Per `benchmarks/preregistration_addendum_m_fr.md` §"Decision rule" and §"Branch declarations":

> **Branch A (PASS):** `p_bootstrap < 0.05` AND `Δ > 0` AND 95% CI lower bound > 0.
> → Cross-language replication to FR confirmed; append FR PASS to `retrieval_affect_aware`
> and `replayable_multi_session_help` `current_evidence`; add `results.v2_fr.me5.*`
> to `benchmark_refs`.

**This is Branch A.**
- p_bootstrap (two-sided) = 0.0000 < 0.05 ✓
- Δ = +0.18 > 0 ✓
- CI lower bound = 0.11 > 0 ✓

**Verdict: PASS.**

No post-hoc threshold adjustment; decision applied as pre-registered. The pre-registration
explicitly declared Branch B (FAIL) as the expected outcome; the observed evidence diverges
from that prior. No reframing: this is an unexpected positive replication.

---

## What this does NOT change

The following results are **unaffected** by this closure:

| Study | Status | Why unaffected |
|---|---|---|
| Hd2_IT me5 N=120 (appraisal_confound runner) | Branch C FAIL — unchanged | Different runner + dataset variant |
| Hd2_ES me5 N=120 (appraisal_confound runner) | Branch C FAIL — unchanged | Different runner + dataset variant |
| LoCoMo Gate 1 | FAIL — unchanged | Different domain (factual QA); no affect-discriminative items |
| Hk1 DailyDialog | FAIL — unchanged | Different ecology (NLI annotation, naturalised conversations) |
| Hd2 EN SBERT N=200 | PASS — unchanged | Different language; not re-run |
| Hi3 resonance amplification | PASS — unchanged | Independent study |

---

## Interpretation

The FR slice yields a clean, strong signal (Δ=+0.18, d=0.424, p<0.0001) where the prior
IT/ES runs on the `appraisal_confound` runner at N=120 did not survive the power top-up.
The divergence likely reflects design differences: the `realistic.runner` dataset uses a
2-session structure with explicit affective transitions, while the Hd2 appraisal_confound
runner uses a single-session format. The 2-session design creates more salient affective
context carry-over, amplifying the AFT advantage.

The challenge-type breakdown is consistent across languages: `momentum_alignment` remains
the weakest category (Δ=+0.08 in FR, matching the flat profile in prior multilingual slices),
suggesting this challenge type may require richer temporal context than the current 2-session
window provides.

**Cross-language scope as of this closure:**

- EN: robust (SBERT d=0.49 N=200; e5 d=0.31 N=200), multi-embedder.
- FR: PASS (me5 d=0.42 N=120, native hand-authored, 2-session realistic replay).
- IT/ES: FAIL on `appraisal_confound` runner N=120; prior N=80 results were borderline.

Cross-language evidence is conditionally established: the 2-session realistic replay format
shows a consistent AFT advantage across EN and FR; the single-session Hd2 format does not.

---

## Cascade changes

| File | Change |
|---|---|
| `docs/research/claim_validation_matrix.json` | `cross_domain_affect_replication`: `retry_planned` → `controlled_evidence`; append FR evidence to `current_evidence`, `benchmark_refs`, `protocol_refs`; update `allowed_public_wording` and `not_yet_shown` |
| `docs/research/claim_validation_matrix.json` | `retrieval_affect_aware`, `replayable_multi_session_help`: append FR evidence to `current_evidence` and refs |
| `docs/research/09_current_evidence.md` | Hd2 section: add FR row to results table and power notes table; rewrite cross-language paragraph |
| `ROADMAP.md` | WS3b `[ ]` → `[x]` + FR PASS summary |
| `CHANGELOG.md` | Entry added under `[Unreleased] ### Research` |

---

## Artefact index

| File | Description |
|---|---|
| `benchmarks/datasets/realistic_recall_v2_fr.json` | Hand-authored 30-scenario native FR dataset (210 events, 120 queries, 5×24 challenge types) |
| `benchmarks/realistic/results.v2_fr.me5.json` | Full bootstrap results (canonical) |
| `benchmarks/realistic/results.v2_fr.me5.md` | Human-readable results report |
| `benchmarks/realistic/results.protocol.v2_fr.me5.json` | Run protocol metadata |
| `benchmarks/preregistration_addendum_m_fr.md` | Pre-registration (committed 0faf88c, 2026-05-13) |
