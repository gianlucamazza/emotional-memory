# Pre-registration Addendum — Hd2 Power Top-Up Closure (IT + ES, N=120)

**Status:** CLOSED — Branch C (FAIL-FAIL)
**Date executed:** 2026-05-07
**Parent pre-reg:** `benchmarks/preregistration_addendum_hd2_powertopup.md` (committed 4b42dae)
**Parent closure (N=80):** `benchmarks/preregistration_addendum_hd2_closure.md`

> **Epistemic status:** This document records the execution and results of the
> pre-registered Hd2 power top-up (Addendum Hd2-PowerTopUp). The canonical JSON
> result files were written at execution time and are committed; this document provides
> the interpretive closure. All numbers match the committed JSON files exactly.

---

## Motivation recap

From `benchmarks/preregistration_addendum_hd2_closure.md`:

- **Hd2_IT (me5, N=80):** Δ=+0.1625, p_raw=0.012 / p_bonf×5=0.060 (borderline FAIL at family level), d=0.290.
- **Hd2_ES.me5 (N=80):** Δ=+0.1125, p=0.110, d=0.189. Closure note: *"d=0.189 requires N≈120 for 80% power"*.

The pre-reg for this top-up was committed before dataset generation (4b42dae, 2026-05-07)
to prevent data-snooping. Datasets were extended from 20 → 30 scenarios (+10 per language)
via `scripts/generate_realistic_v2_topup.py` (LLM-assisted, valence/arousal programmatic,
balanced challenge-type rotation). Pre-reg commited, datasets committed, then benchmarks run.

---

## Execution

```bash
uv run python -m benchmarks.appraisal_confound.runner \
  --embedder multilingual-e5-small \
  --dataset benchmarks/datasets/realistic_recall_v2_it.json \
  --out-json benchmarks/appraisal_confound/results.hd2_it.me5.v120.json \
  --out-md benchmarks/appraisal_confound/results.hd2_it.me5.v120.md \
  --out-protocol benchmarks/appraisal_confound/results.hd2_it.me5.v120.protocol.json

uv run python -m benchmarks.appraisal_confound.runner \
  --embedder multilingual-e5-small \
  --dataset benchmarks/datasets/realistic_recall_v2_es.json \
  --out-json benchmarks/appraisal_confound/results.hd2_es.me5.v120.json \
  --out-md benchmarks/appraisal_confound/results.hd2_es.me5.v120.md \
  --out-protocol benchmarks/appraisal_confound/results.hd2_es.me5.v120.protocol.json
```

---

## Results — Hd2_IT_v120 (Italian, me5, N=120)

| System | N | top1_accuracy | 95% CI |
|---|---|---|---|
| `aft_noAppraisal` | 120 | **0.333** | [0.250, 0.417] |
| `naive_cosine` | 120 | 0.275 | [0.200, 0.358] |

| Metric | Value |
|---|---|
| Δ (AFT − cosine) | **+0.058** |
| 95% CI (bootstrap) | **[-0.042, +0.158]** |
| p (two-sided bootstrap, n=10000, seed=42) | 0.276 |
| Cohen's d | 0.105 |

**Verdict: FAIL.** CI crosses zero; p=0.276 >> 0.05; decision rule `Δ>0 AND CI not crossing 0 AND p_holm<0.05` not met.

---

## Results — Hd2_ES_v120 (Spanish, me5, N=120)

| System | N | top1_accuracy | 95% CI |
|---|---|---|---|
| `aft_noAppraisal` | 120 | **0.267** | [0.192, 0.350] |
| `naive_cosine` | 120 | 0.267 | [0.192, 0.350] |

| Metric | Value |
|---|---|
| Δ (AFT − cosine) | **0.000** |
| 95% CI (bootstrap) | **[-0.100, +0.100]** |
| p (two-sided bootstrap) | 1.000 |
| Cohen's d | 0.000 |

**Verdict: FAIL.** Exact null at N=120; CI precisely symmetric around zero.

---

## N=80 vs N=120 comparison (me5, pre-declared power target)

| Language | N=80 Δ | N=80 p | N=80 d | N=120 Δ | N=120 p | N=120 d | Branch |
|---|---|---|---|---|---|---|---|
| Italian | +0.163 | 0.012 | 0.290 | **+0.058** | 0.276 | 0.105 | C |
| Spanish | +0.113 | 0.110 | 0.189 | **0.000** | 1.000 | 0.000 | C |

The Italian N=80 borderline result (already marginal at Bonf×5=0.060) does not hold at
the pre-declared power target. The effect size collapses by −0.104 Δ-units. The Spanish
N=80 result was already a FAIL; N=120 confirms exact null.

---

## Branch decision (pre-registered)

Per `benchmarks/preregistration_addendum_hd2_powertopup.md` §"Branch declarations":

> **Branch C (FAIL-FAIL):** Effect does not survive power top-up. §6 reports both as
> FAIL at N=120; cross-language evidence not established; Remove or caveat cross-language
> claim; flag as limitation.

**This is Branch C.** No post-hoc threshold adjustment; decision applied as pre-registered.

---

## What this does NOT change

The following results are **unaffected** by this closure:

| Study | Status | Why unaffected |
|---|---|---|
| Hd2 EN (SBERT, N=200, Δ=+0.205, d=0.49) | **PASS — unchanged** | Different dataset, embedder, language; not re-run |
| Hd2 EN (e5-small-v2, N=200, Δ=+0.16, d=0.31) | **PASS — unchanged** | Same |
| Hd2_ES SBERT (N=80, Δ=+0.138, p=0.045, d=0.233) | **PASS at N=80 — unchanged** | Different embedder; pre-reg covers only me5 top-up |
| H_v2_sota (AFT vs Mem0/LangMem, v2) | **PASS — unchanged** | Independent study |
| Hg1 FAIL (LLM appraisal) | **FAIL — unchanged** | Independent study |
| Hi3 PASS (resonance amplification) | **PASS — unchanged** | Independent study |
| S3 Ha/Hb FAIL (ablation) | **FAIL — unchanged** | Independent study |

---

## Interpretation

The N=80 Italian effect (Δ=+0.163, p=0.012) was driven by bootstrap variance at modest
N: with d=0.290 at N=80, power is ~83% but the confidence interval is wide ([0.038, 0.288]).
The +10 LLM-generated scenarios introduced additional challenge-scenario variance; the
combined N=120 delta is +0.058, consistent with a true effect of d≈0.10 or smaller.

No post-hoc mechanistic explanation is warranted — the pre-reg declared that Branch C is
"honest and publishable." The headline EN effect (d=0.49, N=200, robust across both
embedder families) is unaffected.

**Cross-language scope as of this closure:**

- EN: robust, multi-embedder (SBERT d=0.49, e5 d=0.31), N=200.
- Spanish (SBERT N=80, p=0.045, d=0.233): a single borderline positive result, not
  power-replicated. Reported as exploratory; sufficient for "directional positive" only.
- Italian (me5 N=80): borderline null originally; confirmed null at N=120.
- Spanish (me5 N=120): confirmed null.

The paper's §6/§7/§8 and all supporting docs are updated accordingly in the same commit
(see cascade changes below).

---

## Cascade changes

| File | Change |
|---|---|
| `paper/main.tex` | §6 multilingual rewritten (Branch C framing); §7 conclusion period updated; §8 limitations updated; cross-language claim rescoped to "EN only + exploratory ES-SBERT N=80" |
| `docs/research/claim_validation_matrix.json` | `retrieval_affect_aware.current_evidence` updated with N=120 FAIL; `next_study` P2 entry replaced with "robust cross-language replication" |
| `docs/research/09_current_evidence.md` | Hd2 table: rows Hd2_IT_v120 (FAIL) + Hd2_ES_v120 (FAIL) added; power table updated |
| `docs/research/12_multilingual_followup.md` | "Branch C closure" section added; conclusions revised |
| `docs/research/08_limitations.md` | Language scope updated: IT FAIL at N=120; ES.me5 FAIL; ES.sbert single positive |
| `README.md` | Cross-language validation section updated |
| `ROADMAP.md` | P2-1 marked EXECUTED — Branch C |
| `CHANGELOG.md` | Entry added under Unreleased |

---

## Artefact index

| File | Description |
|---|---|
| `benchmarks/appraisal_confound/results.hd2_it.me5.v120.json` | IT N=120 run output (JSON) |
| `benchmarks/appraisal_confound/results.hd2_it.me5.v120.md` | IT N=120 run output (human-readable) |
| `benchmarks/appraisal_confound/results.hd2_it.me5.v120.protocol.json` | IT N=120 execution metadata |
| `benchmarks/appraisal_confound/results.hd2_es.me5.v120.json` | ES N=120 run output (JSON) |
| `benchmarks/appraisal_confound/results.hd2_es.me5.v120.md` | ES N=120 run output (human-readable) |
| `benchmarks/appraisal_confound/results.hd2_es.me5.v120.protocol.json` | ES N=120 execution metadata |
| `benchmarks/datasets/realistic_recall_v2_it.json` | Extended IT dataset (30 scenarios, N=120) |
| `benchmarks/datasets/realistic_recall_v2_es.json` | Extended ES dataset (30 scenarios, N=120) |
| `benchmarks/datasets/realistic_recall_v2_topup_provenance.jsonl` | LLM call audit trail for +10 scenarios |
| `scripts/generate_realistic_v2_topup.py` | Generation script (LLM-assisted, valence/arousal programmatic) |
| `benchmarks/preregistration_addendum_hd2_powertopup.md` | Pre-registration (committed before execution) |
