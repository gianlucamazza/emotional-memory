# Pre-Registration Addendum P — Hg1 Re-run with Recalibrated SEC→Affect Mapping (Hp1)

**Status:** PENDING EXECUTION
**Date written:** 2026-05-31
**Protocol version:** addendum_p_v1
**Parent / prior:** `benchmarks/preregistration_addendum_g.md` (Hg1, FAIL),
`benchmarks/preregistration_addendum_o_mapping_recalibration_closure.md` (mapping M1)

## Motivation

Addendum G (Hg1) tested whether LLM-inferred affect — no preset oracle — lets AFT beat naive
cosine on affect-rich but affect-unlabeled retrieval. It **FAILED** (`aft_llm_dual` top1=0.315
vs `naive_cosine` 0.325; Δ=−0.010, d=−0.032, p_one=0.367). The follow-up diagnosis (WP-1a) and
Addendum N established the failure was **mis-calibration of the SEC→valence/arousal mapping,
not absence of signal** (valence Pearson r=0.81 against oracle, but systematic biases +0.19
valence / −0.14 arousal). Addendum O then **numerically recalibrated** that mapping (model M1,
merged in `main` 9ec4752): on a held-out scenario split the biases dropped to +0.072 valence /
−0.023 arousal with MAE improving on both axes.

**Open question (Hp1):** does the recalibrated mapping (M1) change the Hg1 verdict — i.e. does
AFT with LLM dual-path appraisal now beat naive cosine on affect-free retrieval?

This is a **retrieval** claim, distinct from Addendum O's **calibration** claim. It is the
`next_study` logged on claim `appraisal_llm_real_dual_path`.

## Leakage constraint (decisive design choice)

Addendum O calibrated M1 on scenarios drawn from `realistic_recall_v3`. The original Hg1
dataset `realistic_recall_v3_noAF` (scenarios `s01`–`s50`) is **entirely contained in v3**.
Re-running Hg1 on `v3_noAF` after fitting M1 on v3 would be **train/test leakage**. Therefore
Hp1 requires a **fresh affect-free dataset whose scenarios are disjoint from v3**.

## Dataset requirement (pre-specified)

`realistic_recall_v4_noAF` — affect-rich scenarios, NO preset valence/arousal:

- **Scenario IDs disjoint from v3**: new namespace `p01_…`–`pNN_…` (none of `s01`–`s125`).
- **≥ 40 scenarios**, **≥ 150 queries** total (matching the Addendum G floor for comparability).
- Challenge types spanning all five: `recency_confound`, `semantic_confound`, `affective_arc`,
  `same_topic_distractor`, `momentum_alignment` (≥1 `semantic_confound`, per `_validate_dataset`).
- **No `valence`/`arousal` fields** on any event (LLM must infer); **no `state` field** on queries.
- **Author-blind to system output**: dataset finalized and committed **before any benchmark run**.
- Construction method: a scripted generator `benchmarks/datasets/generate_v4_noAF.py` drafts
  scenarios with gpt-5-mini (one structured-JSON scenario per call), each validated against the
  `_NoAFDataset` schema + `_validate_dataset` before inclusion; topics seeded to be thematically
  disjoint from v3. The generation step is **independent of any system scoring** (author-blind):
  the generator never runs the retrieval systems. The frozen dataset is committed before Phase 3.
- Reproducibility: generator is seeded; the committed dataset JSON is the canonical frozen artifact
  (re-running the generator is not required to reproduce the benchmark).

## Systems under test

Identical to Addendum G (same `runner_hg1.py`):

| System | Appraisal | dual_path | Preset affect |
|--------|-----------|-----------|---------------|
| `aft_llm_dual` | LLMAppraisalEngine | yes | none |
| `naive_cosine` | none | n/a | none |
| `aft_neutral` (Hp2) | none (neutral 0,0.5,0.5) | n/a | none |
| `aft_llm_sync` (Hp3) | LLMAppraisalEngine | no | none |

## Primary hypothesis (Hp1, confirmatory)

**H0:** aft_llm_dual.top1_accuracy ≤ naive_cosine.top1_accuracy
**H1:** aft_llm_dual.top1_accuracy > naive_cosine.top1_accuracy

- **Metric:** top1_accuracy (primary, pre-specified)
- **Direction:** one-tailed (AFT expected to win)
- **Test:** paired bootstrap over per-query correctness, n=10,000 resamples
- **Seed:** 0 (frozen)
- **Significance:** α = 0.05
- **Effect threshold:** Δ > 0.05 (5 percentage points) for practical significance
- **Correction:** none (single confirmatory hypothesis)

These are **identical** to Hg1 by design: the only changes vs Addendum G are (a) the
recalibrated M1 mapping (already in `main`) and (b) the leakage-free dataset. Embedder
(`sbert-bge`), seed, bootstrap, and statistic are held fixed.

## Exploratory hypotheses

**Hp2 (neutral ablation):** aft_llm_dual.top1 > aft_neutral.top1 — does the LLM affect signal
beat a fixed neutral affect?

**Hp3 (sync vs dual):** aft_llm_dual.top1 > aft_llm_sync.top1 — does deferred (dual-path)
elaboration beat synchronous appraisal?

Both exploratory: reported with CIs, no multiple-comparison correction, no pass/fail gate.

## Decision rule

- **Hp1 PASS** (H1 accepted): Δ > 0.05 AND p < 0.05 → recalibrated LLM appraisal enables a
  deployable architecture advantage; the `appraisal_llm_real_dual_path` claim broadens.
- **Hp1 FAIL** (H0 retained): otherwise → the architecture-vs-cosine boundary holds even with a
  well-calibrated mapping; recalibration improves affect fidelity (Addendum O) but does not, on
  its own, convert into retrieval advantage. Honest negative; claim scope unchanged.

## Relationship to prior results (context, not hypothesis)

The Hg1 result (Δ=−0.010, old mis-calibrated mapping) is the natural comparison point. The two
runs differ in mapping AND dataset, so the Hp1 vs Hg1 delta is **descriptive context**, not a
controlled A/B — the confirmatory test is solely Hp1 (dual vs cosine on v4_noAF). The Hg3 PASS
(dual-path > sync) from Addendum G is not disputed.

## Statistical analysis plan

Identical to Addendum D/G:

- Per-query correctness vectors for each system
- Paired bootstrap (same query resampled across systems)
- Report: top1_accuracy, Δ vs naive_cosine, 95% CI (percentile), p-value, Cohen's d

## Prerequisites before execution

- [ ] Construct `benchmarks/datasets/realistic_recall_v4_noAF.json` (≥40 scenarios, ≥150
      queries, scenario IDs disjoint from v3)
- [ ] Dry-validate with `hash` embedder (no LLM) — schema + non-LLM baselines load and are
      non-trivial
- [ ] Commit dataset **before** any LLM benchmark run (author-blind)
- [ ] Configure LLM environment (`EMOTIONAL_MEMORY_LLM_*`, gpt-5-mini, as Addendum O)
- [ ] Run `runner_hg1.py --dataset realistic_recall_v4_noAF.json` → `results.hg1_v4.*`
- [ ] Write closure `benchmarks/preregistration_addendum_p_hg1_rerun_closure.md`
- [ ] Update `docs/research/claim_validation_matrix.json` + `CHANGELOG.md`
