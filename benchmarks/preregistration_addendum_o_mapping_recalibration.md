# Pre-Registration Addendum O — SEC→Affect Mapping Recalibration

**Status:** PENDING EXECUTION
**Date written:** 2026-05-31
**Protocol version:** addendum_o_v1
**Study type:** Confirmatory (pre-registered hypotheses on a held-out test set)
**Parent work:** `benchmarks/preregistration_addendum_n_appraisal_calibration_closure.md` (Hn1/Hn2
FAIL — prompt-only calibration cut valence bias but not arousal); WP-1a diagnostic
(`benchmarks/appraisal_diagnostics/results.diagnostic.gpt5mini.json`).

> **Epistemic status:** hypotheses, dataset, split, model set, thresholds and the
> intercept-constraint decision below are frozen at this commit, BEFORE the fit is computed.
> The baseline numbers referenced come from prior independently-committed runs.

---

## Motivation

Addendum N showed prompt anchoring nearly removes the systematic valence bias (+0.169→+0.044 at
N=150) but **cannot move the arousal bias** (−0.115→−0.118): arousal is governed by the numeric
SEC→arousal projection (`0.5*|novelty| + 0.3*(1−coping) + 0.2*self_relevance`), not by prompt
wording. The pre-registered fallback in the Addendum N closure is a **numeric recalibration of
the SEC→valence/arousal mapping fitted with a train/test split**. This is that study.

Current mapping (`_scherer_project` in `appraisal_schema.py`, duplicated in
`AppraisalVector.to_core_affect()` in `appraisal.py`):
```
coping_signed = 2*coping_potential − 1
valence = 0.4*goal_relevance + 0.3*norm_congruence + 0.2*coping_signed + 0.1*novelty
arousal = 0.5*|novelty| + 0.3*(1−coping_potential) + 0.2*self_relevance
dominance = coping_potential   # not recalibrated (no oracle); out of scope
```

---

## Data & leakage discipline (accertato)

- Fit/measurement set: `benchmarks/datasets/realistic_recall_v3.json` — **125 scenarios, 750
  events**, each with oracle `valence ∈ [−1,1]` and `arousal ∈ [0,1]`.
- **Oracle is on final valence/arousal only, NOT on the 5 SEC dimensions.** Consequence (declared
  limitation): the fit corrects the *aggregate* LLM+mapping error end-to-end; it cannot separate
  "LLM mis-estimates the SECs" from "mapping is mis-weighted". The recalibrated weights are an
  end-to-end calibration for gpt-5-mini, not a pure estimate of the theoretical mapping.
- **Split by scenario** (never by event), 70/30, seed 42, via `numpy.random.default_rng(42)` over
  lexicographically-sorted `scenario_id`. The explicit train/test scenario lists are written to
  `benchmarks/appraisal_calibration/split.scenarios.seed42.json` and committed with this protocol.
  No event leaks across the split because whole scenarios move together.
- **Hg1 re-run is OUT OF SCOPE here.** Verified: `realistic_recall_v3_noAF` (Hg1's retrieval set)
  is, by content, the 727 texts of v2 — all contained in v3 (v3 has only 19 texts not in v3_noAF).
  A leakage-free retrieval re-run therefore needs a scenario set disjoint from v3 (not the 19
  leftovers). That is logged as `next_study`, not executed.

---

## Models (all fitted with numpy `lstsq`; scipy/sklearn unavailable)

Each model predicts valence and arousal separately from the per-event SEC vector.

- **M0 — baseline + intercept.** Keep the current weights and theoretical features; add only the
  mean train residual as an offset (`b_v`, `b_a`). Isolates how much of the bias is a constant shift.
- **M1 — recalibrated weights + intercept (RECOMMENDED), theoretical features.** Same Scherer
  parametrization (`coping_signed`, `|novelty|`, `(1−coping)`) but coefficients re-fit. Preserves
  interpretability and can rebalance arousal — the axis Addendum N couldn't fix.
- **M2 — free linear on the 5 raw SECs (control).** Performance ceiling; quantifies what the
  theoretical structure costs.

**G1 (frozen decision): valence intercept is constrained to 0**; arousal intercept is free. This
preserves the invariant *neutral appraisal → valence 0* (asserted exactly by
`tests/test_appraisal.py` and `benchmarks/fidelity/test_appraisal_affect.py`) so promotion does
not break those tests, while arousal — whose neutral value is 0.15, not 0 — keeps a free offset.

Range note: if promoted, the projection must clamp valence/arousal to their domains (fitted
weights can exceed range on extreme inputs); validated at promotion, not in the offline fit.

---

## Hypotheses (evaluated on the held-out TEST scenarios)

Baseline = current mapping applied to the same test events. Metrics per axis: **bias** (mean
residual), **MAE**, **Pearson r** vs oracle, plus valence **sign accuracy**. CIs via
`benchmarks/common/statistics.bootstrap_ci` (n=10000, seed 42); paired diffs via
`paired_bootstrap_diff`; Holm correction across the family.

- **Ho1 (primary, directional):** the chosen recalibrated model reduces the absolute bias on
  **both** axes vs baseline AND brings both `|bias| ≤ 0.10`. (This is exactly what Addendum N
  failed on arousal.)
- **Ho2 (primary, guardrail):** no quality regression — `MAE ≤ baseline` on both axes, and
  `Pearson r ≥ baseline − 0.05` on both axes.
- **Ho3 (secondary, parsimony):** promote the simplest model whose test-MAE is within 0.01 of the
  best (M0 preferred over M1 over M2 on ties).
- **Ho4 (secondary, independent set):** the candidate mapping, injected via a custom
  `AppraisalSchema`, does not regress the 15-phrase gold set
  (`benchmarks/appraisal_quality/`): `#failures ≤ baseline`.

Robustness (reported, not decisive): **5-fold CV by scenario** on the train split — report
mean±sd of bias/MAE/r and the sd of fitted coefficients (high-variance coefficients are fragile).
The confirmatory verdict rests solely on the single held-out test split, examined once.

## Frozen parameters

| Parameter | Value |
|---|---|
| Model / LLM | gpt-5-mini (matches Hg1 / WP-1a / Addendum N) |
| Dataset | realistic_recall_v3.json (125 scenarios / 750 events) |
| Split | 70/30 by scenario, seed 42 |
| Fit | numpy.linalg.lstsq; valence intercept = 0; arousal intercept free |
| n_bootstrap / CI | 10000 / 95% |
| bias threshold | 0.10 |
| parsimony ΔMAE | 0.01 |
| guardrail | MAE not worse; Pearson r not below baseline − 0.05 |

## Procedure

1. Commit this protocol + the frozen split JSON before the fit.
2. Dump per-event SECs once: `benchmarks/appraisal_calibration/dump_sec.py` calls
   `LLMAppraisalEngine.appraise()` for all 750 events, recording
   `(memory_id, scenario_id, 5 SECs, oracle_valence, oracle_arousal)` →
   `sec_dump.gpt5mini.jsonl`. ~750 gpt-5-mini calls (~$2–4). Single LLM cost of the study.
3. Fit M0/M1/M2 on TRAIN only (`fit.py`, pure numpy); evaluate on TEST; run 5-fold CV on train.
4. Gold-set Ho4 via the candidate schema (`make bench-appraisal`).
5. Closure doc with before/after table + Ho1–Ho4 verdict; promote only if Ho1+Ho2 hold.

## Reporting rules

1. Ho1/Ho2 are confirmatory; reported regardless of outcome.
2. seed 42 frozen; no re-seed after results are known.
3. Honest-negative is a valid result (mapping can't beat baseline at parity) — documented, no
   promotion. Hg1 retrieval re-run is a separate future addendum on a disjoint scenario set.
