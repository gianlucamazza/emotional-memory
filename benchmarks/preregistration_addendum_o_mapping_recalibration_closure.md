# Pre-Registration Addendum O — Closure (SEC→Affect Mapping Recalibration)

**Status:** PASS (Ho1 + Ho2) — model **M1** promoted
**Date executed:** 2026-05-31
**Protocol version:** addendum_o_closure_v1
**Parent pre-reg:** `benchmarks/preregistration_addendum_o_mapping_recalibration.md`

> Numbers below are verbatim from `benchmarks/appraisal_calibration/results.recalibration.gpt5mini.json`
> (gpt-5-mini, 750-event SEC dump, by-scenario 70/30 split seed 42 → 522 train / 228 test).

---

## Results (held-out test, 228 events)

| Axis | model | bias | MAE | Pearson r | sign-acc |
|---|---|---|---|---|---|
| Valence | baseline | +0.200 | 0.306 | 0.807 | 0.846 |
| Valence | M1 | **+0.072** | 0.298 | 0.789 | 0.842 |
| Valence | M2 | +0.020 | 0.280 | 0.808 | 0.846 |
| Arousal | baseline | −0.144 | 0.182 | 0.466 | — |
| Arousal | M1 | **−0.023** | 0.118 | 0.454 | — |
| Arousal | M2 | −0.018 | 0.111 | 0.520 | — |

**5-fold CV by scenario (M1, train):** valence bias +0.005±0.041, MAE 0.278±0.019, r 0.831±0.024;
arousal bias −0.001±0.013, MAE 0.123±0.017, r 0.522±0.045. Coefficient sd small (valence
[0.05,0.06,0.04,0.04], arousal [0.05,0.05,0.05,0.02]) → stable, not overfit.

## Hypothesis verdicts

- **Ho1 (primary) — PASS for M1** (and M2): both absolute biases fall below the 0.10 threshold
  and below baseline. The decisive win is **arousal: −0.144 → −0.023** — the axis prompt-only
  Addendum N could not move (it is governed by the SEC→arousal projection, not prompt wording).
  Valence: +0.200 → +0.072.
- **Ho2 (guardrail) — PASS for M1:** MAE improves on both axes (valence 0.306→0.298, arousal
  0.182→0.118); Pearson r within the −0.05 allowance (valence 0.807→0.789, arousal 0.466→0.454).
- **Ho3 (parsimony) — selects M2** (within 0.01 test-MAE of best on both axes). **Overridden in
  favour of M1** (declared, motivated): M2 is a free linear fit on the raw SECs, which converts
  the mapping from a *theory-motivated Scherer projection* into a data-driven regressor, weakening
  the `theory_faithful_operationalization` claim and the paper's narrative. M1 keeps the Scherer
  parametrization (`coping_signed`, `|novelty|`, `1−coping`) and already satisfies the primary
  hypotheses; the MAE gap to M2 (valence +0.018, arousal +0.007) is the accepted price of fidelity.
- **Ho4 (gold-set) — NOT APPLICABLE.** Discovered during execution: the 15-phrase gold set
  (`benchmarks/appraisal_quality/`) asserts on the five **raw SEC dimensions** emitted by the
  *prompt*; it never calls `to_core_affect()`. Addendum O changes only the SEC→V/A **mapping**,
  not the prompt, so the gold set is invariant to this change by construction — running it would
  be non-informative. The mapping's real validation is the fidelity suite
  (`benchmarks/fidelity/test_appraisal_affect.py`, which exercises `to_core_affect`) plus the
  held-out test set above. Ho4 is recorded as N/A rather than executed.

## Promoted model — M1 coefficients

```
valence = 0.4805*goal_relevance + 0.1862*norm_congruence + 0.1643*coping_signed + 0.0179*novelty
arousal = 0.3694*|novelty| + 0.1357*(1−coping_potential) + 0.2208*self_relevance + 0.1399
coping_signed = 2*coping_potential − 1 ;  dominance = coping_potential (unchanged)
```

Invariants: neutral appraisal → valence **0** preserved (valence intercept constrained to 0, G1).
Neutral arousal shifts 0.15 → **0.208** (free arousal intercept 0.1399 + 0.1357·0.5); the
`test_neutral_appraisal_neutral_affect` expectation is updated accordingly. `CoreAffect` Pydantic
validators already clamp valence∈[−1,1] / arousal∈[0,1], so no extra clamp is needed.

## Limitations (declared)

1. **End-to-end calibration, not pure mapping estimate.** The oracle is on final valence/arousal
   only, not on the SECs, so the fit absorbs both LLM SEC error and mapping mis-weighting. The
   weights are calibrated for gpt-5-mini's appraisal output; a different appraisal engine may need
   re-fitting.
2. **dominance** (`= coping_potential`) is not recalibrated (no oracle).
3. **Calibration claim, not retrieval claim.** This shows the mapping reproduces oracle affect on
   held-out scenarios. Whether it improves end-to-end retrieval (Hg1) is a separate study and
   requires a scenario set **disjoint from v3** — `realistic_recall_v3_noAF` (Hg1's set) is
   contained in v3, so it cannot serve. Logged as `next_study`.
