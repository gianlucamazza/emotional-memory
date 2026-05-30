# Pre-Registration Addendum N — Appraisal Prompt Calibration

**Status:** PENDING EXECUTION
**Date written:** 2026-05-30
**Protocol version:** addendum_n_v1
**Study type:** Exploratory (prompt engineering; not a confirmatory retrieval hypothesis)
**Parent work:** `benchmarks/appraisal_diagnostics/protocol.md` (WP-1a),
`benchmarks/preregistration_addendum_g_closure.md` (Hg1 FAIL)

> **Epistemic status:** This file establishes the protocol for Addendum N before any
> calibrated-prompt result is collected. The hypotheses, metrics, datasets and thresholds
> below are frozen at this commit. The baseline numbers it references come from a prior,
> independently-committed run (`results.diagnostic.gpt5mini.json`, N=750, seed 42).

---

## Motivation

The WP-1a diagnostic (gpt-5-mini, N=750 events of `realistic_recall_v3.json`) returned the
decision-tree verdict **P1d** (bias AND variance both over threshold), but the underlying
numbers show the appraisal signal is **mis-calibrated, not absent**:

| Axis | bias (LLM − oracle) | 95% CI | std | Pearson r | sign-acc |
|---|---|---|---|---|---|
| Valence | **+0.186** | [0.159, 0.212] | 0.362 | **0.811** | **0.864** |
| Arousal | **−0.138** | [−0.151, −0.125] | 0.180 | 0.369 | — |

SEC descriptives: `self_relevance` mean 0.82 (near-saturated), `coping_potential` mean 0.67.

A valence Pearson r of 0.81 and sign accuracy of 0.86 mean the model *understands* event
valence; it simply reports it too positively (+0.19) and under-activates arousal (−0.14). This
is the P1b regime (fix the prompt) with a P1c tail (reduce variance). Addendum G's `falsified`
status is not overturned — it is **qualified**: the affect signal is recoverable via
calibration. This study tests whether a calibrated zero-shot prompt reduces those biases.

---

## Leakage constraint (pre-specified)

`realistic_recall_v3.json` (oracle affect, this study's measurement set) and
`realistic_recall_v3_noAF.json` (the dataset Hg1 retrieves on) are the **same scenarios**.
Tuning the prompt against v3 and then re-running Hg1 on v3_noAF would be train/test leakage.
Therefore:

1. The **primary** validation is the bias measured by the diagnostic runner on v3 — this is a
   calibration-quality measurement, **not** a retrieval benchmark, and is reported as such.
2. The **independent** validation is the 15-phrase gold set in
   `benchmarks/appraisal_quality/dataset.py`, which shares no items with v3.
3. **Hg1 is NOT re-run in this addendum.** A future Hg1 re-run must use a scenario set disjoint
   from the calibration set; that is logged as `next_study`, not executed here.

---

## Intervention (frozen scope)

Calibrate the Scherer CPM **prompt only** — `_SCHERER_SYSTEM_PROMPT` in
`src/emotional_memory/appraisal_schema.py` and its twin `_SYSTEM_PROMPT` in
`src/emotional_memory/appraisal_llm.py` (kept identical). The numeric projection
`AppraisalVector.to_core_affect()` / `_scherer_project()` is **not** modified, so any change in
bias is attributable to the prompt alone and is trivially reversible.

Calibration levers, each targeting a measured bias:
- Anti-positivity anchors (valence bias +0.19): reserve high positive `goal_relevance` /
  `norm_congruence` for clearly favourable outcomes; default to 0 for neutral/ambiguous events.
- Arousal anchors (arousal bias −0.14): tie high novelty / low coping to surprising or
  threatening events; clarify that calm-but-significant events are not low-arousal.
- `self_relevance` scale anchors (mean 0.82): 0 = irrelevant, 0.5 = concerns others but
  relevant, 1 = directly about the subject.
- 2–3 balanced few-shot examples (one negative high-arousal, one positive, one neutral) to
  reduce variance.

---

## Hypotheses

**Hn1 (primary, directional):** the calibrated prompt reduces the absolute valence bias and the
absolute arousal bias versus baseline, measured on a fixed diagnostic subsample
(`--n 150 --seed 42`).
- Pass: `|bias_valence_calibrated| < |bias_valence_baseline|` **AND**
  `|bias_arousal_calibrated| < |bias_arousal_baseline|`, where the baselines are the N=150
  re-measurement of the current prompt (so the comparison is at equal N).
- Guardrail: valence Pearson r must not drop below baseline − 0.05 (no trading bias for noise).

**Hn2 (secondary):** the calibrated prompt does not regress the directional gold set —
`# failing assertions_calibrated ≤ # failing assertions_baseline` on
`benchmarks/appraisal_quality` (median over `EMOTIONAL_MEMORY_LLM_REPEATS`).

---

## Procedure

1. Commit this pre-registration and the unchanged baseline-at-N=150 measurement **before**
   editing the prompt.
2. Baseline @ N=150: `uv run python -m benchmarks.appraisal_diagnostics.runner --n 150 --seed 42`
   → `results.diagnostic.gpt5mini.n150.{json,md}` (current prompt).
3. Apply the prompt calibration.
4. Calibrated @ N=150: same command → `results.diagnostic.gpt5mini.calibrated.{json,md}`.
5. Gold set baseline/calibrated: `make bench-appraisal` before and after the edit.
6. Write the closure doc with the before/after table and the Hn1/Hn2 verdict.

## Frozen parameters

| Parameter | Value |
|---|---|
| Model | gpt-5-mini (matches Hg1 / WP-1a) |
| Diagnostic subsample N | 150 |
| Seed | 42 |
| n_bootstrap | 10 000 |
| bias_threshold / std_threshold | 0.10 / 0.30 (unchanged) |
| Gold-set repeats | `EMOTIONAL_MEMORY_LLM_REPEATS` (default 3) |

## Reporting rules

1. Hn1 is the primary outcome; reported regardless of direction.
2. Hn2 is secondary.
3. Both are exploratory — they cannot be promoted to a confirmatory retrieval claim, which
   requires a leakage-free Hg1 re-run on a disjoint scenario set.
4. seed=42 frozen; no re-seeding after results are known.
5. Honest-negative outcome (bias not reduced, or gold set regresses) is a valid result and is
   documented; the fallback is a numeric recalibration of `_scherer_project` weights fitted with
   a train/test split, deferred to a separate addendum.
