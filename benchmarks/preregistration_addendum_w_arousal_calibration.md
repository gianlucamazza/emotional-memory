# Pre-registration Addendum W — Hw1/Hw2: Affine calibration of direct-VAD arousal

**Status:** PRE-REGISTERED (2026-06-27) — not yet executed. Awaiting the per-item
prediction dump from the Addendum V harness (`benchmarks/appraisal_vad/runner.py
--dump-predictions`, one LLM pass, needs `EMOTIONAL_MEMORY_LLM_API_KEY`). The calibration
analysis itself is deterministic and offline (no LLM). This file is registered **before**
the result is observed, per the project's pre-registration discipline.
**Date (pre-reg):** 2026-06-27
**Dataset:** `benchmarks/datasets/emobank_v1.json` (EmoBank human VAD, N=300, CC-BY-SA 4.0;
native splits train=250 / dev=27 / test=23)
**No engine / library change** — this is a benchmark-side measurement study. If Hw1 PASSES,
a _separate_ follow-up would wire a calibration option into the library.

## Motivation

Addendum V found that asking the LLM for valence/arousal/dominance **directly** beats the
SEC→projection on the _correlation_ of every axis, but on arousal it **trades MAE for r**:

| arousal (vs EmoBank)                 | Pearson r | bias (pred−human) |       MAE |
| ------------------------------------ | --------: | ----------------: | --------: |
| `scherer_m1` (production projection) |     0.228 |            −0.060 | **0.112** |
| `direct_vad` (raw)                   | **0.582** |            −0.093 |     0.193 |

Direct-VAD has the far better _linear structure_ (r 0.58 vs 0.23) but a worse absolute error,
driven by a systematic under-prediction (bias −0.093) and a wider output spread than EmoBank's
narrow arousal range (human arousal ∈ [0.33, 0.80], centered ≈0.5). A Pearson r is **invariant**
under an affine transform, so a fit `arousal_cal = a·arousal_direct + b` can correct the
scale/offset — keeping the better r while cutting MAE. The open question: does affine
calibration make direct-VAD arousal **dominate** the production projection on MAE too (not just
r)? This is the residual follow-up flagged in Addendum V (arousal MAE caveat).

## Methods

- **Inputs (per EmoBank item):** `x = direct_vad` raw arousal prediction, `y =` EmoBank human
  arousal, and `s = scherer_m1` arousal prediction — all from a single Addendum V prediction
  dump (paired per row; same N=300, same LLM, seed=42).
- **Estimator:** ordinary least squares affine fit `(a, b) = argmin Σ (a·x + b − y)²`, fit on a
  **train** partition only and evaluated on a disjoint **test** partition (leakage-free).
- **Two pre-declared evaluation protocols:**
    - **P1 — native split (headline):** fit `(a, b)` on the EmoBank `train` split (N=250),
      evaluate on the held-out `dev`+`test` split (N=50). This is the deployable-coefficients
      scenario and respects the dataset's own protocol.
    - **P2 — 5-fold CV (robustness):** shuffled 5-fold over all N=300 (seed=42); each item gets an
      out-of-fold calibrated value from a model that never saw it; metrics pooled over folds.
- **Metrics on held-out items:** MAE, bias, Pearson r for raw `x`, calibrated `a·x+b`, and
  `scherer` `s`, each vs `y`.

## Hypotheses

- **Hw1 (primary — calibration reduces arousal MAE).** On held-out items, mean per-item
  `|raw_err| − |cal_err| > 0` with the 95% paired-bootstrap CI excluding 0 (P1 headline;
  P2 as robustness). Tested via `paired_bootstrap_diff(|raw_err|, |cal_err|)`,
  n_bootstrap=2000, seed=42.
- **Hw2 (secondary — calibrated direct-VAD beats the projection on MAE).** On held-out items,
  mean `|scherer_err| − |cal_err| > 0` with 95% CI excluding 0 → calibrated direct-VAD dominates
  `scherer_m1` on MAE _and_ r (r already won in Addendum V).
- **Guard Gw (no monotonicity break / r regression).** The full-sample slope `a > 0` and the
  held-out calibrated Pearson r is within 0.02 of the raw r (affine preserves r within a fold;
  the guard catches a degenerate fit or sign flip).

## Decision rule

- **Adopt the affine arousal calibration** (and open a follow-up to ship it as a library option)
  iff **Hw1 PASSES (P1)** and **Gw holds**. Hw2 strengthens the case to "dominates on every
  metric" but is not required for adoption.
- If Hw1 FAILS, the raw direct-VAD arousal stands as-is and the MAE caveat remains documented
  honestly (better r, worse MAE — a real trade-off, not fixed by a linear map).
- Result reported as measured; no post-hoc reframing. The slope/intercept `(a, b)` and all
  held-out metrics are written to `benchmarks/arousal_calibration/results.{json,md}`.

## Execution

1. One-time LLM pass to produce the paired prediction dump (needs the API key):

    ```bash
    uv run python -m benchmarks.appraisal_vad.runner \
        --dump-predictions benchmarks/arousal_calibration/predictions.json
    ```

2. Deterministic offline calibration + verdicts (no LLM, reproducible):

    ```bash
    make bench-arousal-calibration   # = runner over the dump above
    ```

The affine-fit and cross-validation math is unit-tested on synthetic data in
`tests/test_arousal_calibration.py` (no LLM, runs in CI).
