# Closure Addendum W — Affine calibration of direct-VAD arousal (EXECUTED)

**Date (executed):** 2026-06-27
**Verdict:** **adopt calibration = YES** — Hw1 PASS, Hw2 PASS, Gw OK.
**Pre-registration:** `preregistration_addendum_w_arousal_calibration.md` (registered with the
harness before the result was observed).
**Inputs:** `benchmarks/arousal_calibration/predictions.json` — one Addendum V LLM pass
(`gpt-5-mini`, seed=42, N=300, **0 fallback** in either arm). Calibration is deterministic and
offline. Full numbers: `benchmarks/arousal_calibration/results.{json,md}`.

## Result

Direct-VAD arousal (Addendum V) had the better correlation (r≈0.58) but a worse MAE than the
SEC→projection (0.19 vs 0.11), driven by a systematic under-prediction (bias −0.10) and a wider
spread than EmoBank's narrow arousal range. An OLS affine map fixes scale/offset while leaving
the Pearson r intact.

| arousal (held-out)                 |       MAE |   bias | Pearson r |
| ---------------------------------- | --------: | -----: | --------: |
| direct_vad (raw)                   |     0.200 | −0.094 |     0.523 |
| **direct_vad (calibrated)**        | **0.040** | +0.002 |     0.523 |
| scherer_m1 (production projection) |     0.105 | −0.049 |     0.349 |

(native split, fit on train N=250, evaluated on dev+test N=50; the 5-fold CV protocol over all
N=300 agrees: calibrated MAE 0.041 vs raw 0.197 vs scherer 0.116, r 0.565.)

- **Hw1 PASS** — calibrated < raw MAE: ΔMAE +0.160 [+0.125, +0.194], p<0.001 (≈80% MAE cut).
- **Hw2 PASS** — calibrated < scherer MAE: ΔMAE +0.066 [+0.047, +0.086], p<0.001. Calibrated
  direct-VAD now **dominates the production projection on _both_ axes** — r (0.57 vs 0.21, from
  Addendum V) _and_ MAE (0.04 vs 0.12).
- **Gw OK** — slope a=+0.16 > 0 and r preserved (affine-invariant). The guard mattered: it rules
  out the degenerate "predict the mean" optimum that would minimize MAE by destroying r.

**Deployable coefficients (full-sample fit):** `arousal_cal = 0.1627·arousal_direct + 0.4491`.

## Honest caveats

- **Narrow-target effect.** EmoBank arousal is low-variance (range 0.33–0.80, mean ≈0.52), so the
  small slope (0.16) + large intercept (0.45) largely shrink the prediction toward the corpus
  mean. Part of the dramatic MAE cut therefore reflects EmoBank's narrow arousal distribution, not
  only a better predictor. On a wider-variance corpus the MAE gain would be smaller. What is
  **not** corpus-specific is the r-preservation: the calibrated predictor keeps the rank/linear
  signal (r=0.57) that the projection lacks (0.21), so the _dominance on both axes_ is the durable
  claim, not the absolute 0.04 MAE.
- **EmoBank-fit coefficients.** The affine `(a, b)` are fit on EmoBank; deploying them on another
  arousal scale requires re-fitting (the harness `--dump-predictions` → `bench-arousal-calibration`
  path makes that a single LLM pass plus a deterministic step).
- **Scope.** This validates the _measurement_ (arousal calibration vs human gold). It does not by
  itself change any retrieval result; whether calibrated arousal improves downstream retrieval is a
  separate question (the affect signal's retrieval value is bounded by Addenda U/T2A regardless).

## Follow-up (not in this study)

If adopted in the library, ship the calibration as an opt-in post-processing step on the
`DIRECT_VAD_SCHEMA` arousal output (a pre-registered library change, with the coefficients
re-fittable per corpus), keeping the raw schema as default for theory fidelity.
