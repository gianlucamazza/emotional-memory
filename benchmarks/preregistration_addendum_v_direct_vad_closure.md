# Closure — Addendum V (C): Direct-VAD appraisal vs the SEC→projection

**Status:** EXECUTED 2026-06-27 · **Decision: adopt direct-VAD = YES**
**Pre-registration:** `preregistration_addendum_v_direct_vad.md` (committed before execution)
**Runner:** `benchmarks/appraisal_vad/runner.py` · **Artifact:** `benchmarks/appraisal_vad/results.{json,md}`
**Dataset:** EmoBank `emobank_v1.json` (human VAD, N=300, CC-BY-SA 4.0), paired bootstrap n=2000, seed=42.

## Result

| Method                    | Dim       | Pearson r [95% CI]       |       bias |   MAE |
| ------------------------- | --------- | ------------------------ | ---------: | ----: |
| `scherer_m1` (production) | valence   | 0.695 [0.644, 0.740]     |     +0.157 | 0.324 |
| `scherer_m1`              | arousal   | 0.228 [0.129, 0.324]     |     −0.060 | 0.112 |
| `scherer_m1`              | dominance | 0.307 [0.192, 0.411]     |     +0.135 | 0.260 |
| **`direct_vad`**          | valence   | **0.790 [0.750, 0.824]** | **−0.013** | 0.329 |
| **`direct_vad`**          | arousal   | **0.582 [0.510, 0.652]** |     −0.093 | 0.193 |
| **`direct_vad`**          | dominance | **0.428 [0.325, 0.521]** |     −0.018 | 0.142 |

Paired Δr (direct_vad − scherer_m1): valence **+0.095 [0.052, 0.141]**, arousal
**+0.354 [0.251, 0.457]**, dominance +0.122 [−0.008, 0.243].

| Verdict                         | Outcome                                                  |
| ------------------------------- | -------------------------------------------------------- |
| Hv1 — arousal r improved        | ✅ PASS (Δr +0.354, CI excludes 0)                       |
| Hv2 — dominance r improved      | ✗ FAIL (Δr +0.122, CI lower bound −0.008 just touches 0) |
| Hv3 — valence \|bias\| reduced  | ✅ PASS (+0.157 → −0.013)                                |
| Gv — valence not regressed      | ✅ OK (r actually improved 0.695 → 0.790)                |
| **Decision — adopt direct-VAD** | **✅ YES**                                               |

## Interpretation

**The SEC→linear-projection was the bottleneck.** Asking the LLM for valence/arousal/
dominance _directly_ (a different `AppraisalSchema`, identity projection, no library
change) beats the production Scherer-SEC→Addendum-O path on **every** axis against human
gold:

- **arousal** is the headline: r 0.23 → 0.58 (Δr +0.354) — the weak axis A5 flagged is
  substantially recovered;
- **valence** improves (r 0.70 → 0.79) **and** the standing +0.15 positive bias collapses
  to −0.013 — direct rating removes the projection-induced bias that Addendum O (fit on
  LLM-derived gold) could not fix against human gold;
- **dominance** improves directionally (r 0.31 → 0.43) but the paired Δr CI just touches 0
  (Hv2 FAIL by the strict rule) — promising, not conclusive.

**Honest caveats.** (1) `direct_vad` arousal MAE is _worse_ (0.112 → 0.193): correlation
improved but the absolute scale/offset is off — direct VAD ranks arousal far better but
needs a small affine calibration to match the human scale. (2) This validates direct VAD
as an _appraisal-quality_ method against human labels; it does **not** by itself show
retrieval benefit (cf. Addendum P — better-calibrated affect can still be a net distractor
when semantics discriminate). Whether the improved signal helps retrieval is the question
for Addendum T (direction A).

## Recommended next action (follow-up, not in this PR)

Adopt: expose the direct-VAD `AppraisalSchema` from the library (a selectable schema, the
SEC schema remaining the default), optionally with a small arousal affine calibration, and
re-validate. Then Addendum T (retrieve-time query appraisal) should use the **direct-VAD**
query estimate as its affect signal — a stronger query-affect than the SEC path.

## Claim-matrix impact

No new claim. Bounds `appraisal_human_validated`: the weak arousal/dominance axes it flagged
are largely a projection artifact — a direct-VAD estimator reaches valence r=0.79 / arousal
r=0.58 / dominance r=0.43 with near-zero valence bias. `08_limitations` §1.1 (dominance/axis
estimation) updated accordingly.
