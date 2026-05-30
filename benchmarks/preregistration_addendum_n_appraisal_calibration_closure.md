# Pre-Registration Addendum N — Closure (Appraisal Prompt Calibration)

**Status:** FAIL (Hn1 FAIL, Hn2 FAIL) — with a strong valence-axis sub-result
**Date executed:** 2026-05-30
**Protocol version:** addendum_n_v1 (closure)
**Parent pre-reg:** `benchmarks/preregistration_addendum_n_appraisal_calibration.md`

> **Epistemic status:** Numbers below are verbatim from
> `results.diagnostic.gpt5mini.n150.json` (baseline, current prompt) and
> `results.diagnostic.gpt5mini.calibrated.json` (calibrated prompt), both N=150, seed 42,
> gpt-5-mini. Gold-set counts from `make bench-appraisal` before/after the prompt edit (the
> baseline measured by temporarily reverting the prompt via `git stash`). Interpretive criteria
> were frozen before execution.

---

## Results

### Hn1 (primary) — bias reduction on the diagnostic subsample (N=150, seed 42)

| Axis | bias baseline | bias calibrated | Δ\|bias\| | std b→c | Pearson r b→c |
|---|---|---|---|---|---|
| Valence | +0.169 | **+0.044** | **−0.125** | 0.286 → 0.308 | 0.883 → 0.873 |
| Arousal | −0.115 | −0.118 | **+0.003** | 0.162 → 0.152 | 0.483 → 0.555 |

Valence sign accuracy: 0.927 → 0.920. `self_relevance` mean: 0.840 → **0.918** (more saturated,
the opposite of the intended de-saturation).

**Hn1 FAIL.** The pre-registered criterion requires *both* absolute biases to fall. Valence bias
drops sharply (+0.169 → +0.044, now below the 0.10 threshold), but **arousal bias does not move**
(−0.115 → −0.118). The AND condition is not met. The r guardrail holds (valence r −0.010, within
the −0.05 allowance; arousal r actually improves). Note the runner still prints `P1d` for the
calibrated prompt because valence residual std (0.308) is above the 0.30 variance threshold.

### Hn2 (secondary) — directional gold set (`benchmarks/appraisal_quality`, 15 cases)

| | Baseline | Calibrated |
|---|---|---|
| Passed | 15 / 15 | **14 / 15** |

**Hn2 FAIL.** Regression on `routine_lunch` ("I ate the same sandwich I always have for lunch") —
`self_relevance` median 1.000 vs the asserted `< 0.5`. This is consistent with the diagnostic:
the calibrated prompt raised mean `self_relevance` (0.84 → 0.92) rather than lowering it. The
"keep self_relevance high for significant events" instruction dominated the de-saturation anchor
on first-person sentences.

---

## Interpretation

The intervention **failed its pre-registered criteria** (both hypotheses FAIL) and must not be
reported as a successful calibration. Two honest takeaways stand:

1. **Strong positive sub-result on valence.** Prompt anchoring nearly eliminated the systematic
   positive valence bias (+0.169 → +0.044). Since valence carries the dominant weight in
   `to_core_affect` and had the highest baseline correlation (r≈0.81–0.88), this is the most
   consequential axis — direct evidence that the Hg1 null is, at least on valence, a calibration
   problem rather than a capability one.
2. **The prompt-only lever is insufficient and has side effects.** Arousal bias is untouched
   (its driver is the SEC→arousal *mapping*, not the prompt wording), valence residual variance
   rose, and `self_relevance` over-saturated, regressing one gold-set case. Anchors traded off
   against each other.

The pre-registered fallback applies: a complete fix needs a **numeric recalibration of the
SEC→valence/arousal mapping** (`_scherer_project`), fitted with a train/test split, rather than
further prompt tweaking. Arousal specifically depends on `0.5*|novelty| + 0.3*(1−coping) +
0.2*self_relevance`, which the prompt cannot directly rebalance.

**Leakage discipline honoured:** Hg1 was **not** re-run (its dataset shares scenarios with the
calibration set). No retrieval claim is made.

---

## Disposition of the prompt change

Because both hypotheses FAIL, the calibrated prompt is **not** merged as an improvement on the
strength of this study alone. See the PR for the keep-vs-revert decision; either way the
diagnosis, the before/after numbers, and this closure are retained as the evidence trail, and the
next study is the mapping recalibration.

## Reproduce

```bash
# baseline (revert prompt first), then calibrated:
uv run python -m benchmarks.appraisal_diagnostics.runner --n 150 --seed 42 \
  --out-json benchmarks/appraisal_diagnostics/results.diagnostic.gpt5mini.n150.json \
  --out-md   benchmarks/appraisal_diagnostics/results.diagnostic.gpt5mini.n150.md
make bench-appraisal   # 15-phrase gold set
```
