# Pre-registration Addendum V — Hv1/Hv2/Hv3: Direct-VAD appraisal vs the SEC→projection

**Status:** EXECUTED (2026-06-27) — **adopt direct-VAD = YES** (Hv1 PASS, Hv3 PASS, Gv OK;
Hv2 dominance positive but CI touches 0). See `preregistration_addendum_v_direct_vad_closure.md`.
**Date (pre-reg):** 2026-06-27
**Dataset:** `benchmarks/datasets/emobank_v1.json` (EmoBank human VAD, N=300, CC-BY-SA 4.0)
**LLM:** resolved from `EMOTIONAL_MEMORY_LLM*\*` (`.env`pins`gpt-5-mini`)
**No engine change** — both methods are `LLMAppraisalEngine`with different`AppraisalSchema`.

## Motivation

Addendum S validated the **production** appraisal path (LLM → 5 Scherer SECs →
the Addendum-O-recalibrated linear projection `_scherer_project`) against human gold:
**valence r=0.70 (good), but arousal r=0.28 and dominance r=0.33 (weak), and a +0.15
valence bias persists**. Addendum O's recalibration is _already live_ in `_scherer_project`
(it was fit against LLM-derived oracle affect, not human gold — hence the residual human-gold
bias). The open question (direction C): is the **SEC→linear-projection** itself the bottleneck
on arousal/dominance? This addendum tests a genuinely different estimator — **asking the LLM for
valence/arousal/dominance directly** — against the same human gold, paired per item.

## Methods (both vs EmoBank human VAD)

- `scherer_m1` — production: `LLMAppraisalEngine` (default `SCHERER_CPM_SCHEMA`) → 5 SECs →
  `_scherer_project` (Addendum O M1) → `CoreAffect`. This reproduces the Addendum S `llm` result.
- `direct_vad` — `LLMAppraisalEngine` with a custom 3-dimension `AppraisalSchema` (valence
  [-1,1], arousal [0,1], dominance [0,1]); the LLM outputs V/A/D **directly**; identity
  projection to `CoreAffect`. **No library change** — `AppraisalSchema` is already pluggable.

One call per text per method (repeats=1, cache off). Predictions are paired per EmoBank row.

## Hypotheses

- **Hv1.** `direct_vad` arousal Pearson r > `scherer_m1` arousal r.
- **Hv2.** `direct_vad` dominance Pearson r > `scherer_m1` dominance r.
- **Hv3.** `direct_vad` |valence bias| < `scherer_m1` |valence bias| (improves the +0.15).
- **Guard Gv (no-regression on the good axis).** `direct_vad` valence r must not drop materially
  below `scherer_m1`: the paired Δr_valence 95% CI lower bound > −0.05.

## Statistical analysis plan (pre-declared)

- Per dimension, per method: Pearson r, bias (mean residual pred−human), MAE, with 95% bootstrap
  CI (reuse `benchmarks/human_gold_appraisal/runner.py` `_dimension_stats`).
- **Paired** Δr = r(direct_vad) − r(scherer_m1) per dimension, with 95% bootstrap CI over the
  shared item indices, n_bootstrap=2000, seed=42. Hv1/Hv2 PASS iff Δr > 0 and CI excludes 0.
- N = 300 (full committed subset).

## Decision rule

- **Adopt direct-VAD** as an alternative production appraisal schema iff **(Hv1 or Hv2) PASS**
  **and** Gv holds (valence not regressed). Then a follow-up wires the VAD schema into the library.
- Otherwise the SEC→projection stands; arousal/dominance remain a known weak axis (honest).
- Result reported as measured; no post-hoc reframing.

## Execution

```bash
make bench-appraisal-vad                                           # full run (requires API key)
uv run python -m benchmarks.appraisal_vad.runner --limit 4         # quick check
```

**Pre-registration integrity:** committed before execution; the closure
(`..._closure.md`) reports realized r/bias/MAE per method, the paired Δr per dimension, and the
Hv1/Hv2/Hv3 + Gv verdicts.
