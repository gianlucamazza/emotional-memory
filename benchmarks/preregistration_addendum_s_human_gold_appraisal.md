# Pre-registration Addendum S — Hs1: Appraisal construct validity vs human-gold affect (EmoBank)

**Status:** EXECUTED (2026-06-26) — LLM valence **human-validated** (r=0.70);
arousal/dominance weak-positive; keyword engine not validated. See
`preregistration_addendum_s_human_gold_appraisal_closure.md`.
**Date (pre-reg):** 2026-06-26
**Dataset:** `benchmarks/datasets/emobank_v1.json` (300 rows sampled seed=42 from
EmoBank, ≥4 words; human VAD on a 1–5 scale mapped to CoreAffect ranges)
**LLM:** resolved from `EMOTIONAL_MEMORY_LLM*\*` (`.env`pins`gpt-5-mini`)
**Issue:** #62 (A5). Closes the "future work" item in
`docs/research/problem_register_2026-06.md` §A5 / §7.

## Motivation

The affect signal has been _diagnosed_ (Addendum N: valence Pearson r≈0.88 vs gold,
+0.169 bias) and _partly recalibrated_ (Addendum O), but the "gold" used so far is
itself **LLM-derived**, not human. A5 asks the construct-validity question
directly: how well does the appraisal pipeline reproduce **human-annotated** affect?

EmoBank is the appropriate corpus: it provides continuous human Valence/Arousal/
Dominance ratings — a direct match to `CoreAffect(valence, arousal, dominance)` —
so no categorical-to-VAD remapping is needed.

## Scale mapping (frozen in the dataset)

EmoBank VAD on 1–5 (midpoint 3) → CoreAffect:
`valence = (V−3)/2 ∈ [−1,1]`, `arousal = (A−1)/4 ∈ [0,1]`, `dominance = (D−1)/4 ∈ [0,1]`.

## Systems

- `llm` — `LLMAppraisalEngine` (Scherer CPM prompt) → `AppraisalVector.to_core_affect()`.
- `keyword` — `KeywordAppraisalEngine` (rule-based, no LLM) → `to_core_affect()`. Baseline.

One appraisal call per text (repeats=1), cache disabled. Predicted vs human are
paired per row.

## Hypotheses / quantities

For each engine and each dimension (valence / arousal / dominance):

- **Hs1 (primary).** Pearson r between predicted and human affect, with 95%
  bootstrap CI. Construct validity is supported on a dimension iff the r CI lower
  bound > 0 (significantly positive).
- **Bias.** Mean residual `predicted − human`, with 95% bootstrap CI.
- **MAE.** Mean absolute residual.

## Statistical analysis plan (pre-declared)

- Pairing: per row (identical texts across engines).
- Pearson r and bias CIs by percentile bootstrap, `n_bootstrap=2000`, `seed=42`
  (`benchmarks/common/statistics.bootstrap_ci`).
- N = 300 (full committed subset; no further subsetting in the confirmatory run).

## Decision rule

- A dimension is reported as **human-validated (directional)** iff its Pearson r
  CI lower bound > 0 for the `llm` engine. Otherwise **not human-validated**.
- No "accurate appraisal" wording survives unless the corresponding dimension's r
  is both positive and CI-separated from 0 **and** |bias| is reported alongside.
- Result stands as measured; no post-hoc reframing.

## Execution

```bash
make bench-human-gold                 # full run (requires API key)
uv run python -m benchmarks.human_gold_appraisal.runner --dry-run   # keyword-only, no LLM
```

**Pre-registration integrity:** committed **before** the LLM run; the closure
(`..._closure.md`) reports realized r / bias / MAE per dimension and the resulting
appraisal-claim scope.
