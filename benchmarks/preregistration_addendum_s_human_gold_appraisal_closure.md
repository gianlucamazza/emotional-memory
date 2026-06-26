# Closure — Addendum S (A5): Appraisal construct validity vs human-gold affect (EmoBank)

**Status:** EXECUTED 2026-06-26 · LLM valence **human-validated**; arousal/dominance
weak-positive; keyword engine **not** human-validated.
**Pre-registration:** `preregistration_addendum_s_human_gold_appraisal.md` (committed before execution)
**Runner:** `benchmarks/human_gold_appraisal/runner.py` · **Artifact:** `benchmarks/human_gold_appraisal/results.{json,md}`
**Dataset:** `benchmarks/datasets/emobank_v1.json` (EmoBank, N=300, CC-BY-SA 4.0),
bootstrap n=2000, seed=42.

## Result (Pearson r vs human VAD, predicted − human bias)

### `llm` engine (`LLMAppraisalEngine`, Scherer CPM, gpt-5-mini)

| Dimension | Pearson r [95% CI]          | bias [95% CI]           |   MAE | human-validated |
| --------- | --------------------------- | ----------------------- | ----: | :-------------: |
| valence   | **+0.703 [+0.655, +0.746]** | +0.152 [+0.115, +0.190] | 0.319 |       ✅        |
| arousal   | +0.276 [+0.172, +0.373]     | −0.058 [−0.071, −0.044] | 0.108 |    ✅ (weak)    |
| dominance | +0.333 [+0.224, +0.432]     | +0.130 [+0.102, +0.158] | 0.256 |  ✅ (moderate)  |

### `keyword` engine (`KeywordAppraisalEngine`, rule-based)

| Dimension | Pearson r [95% CI]      | bias [95% CI] |   MAE |       human-validated        |
| --------- | ----------------------- | ------------- | ----: | :--------------------------: |
| valence   | +0.070 [−0.010, +0.148] | +0.018        | 0.143 |              ✗               |
| arousal   | +0.106 [+0.012, +0.200] | −0.288        | 0.288 | ✗ (CI barely > 0, huge bias) |
| dominance | −0.011 [−0.144, +0.110] | −0.023        | 0.052 |              ✗               |

## Interpretation

- **Valence is human-validated** for the LLM engine: r=0.70 against human gold —
  the first validation of the affect signal against _human_ (not LLM-derived)
  labels. This is lower than the r≈0.88 measured against LLM-gold (Addendum N),
  as expected: human annotation is noisier and the LLM-gold comparison was partly
  circular.
- **The +0.169 positive valence bias (Addendum N) persists vs human gold**
  (+0.152). Addendum O's mapping recalibration is not in this path; the bias is a
  real, standing limitation.
- **Arousal and dominance are only weakly-to-moderately correlated** (r=0.28 /
  0.33) — positive and CI-separated from 0, but far from valence. Appraisal claims
  must not generalize valence's validity to these dimensions.
- **The rule-based `KeywordAppraisalEngine` is not human-validated** on any
  dimension (valence r=0.07). English-only keyword rules do not reproduce human
  affect; only the LLM engine should be cited for construct validity.

## Claim-matrix impact

Adds `appraisal_human_validated` = `early_controlled_evidence`,
evidence level `3_appraisal_quality`, `requires_oracle_affect=false`. The existing
`appraisal_directionally_useful` claim is unchanged in wording; its evidence is
extended to cite this human-gold comparison. No "accurate appraisal" wording is
introduced beyond the validated valence dimension, with the +0.15 bias stated.
