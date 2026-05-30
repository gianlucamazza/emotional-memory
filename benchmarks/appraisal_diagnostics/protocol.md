# Appraisal Diagnostics — WP-1a Protocol

**Status**: Frozen 2026-05-13
**Analyst**: Gianluca Mazza
**Study type**: Exploratory diagnostic (not confirmatory — no pre-registered hypothesis)

## Motivation

Hg1 result (from `benchmarks/appraisal_confound/results.hg1.md`):
- `aft_llm_sync` recall@3 = 0.130 vs `aft_neutral` = 0.315 (Δ = −0.185, d ≈ −1.0).
- Synchronous LLM appraisal is *actively harmful*, not merely inert.
- `aft_llm_dual` (deferred) = 0.315 ≈ `aft_neutral` — dual-path mitigates but adds no benefit.

Root cause is unknown. This runner characterizes the LLM appraisal output quality to guide
the next intervention.

## Dataset

`benchmarks/datasets/realistic_recall_v3.json` — 50 scenarios × 2 sessions, ~250 events.
Each event has oracle `valence` ∈ [−1, +1] and `arousal` ∈ [0, 1].

## Procedure

1. Load all events from the dataset.
2. For each event text: call `LLMAppraisalEngine.appraise(content)` → `AppraisalVector`.
3. Derive `CoreAffect` via `AppraisalVector.to_core_affect()`.
4. Residuals: `res_valence = llm_valence − oracle_valence`, `res_arousal = llm_arousal − oracle_arousal`.
5. Bootstrap 95% CI on mean residuals (n_bootstrap=10000, seed=42, percentile method).
6. Confusion matrix on valence sign (positive = valence ≥ 0).
7. Descriptive stats (mean, std, min, max) on the 5 raw SEC dimensions (no oracle available).

## Frozen parameters

| Parameter | Value |
|---|---|
| Seed | 42 |
| n_bootstrap | 10 000 |
| CI | 95% |
| bias_threshold | 0.10 |
| std_threshold | 0.30 |

## Decision tree (pre-registered)

Evaluated on valence and arousal residuals jointly:

| Condition | Decision |
|---|---|
| \|bias\| > 0.10 AND std > 0.30 | P1d — reframe: document zero-shot LLM appraisal unreliable |
| \|bias\| > 0.10 only | P1b — fix appraisal prompt |
| std > 0.30 only | P1c — add confidence gating |
| neither | P1_OK — investigate other confounders |

## Output

- `benchmarks/appraisal_diagnostics/results.diagnostic.json`
- `benchmarks/appraisal_diagnostics/results.diagnostic.md`

## Frozen — 2026-05-13

No changes to thresholds, dataset, or decision tree after this date.

## Post-freeze correction (2026-05-30)

Documentation-only clarification; thresholds, dataset, and decision tree are unchanged.

The "Dataset" section above states "50 scenarios × 2 sessions, ~250 events". The actual
`realistic_recall_v3.json` the runner loads contains **125 scenarios × 2 sessions = 250
sessions, 750 events**. The frozen prose under-counted (it matches the *affect-free* sibling
`realistic_recall_v3_noAF.json`, which has 50 scenarios). The runner has always iterated all
events in `realistic_recall_v3.json`; only this description was stale. Use `--n` to subsample
if a smaller run is desired. The "Output" filenames are illustrative — runs use
per-model names, e.g. `results.diagnostic.gpt5mini.{json,md}`.
