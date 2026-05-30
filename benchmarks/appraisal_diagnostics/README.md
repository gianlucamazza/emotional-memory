# Appraisal Diagnostics (WP-1a)

Protocol: [`protocol.md`](protocol.md) (frozen 2026-05-13). Exploratory diagnostic — no
confirmatory hypothesis.

## Purpose

Characterise *why* automatic LLM appraisal fails to help retrieval (Hg1 FAIL — see
[`../preregistration_addendum_g_closure.md`](../preregistration_addendum_g_closure.md)).
Hg1 only told us the net effect was null; it did not say whether the appraisal signal is
**biased**, **noisy**, or both. This runner measures the appraisal output directly against
ground-truth affect so the next intervention is evidence-driven, not a guess.

## Design

For each event in `benchmarks/datasets/realistic_recall_v3.json` (the **oracle** variant,
with preset `valence`/`arousal` per event), call `LLMAppraisalEngine.appraise()`, project to
`CoreAffect` via `AppraisalVector.to_core_affect()`, and compare against the oracle:

- **Residuals** (LLM − oracle) for valence and arousal: bias (mean, bootstrap 95% CI), std,
  MAE, Pearson r with the oracle.
- **Valence sign confusion** (LLM vs oracle, positive = value ≥ 0) with accuracy.
- **Descriptive stats** for the 5 raw SEC dimensions (no oracle for these).

Note: this uses `realistic_recall_v3.json` (oracle affect) as ground truth — *not*
`realistic_recall_v3_noAF.json` (the affect-free dataset Hg1 retrieved on). The oracle is
required here precisely to measure the appraisal error.

## Decision tree (pre-registered, thresholds frozen)

Evaluated jointly on valence and arousal residuals (`bias_threshold=0.10`, `std_threshold=0.30`):

| Condition | Decision |
|---|---|
| \|bias\| > 0.10 **and** std > 0.30 | **P1d** — zero-shot appraisal unreliable; document / consider fine-tuned appraisal |
| \|bias\| > 0.10 only | **P1b** — fix the appraisal prompt (directional error) |
| std > 0.30 only | **P1c** — add confidence gating |
| neither | **P1_OK** — appraisal quality is not the cause; investigate downstream retrieval |

## Execution

```bash
make bench-appraisal-diagnostics-dry   # smoke test, fixed vector, no API key
make bench-appraisal-diagnostics       # full run (requires EMOTIONAL_MEMORY_LLM_API_KEY)
```

Direct invocation also works (the runner loads `.env` if `python-dotenv` is installed):

```bash
uv run python -m benchmarks.appraisal_diagnostics.runner --seed 42 \
  --out-json benchmarks/appraisal_diagnostics/results.diagnostic.<model>.json \
  --out-md   benchmarks/appraisal_diagnostics/results.diagnostic.<model>.md
```

Frozen parameters: `seed=42`, `n_bootstrap=10000`, `CI=95%`. Use `--n` to subsample events
(default: all ~750).

## Outputs

- `results.diagnostic.<model>.json` — residual stats, SEC descriptives, confusion, decision.
- `results.diagnostic.<model>.md` — human-readable report.

## Interpretation guide

| Decision | What it means for the appraisal gap |
|---|---|
| P1b | Prompt is steering the wrong way — rewrite the Scherer CPM prompt (anchors, few-shot). |
| P1c | Direction is right but noisy — aggregate samples / gate on confidence. |
| P1d | Signal is weak in both axes — zero-shot is insufficient; document or fine-tune. |
| P1_OK | Appraisal is fine — the Hg1 null comes from how affect enters retrieval scoring. |
