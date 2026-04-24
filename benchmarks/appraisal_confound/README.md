# Appraisal Confound Study

Pre-registration: [`benchmarks/preregistration_addendum_v2.md`](../preregistration_addendum_v2.md) §Addendum A.

## Purpose

Isolates whether the AFT retrieval advantage over naive cosine comes from the
**6-signal architecture** (mood congruence, resonance, momentum, decay) or from
the **appraisal engine** injecting richer affect signals into the retrieval.

Without this study, the claim "AFT improves retrieval" could be explained by
the appraisal layer doing most of the work — not by the architecture itself.

## Design

Three conditions on `realistic_recall_v1` (same dataset as the main comparative
benchmark):

| Condition | Appraisal | Affect source |
|---|---|---|
| `aft_noAppraisal` | None | Scenario preset (valence, arousal) |
| `aft_keyword` | `KeywordAppraisalEngine` | Inferred from content (deterministic, no LLM) |
| `naive_cosine` | None | N/A — pure cosine baseline |

**No LLM API key required.** All conditions are deterministic.

## Hypotheses (pre-registered)

**Ha2** — `aft_keyword.top1_accuracy > naive_cosine.top1_accuracy`
- Δ > 0 and 95% bootstrap CI excludes 0
- If PASS: AFT architecture adds value beyond raw cosine (even without LLM)

**Hb2** — `|aft_keyword - aft_noAppraisal| < 0.05` (equivalence)
- If PASS: preset and inferred affect are functionally equivalent → architecture is the driver
- If FAIL: keyword appraisal changes retrieval meaningfully → appraisal inference matters

## Execution

```bash
make bench-appraisal-confound        # sbert-bge embedder (default)
make bench-appraisal-confound-hash   # hash embedder (sanity check)
```

## Outputs

- `benchmarks/appraisal_confound/results.json` — scores, Δ, CI, Cohen's d
- `benchmarks/appraisal_confound/results.md` — human-readable report

## Interpretation guide

| Ha2 | Hb2 | Conclusion |
|---|---|---|
| PASS | PASS | Architecture drives the advantage; appraisal engine is neutral on this dataset |
| PASS | FAIL | Both architecture and appraisal inference contribute |
| FAIL | — | No AFT advantage even with keyword appraisal; examine retrieval config |
