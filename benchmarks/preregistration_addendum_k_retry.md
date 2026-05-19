# Pre-registration Addendum K-retry — Hk1 retry: Affective Trajectory (DailyDialog, power-up)

**Status:** PENDING_EXECUTION
**Date (pre-reg):** 2026-05-13
**Embedder:** `multilingual-e5-small`
**Dataset:** DailyDialog personas — new generation, N_personas=600, seed=1
**Parent closure:** `benchmarks/preregistration_addendum_k_dailydialog_closure.md`

---

## Motivation

Addendum K (closure 2026-05-13) produced a clear FAIL on aggregate Hk1, with one positive
trend on the `affective_trajectory` query type:

| Metric | N queries | AFT | naive_cosine | Δ | 95% CI | p_holm | d |
|---|---|---|---|---|---|---|---|
| affective_trajectory | 39 | 0.385 | 0.282 | +0.103 | [−0.077, +0.282] | 0.734 | +0.186 |

The CI crosses zero and p_holm=0.734 because N=39 is severely underpowered. The `affective_trajectory`
type requires a valence-sign-change across sessions (first vs last turn), which occurred in only
39/120 personas (~32.5%). This is structurally expected: the query tests whether AFT's
affective momentum signal tracks temporal valence direction — a theoretically motivated signal.

Power analysis at d=0.186 (observed), alpha=0.05, one-sided:

| N queries | Power |
|---|---|
| 39 (observed) | 0.31 |
| 120 | 0.65 |
| 179 | 0.80 |
| 200 | 0.84 |
| ~195 (expected at N=600 personas) | ~0.82 |

This addendum pre-registers a focused retry on `affective_trajectory` only, with N=600 personas
(expected ~195 valid trajectory queries, power ≈ 0.82), registered before any data are loaded.

---

## Hypothesis Hk1-retry

> On DailyDialog personas — N=600 synthetic personas, seed=1,
> embedder `multilingual-e5-small`, top_k=2 — **AFT `top1_accuracy` > naive_cosine
> `top1_accuracy` on `affective_trajectory` query type**.

Single hypothesis, m=1 (no Holm correction). One-tailed (positive Δ expected).

**Prior FAIL declared:** The aggregate Hk1 test failed. This retry is a focused follow-up
on a single pre-theorised query type with an underpowered positive trend. A second FAIL
closes `cross_domain_affect_replication` as `falsified`.

---

## Dataset construction

Identical to Addendum K §Dataset construction, except:
- **N_personas = 600** (was 120)
- **Seed = 1** (was 0; different seed for independence from original run)
- **Output file:** `benchmarks/datasets/dailydialog_personas_retry.json`

The `affective_trajectory` criterion is unchanged: a valid trajectory query requires that
at least one session has valence-last-turn > valence-first-turn (positive) or < (negative).

---

## Statistical analysis plan

- **Comparison:** AFT vs naive_cosine, `affective_trajectory` queries only.
- **Primary metric:** `top1_accuracy` (top-1 hit among top_k=2).
- **Test:** Paired bootstrap difference, n=10,000, seed=1, one-tailed.
- **McNemar:** exact two-tailed.
- **Effect size:** Cohen's d (paired differences).
- **Family correction:** none (m=1, single hypothesis).
- **Bootstrap CI:** 95% percentile CI on Δ.
- **Minimum N:** ≥ 100 valid trajectory queries required; if below threshold, extend
  persona generation until ≥ 100 are collected (up to N_personas=800 max).

---

## Decision rule

Hk1-retry **PASSES** if and only if:

1. N valid trajectory queries ≥ 100.
2. `p_bootstrap` < 0.05 (one-tailed).
3. Δ (`top1_accuracy` AFT − naive_cosine) > 0.
4. 95% bootstrap CI does not cross 0 (all-positive).

**Marginal:** if `0.04 < p < 0.05`, classify as "PASS marginal" and flag in closure.

### Branch A — PASS

- `cross_domain_affect_replication` → `controlled_evidence` (positive, trajectory only).
- Note in matrix: "retry PASS on affective_trajectory; aggregate Hk1 remains FAIL".
- Paper §6: add retry subparagraph; acknowledge scope is trajectory type, not aggregate.

### Branch B — FAIL

- `cross_domain_affect_replication` → `falsified` (both aggregate Hk1 and retry fail).
- Paper §8: "AFT shows no significant advantage on DailyDialog in two pre-registered studies
  (Hk1 aggregate N=120 FAIL; Hk1-retry trajectory N≥100 FAIL)."
- This is a publishable negative result confirming regime specificity.

---

## Runner

```bash
# Generate personas (seed=1, N=600)
uv run python -m benchmarks.dailydialog.persona_builder \
    --n 600 --seed 1 \
    --out benchmarks/datasets/dailydialog_personas_retry.json

# Run benchmark
uv run python -m benchmarks.dailydialog.runner \
    --personas benchmarks/datasets/dailydialog_personas_retry.json \
    --embedder me5 --seed 1 \
    --out-json benchmarks/dailydialog/results_retry.json \
    --out-md   benchmarks/dailydialog/results_retry.md
```

Results analysis: filter `results_retry.json` for `query_type == "affective_trajectory"` rows.
The runner segregates per-type internally (`runner.py:148-202`).

Closure file: `benchmarks/preregistration_addendum_k_retry_closure.md`

---

## Scope

**In scope:** affective_trajectory query type only, DailyDialog, English, me5 embedder.
**Out of scope:** other query types (already closed in Addendum K); other datasets;
multilingual DailyDialog; LLM appraisal (separate WP-1 diagnostic).

---

## Frozen — 2026-05-13

No changes to hypothesis, N, seed, or decision rule after this date.
Runner command must be executed without modification to constitute a valid pre-registered run.
