# Closure — Addendum T (A): Retrieve-time query appraisal vs oracle state-injection

**Status:** EXECUTED 2026-06-27 · **Ht1 PASS — production-reachable**
**Pre-registration:** `preregistration_addendum_t_query_appraisal.md` (committed before execution)
**Runner:** `benchmarks/query_appraisal/runner.py` · **Artifact:** `benchmarks/query_appraisal/results.{json,md}`
**Config:** `realistic_recall_v2` (N=200), `sbert-bge`, direct-VAD query appraisal (`DIRECT_VAD_SCHEMA`),
paired bootstrap n=2000, seed=42.

## Result

Three arms; only the query-affect source differs (oracle `query.state` vs direct-VAD on the query text).

| Contrast                                | top1           | Δ [95% CI]                  |      p |
| --------------------------------------- | -------------- | --------------------------- | -----: |
| aft_oracle vs cosine (upper bound)      | 0.520 vs 0.325 | +0.195 [+0.145, +0.250]     | <0.001 |
| **aft_query_appraised vs cosine (Ht1)** | 0.440 vs 0.325 | **+0.115 [+0.055, +0.180]** | <0.001 |
| aft_query_appraised vs aft_oracle (gap) | 0.440 vs 0.520 | −0.080 [−0.125, −0.040]     | <0.001 |

**Recovery fraction** (appraised−cosine)/(oracle−cosine) = **0.59**.

Affect-favorable subset (Addendum U criterion, N=125): oracle Δ=+0.304 [0.224, 0.384],
**appraised Δ=+0.248 [0.168, 0.320]** → recovery ≈ **0.82** where affect discriminates.

Diagnostic — corr(appraised query affect, oracle state): valence r=0.80, arousal r=0.56.

## Interpretation

**This is the first mechanism that moves the state-injection boundary (A2) rather than only
characterizing it.** Every prior attempt that kept the oracle state and reshaped the channel
failed (Hj1 tuning, Hl routing, Hq gating, Hp recalibration). Here the oracle `query.state` is
**replaced** by appraising the query text at retrieve-time — a production-reachable operation —
and AFT **still beats cosine** (+0.115, p<0.001), recovering **~59% of the oracle advantage**
overall and **~82% on the affect-discriminative subset**.

Why it works: the diagnostic shows the appraised query affect tracks the oracle state (valence
r=0.80, arousal r=0.56), so direct-VAD supplies most of the discriminating signal the oracle
injected. The residual gap (−0.080) is the price of imperfect query appraisal — consistent with
Addendum V (valence well estimated, arousal weaker).

**Scope / honest caveats.** (1) Bounded to the affect-discriminative regime that
`realistic_recall_v2` over-represents (Addendum U: 62.5% favorable; the recovery concentrates
there). (2) Still SBERT on the curated benchmark; **not** yet shown on naturalistic, affect-sparse
QA — the LoCoMo/DailyDialog FAILs are untested with query appraisal (the natural next study,
"T2A"). So "production-reachable" means **the oracle is not required** for the affect-discriminative
advantage, not that AFT now wins on factual/naturalistic QA.

## Claim-matrix impact

Re-frames A2 and bounds `downstream_value`: the headline advantage is **largely production-reachable
without oracle affect** — appraising the query at retrieve-time recovers ~59% (≈82% on the
affect-discriminative subset) while still beating cosine. The state-injection boundary is **partially
dissolved**, not closed. `08_limitations` §2.4 updated. Next: a production engine API for query
appraisal + a naturalistic re-test (LoCoMo/DailyDialog with query appraisal).
