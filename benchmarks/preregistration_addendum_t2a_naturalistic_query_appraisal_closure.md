# Closure — Addendum T2A: Retrieve-time query appraisal on naturalistic dialogue (DailyDialog)

**Status:** EXECUTED 2026-06-27 · **Ht2a FAIL — the query-appraisal advantage does not extend
to naturalistic affect-conditioned dialogue**
**Pre-registration:** `preregistration_addendum_t2a_naturalistic_query_appraisal.md` (committed before execution)
**Runner:** `benchmarks/dailydialog/t2a_runner.py` · **Artifact:** `benchmarks/dailydialog/t2a_results.{json,md,protocol.json}`
**Config:** `dailydialog_personas_v1.json` (N=120 personas, 396 queries), `multilingual-e5-small`,
direct-VAD query appraisal (`DIRECT_VAD_SCHEMA`) via the public `query_affect` API,
paired bootstrap n=10,000, seed=0, Holm m=4.

## Result

Three arms; encode path identical (oracle session PAD at encode), only the query-affect source differs.

| Arm                 | top1  | hit@k |
| ------------------- | ----- | ----- |
| naive_cosine        | 0.220 | 0.389 |
| aft (stale-state)   | 0.202 | 0.364 |
| aft_query_appraised | 0.212 | 0.386 |

| Contrast                                 | Δ [95% CI]                  |    p_holm |
| ---------------------------------------- | --------------------------- | --------: |
| **aft_query_appraised vs cosine (Ht2a)** | **−0.008 [−0.056, +0.040]** | **1.000** |
| aft_query_appraised vs aft (Ht2a-ref)    | +0.010 [−0.013, +0.033]     |     0.224 |
| aft vs cosine (Hk1 reproduction)         | −0.018 [−0.066, +0.030]     |     0.753 |

Directional types passing Holm: **0/3**. **Ht2a verdict: FAIL** (Branch B).

Diagnostic — appraised query affect vs target-session oracle PAD: **valence r=0.69, arousal r=0.74**.

## Interpretation

**The query appraisal worked; the regime is the bottleneck.** The diagnostic shows the appraised
query affect tracks the target session's oracle PAD as well as in Addendum T (valence r=0.69,
arousal r=0.74, vs T's 0.80/0.56). So the FAIL is **not** an appraisal-quality failure — the
affect signal is faithful. It simply does not discriminate on naturalistic, affect-sparse
dialogue retrieval, where lexical/semantic overlap (cosine) already captures the relevant signal.

Query appraisal recovers a negligible, non-significant +0.010 over the stale-state arm
(Ht2a-ref), and the system still ties cosine (Δ−0.008, ns). The aft-vs-cosine reproduction
(Δ−0.018) reproduces the published Hk1 null (Δ−0.008).

**This bounds Addendum T precisely.** T showed retrieve-time query appraisal is
production-reachable (no oracle) **on the curated `realistic_recall_v2` benchmark**, which
Addendum U found is ~62.5% affect-discriminative by construction. T2A tests the same mechanism
on naturalistic affect-conditioned dialogue and finds **no advantage**: the production-reachable
recovery is confined to the affect-discriminative regime and does **not** generalize to
naturalistic QA. The state-injection boundary is partially dissolved _within that regime only_.

## Claim-matrix impact

- `downstream_value` / A2: the naturalistic limit of the query-appraisal mechanism is now
  **measured**, not merely asserted. Addendum T's "production-reachable" advantage is
  regime-bound (affect-discriminative); on DailyDialog query appraisal neither helps nor hurts.
- `affective_dialogue_replication` (Hk1): T2A is a stronger negative — even with faithful
  retrieve-time query appraisal, AFT does not beat cosine on naturalistic short-turn dialogue.
- `08_limitations` §2.4 and `problem_register_2026-06.md` (naturalistic re-test item) updated.

## Secondary — LoCoMo bound-confirmation

Not executed. The DailyDialog FAIL with a faithful diagnostic (r 0.69/0.74) already establishes
that query appraisal does not extend beyond the affect-discriminative regime; LoCoMo (factual QA,
Δ−0.159 gap) would only re-confirm a larger negative and was de-scoped to save LLM budget. The
expectation (FAIL) is documented in the pre-registration; running it is left as optional future
confirmation.

**Honest framing.** Ht2a is a pre-registered negative, reported as measured. It does not
invalidate Addendum T (curated) or the headline; it sharpens their scope: the query-appraisal
mechanism is production-reachable **and bounded to the affect-discriminative regime**.
