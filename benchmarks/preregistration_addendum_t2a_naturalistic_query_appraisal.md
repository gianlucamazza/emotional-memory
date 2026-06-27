# Pre-registration Addendum T2A — Ht2a: Retrieve-time query appraisal on naturalistic dialogue (DailyDialog)

**Status:** EXECUTED (2026-06-27) — **Ht2a FAIL** (query appraisal does not extend to naturalistic
dialogue: appraised vs cosine Δ−0.008, p*holm=1.000; faithful diagnostic valence r=0.69, arousal
r=0.74). See `preregistration_addendum_t2a_naturalistic_query_appraisal_closure.md`.
**Date (pre-reg):** 2026-06-27
**Embedder:** `multilingual-e5-small` (matches Addendum K, so the only changed factor vs Hk1 is the query-affect channel)
**Dataset:** `benchmarks/datasets/dailydialog_personas_v1.json` (N=120 personas, 396 queries — identical to Addendum K Hk1)
**LLM:** direct-VAD appraisal of the query text (`DIRECT_VAD_SCHEMA`, Addendum V), resolved from `EMOTIONAL_MEMORY_LLM*\*` (`.env`). ~480 calls (one per query, cached).
**Parent closures:** `preregistration_addendum_k_dailydialog_closure.md`(Hk1 FAIL, Δ−0.008 ns) ·`preregistration_addendum_t_query_appraisal_closure.md`(Ht1 PASS on curated`realistic_recall_v2`)

---

## Motivation

Addendum T showed, on the **curated** `realistic_recall_v2` benchmark, that replacing the
oracle `query.state` with the query's affect **appraised at retrieve-time** (direct-VAD) lets
AFT beat cosine without any oracle (Ht1 PASS, recovery ~0.59 overall, ~0.82 on the
affect-discriminative subset). The published caveat is explicit: this is **bounded to the
affect-discriminative regime** that `realistic_recall_v2` over-represents (Addendum U: 62.5%
favorable), and it was **never tested on naturalistic, affect-sparse QA**.

Addendum K (Hk1) tested AFT on DailyDialog (real dialogue + synthetic personas) and found
**no advantage** over cosine (Δ−0.008, ns). But Hk1's AFT arm got its query-side affect from
the _leftover runtime state_ (the last ingested session's affect) — it never appraised the
query. The mechanism that moved the boundary in Addendum T was never applied here.

**The question:** does retrieve-time query appraisal — the production-reachable mechanism that
worked on the curated benchmark — recover any advantage on naturalistic affect-conditioned
dialogue, where AFT with stale-state previously tied cosine?

This is the natural "T2A" follow-up named in the Addendum T closure. It uses the new public
`query_affect` API (s3-only override; see CHANGELOG 0.13.0 / `retrieve_with_query_appraisal`).

---

## Arms (same dataset, embedder, top_k, encode path; only the query-affect source differs)

1. `naive_cosine` — `NaiveCosineDailyDialogAdapter` (no affect). Baseline.
2. `aft` — `AFTDailyDialogAdapter` (existing): oracle session PAD injected at **encode**;
   query-side affect = leftover runtime state. = Hk1's AFT arm. Reference.
3. `aft_query_appraised` — **new arm**: identical encode path (oracle session PAD at encode),
   but at retrieve the query's affect is obtained by **appraising the query text** with
   `DIRECT_VAD_SCHEMA` → `CoreAffect` and passed as `query_affect` via the public API
   (`engine.retrieve(query, query_affect=...)`). No oracle on the query side; no state mutation.

The isolation is exactly on the query channel, mirroring Addendum T. The encode side keeps the
oracle session affect (as in Hk1) so the only changed factor vs Hk1's AFT arm is the query-affect
source (stale state → appraised query).

---

## Hypotheses / quantities

- **Ht2a (primary).** `aft_query_appraised` `top1_accuracy` > `naive_cosine` on aggregate
  affect-conditioned recall AND on ≥2 of 3 directional query types (Type 1 emotion-state recall,
  Type 2 affect-conditioned content, Type 3 affective trajectory). One-tailed (directional+).
- **Ht2a-ref (secondary).** `aft_query_appraised` vs `aft` (does appraising the query beat the
  stale-state arm?) — quantifies the mechanism's marginal contribution on naturalistic data.
- **Diagnostic D.** Pearson r between the appraised query affect (valence, arousal) and the
  per-query oracle PAD (the target session's PAD) — explains any recovery gap, comparable to
  Addendum T's r=0.80 / r=0.56.

---

## Statistical analysis plan (pre-declared)

- **Primary metric:** `top1_accuracy` (top-1 hit among top_k=2). **Secondary:** `hit_at_k`.
- **Test:** paired bootstrap difference, n=10,000, seed=0, one-tailed
  (`benchmarks/common/statistics.paired_bootstrap_diff`); McNemar exact two-tailed; Cohen's d.
- **Family correction:** Holm–Bonferroni, m=4 (aggregate + 3 directional types; Type 4 control
  reported, not in family) — identical to Hk1.
- **N:** 120 personas × 4 queries = 480 queries. 95% percentile bootstrap CI on Δ.

---

## Decision rule (pre-declared, ex-ante)

`aft_query_appraised` **passes Ht2a** iff, vs `naive_cosine`:

1. `p_holm` < 0.05 (one-tailed) on aggregate `top1_accuracy`.
2. Δ (`top1_accuracy` appraised − cosine) > 0.
3. 95% bootstrap CI does not cross 0 (all-positive).
4. ≥ 2 of 3 directional query types (Type 1/2/3) show Δ > 0 individually.

Marginal handling: `0.04 < p_holm < 0.05` → "PASS marginal", flagged. No post-hoc threshold
adjustment. Result stands as measured; no post-hoc reframing.

### Branch A — PASS

Retrieve-time query appraisal **generalises** the Addendum T mechanism to naturalistic
affect-conditioned dialogue: AFT (with query appraisal) beats cosine where stale-state AFT
(Hk1) did not.

- Paper: boundary/limitations section reports T2A PASS as evidence the production-reachable
  mechanism extends beyond the curated benchmark (still affect-conditioned, not factual).
- `claim_validation_matrix.json`: note on A2/`downstream_value` — naturalistic generalisation
  of the query-appraisal mechanism established (within affect-conditioned dialogue).

### Branch B — FAIL

Any pass condition unmet.

- Paper: boundary/limitations states the query-appraisal advantage **does not extend** to
  naturalistic affect-conditioned dialogue under this protocol (Δ={observed}, p_holm={observed});
  the Addendum T recovery is bounded to the curated affect-discriminative regime (Addendum U).
- `claim_validation_matrix.json`: the bound is now **measured on naturalistic data**, not merely
  asserted.
- This is a publishable, honest scoping result — it sharpens the state-injection boundary.

**Neither outcome invalidates Addendum T (curated) or the headline.** Both are pre-declared.

---

## Secondary — LoCoMo bound-confirmation (optional, expected FAIL)

If time/budget allow, run the same query-appraisal mechanism on LoCoMo (factual QA, Study S1
FAIL regime; AFT judge-acc 0.279 vs cosine 0.441). Expected FAIL: the Δ−0.159 gap is far larger
than any plausible query-appraisal recovery. Purpose: document that the mechanism does **not**
extend to factual QA. Reported as bound-confirmation, **not** a primary hypothesis; its verdict
does not gate Ht2a.

---

## Scope (explicit)

**In scope:** `aft_query_appraised` vs `naive_cosine` (+ vs `aft`) on DailyDialog personas
(N=120, 480 queries), English, `multilingual-e5-small`, `DIRECT_VAD_SCHEMA` query appraisal.

**Out of scope:** affine arousal calibration of direct-VAD (separate study); multilingual
DailyDialog; human evaluation (#27); other corpora (MELD/MSC); push to `origin/main` without
explicit authorisation.

## Execution

```bash
make bench-t2a-dailydialog                                         # full run (requires API key)
uv run python -m benchmarks.dailydialog.runner \
  --systems aft_query_appraised naive_cosine --dry-run            # quick check (5 personas)
```

**Pre-registration integrity:** committed before execution; the closure reports realized top1
per arm, Δ/CI/p_holm, per-type breakdown, the diagnostic r, the Ht2a verdict, and (if run) the
LoCoMo bound-confirmation.
