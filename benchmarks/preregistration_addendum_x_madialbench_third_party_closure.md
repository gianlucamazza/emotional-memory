# Closure Addendum X — Hx1: Third-party emotion-triggered retrieval (MADial-Bench)

**Status:** EXECUTED (2026-07-02) — **Hx1 FAIL**, decisive (not underpowered)
**Pre-registration:** `preregistration_addendum_x_madialbench_third_party.md` (incl. Amendment A1,
committed pre-run) · harness PR #92 (merged before the scored run)
**Run:** `make bench-x-madial`, 160 queries / 160 memories, `bge-small-en-v1.5`,
direct-VAD appraisal (gpt-5-mini), paired bootstrap n=10,000 seed=0.
**Artifacts:** `benchmarks/madialbench/results.json` / `results.md` / `results.protocol.json`

---

## Verdict

**Hx1 FAIL — and inverted.** `aft_query_appraised` does not beat `naive_cosine` on nDCG@5;
cosine is _significantly better_:

- Δ (appraised − cosine) = **−0.083** [−0.123, −0.043], p_one = 0.9998, Cohen's d = −0.317
- nDCG@5: cosine **0.304** vs AFT **0.221**; the deficit is consistent across the whole
  pre-declared grid (MAP/MRR/nDCG/Recall/Precision @1/3/5/10), worst at rank 1
  (nDCG@1 0.219 vs 0.119)
- **Power:** MDE at 80% power = 0.051 < observed |Δ| = 0.083 → this is a _powered
  negative_, not an inconclusive one. All three pre-declared pass conditions fail.

## Diagnostics — the failure is neither appraisal nor regime

- **D1 (appraisal fidelity vs third-party labels): AUC = 0.996** (Happy vs negative
  memories; mean appraised valence +0.890 vs −0.116; n=109/46). Far above the 0.75
  "appraisal-limited" flag threshold — the mechanism was fairly tested.
- **D2 (corpus affect-discriminativeness): 76.9%** of queries have a gold-set mean
  valence displaced >0.2 from the bank mean — _more_ affect-discriminative than the
  curated v2 benchmark (Addendum U: 62.5%). By the regime criterion of Addenda U/T,
  AFT should have been favored here. It lost anyway.

## Post-hoc exploratory (labeled as such, not pre-registered)

Re-appraising all queries and memories with the same schema and correlating query
valence with gold-set mean valence (N=160):

- corr(query valence, gold mean valence) r = 0.42, but **40.0% of queries have
  opposite-sign gold sets**;
- **84/160 queries carry negative appraised valence** (user in distress at the recall
  point), and for **73.8% of them the gold-set mean valence is positive**;
- the bank skews positive (mean valence +0.58; 109/160 memories labeled Happy).

**Interpretation (construct mismatch).** MADial-Bench operationalizes emotion-triggered
recall as _interpersonal emotion regulation_: the assistant is expected to proactively
recall a (typically positive) memory to support a distressed user. That is
**counter-congruent** recall. AFT's affect channel implements _mood-congruent_ retrieval
(Bower 1981) — prefer memories whose affect matches the query state — so on the majority
class (negative query → positive gold) the affect signals actively push the gold
memories _down_ the ranking. The mechanism did exactly what the theory says; the
benchmark rewards the opposite construct. This also explains why cosine, blind to
affect, wins: semantics alone does not fight the gold direction.

## Bound update (this is the durable claim)

The retrieve-time query-appraisal advantage (Addendum T) is now bounded on **three**
independent axes, all measured:

1. **Regime** (Addendum U/T2A): affect-discriminative queries only; null on
   naturalistic dialogue (Δ=−0.008, ns).
2. **Provenance**: positive evidence exists only on author-crafted corpora; on the one
   released third-party retrieval-native corpus it is significantly negative
   (Δ=−0.083, this study).
3. **Construct** (new): AFT's congruence prior matches _intrapersonal_ mood-congruent
   recall; third-party benchmarks that operationalize _supportive/regulatory_ recall
   reward counter-congruence, where the affect channel becomes an active penalty.

Honest scoping per Branch B: the circularity objection (Addendum U) is **not** broken
by external data — it is sharpened. Neither Addendum T (curated) nor the headline is
invalidated; their scope is narrower than hoped.

## Pre-declared decisions on exploratory arms

- `aft_full_stack` (real dataset dates + default decay): **dropped**, as pre-allowed.
  With the primary arm significantly under cosine, adding decay on 2023–2024 date gaps
  cannot change the verdict and the timestamp rewrite adds protocol surface for no
  inferential value.
- `mem0`: **dropped**, as pre-allowed — the v2 SOTA adapter is dataset-specific
  (affect_reference/realistic formats) and would need per-corpus surgery, which the
  pre-registration explicitly disallows as bias-inducing.

## Follow-ups (not scheduled here)

- **Addendum X2** (separate pre-registration): ES-MemEval/EvoEmo (CC-BY-4.0, Zenodo
  10.5281/zenodo.18338564) as the longitudinal QA-shaped replication corpus.
- **Theory-level question raised by the construct mismatch:** a _support-mode_ retrieval
  profile (counter-congruent weighting for assistant-initiated recall) is the natural
  AFT extension suggested by this result. Any such mode must be designed from theory
  (interpersonal emotion regulation literature) and validated on held-out data — tuning
  it on MADial-Bench would recreate the circularity this study exists to measure.

## Propagation

- `docs/research/08_limitations.md` §2.4 — Addendum X update paragraph (three-axis bound).
- `docs/research/claim_validation_matrix.json` — `cross_domain_affect_replication`
  wording + evidence extended with Addendum X; mirrored verbatim in
  `docs/research/09_current_evidence.md`.
- `paper/main.tex` — limitations update + abstract regime-bound sentence qualified;
  addenda range A–X; arXiv bundle regenerated.
- `ROADMAP.md` — Addendum X recorded; X2 listed as unscheduled follow-up.
