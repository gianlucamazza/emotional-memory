# Closure Addendum X2 — Hx2: Third-party longitudinal QA retrieval (ES-MemEval/EvoEmo)

**Status:** EXECUTED (2026-07-07) — **Hx2 FAIL**, decisive (not underpowered).
**Corrected 2026-07-07 (see Data correction below):** direct-VAD re-run after a
harness bug; verdict unchanged, numbers updated.
**Pre-registration:** `preregistration_addendum_x2_esmemeval_third_party.md` (incl.
Amendment A1, committed pre-run) · harness PR #103 (merged before the scored run)
· data-correction PR #112.
**Run:** `make bench-x2-esmem`, 1,133 in-family queries / 401 session documents /
50-candidate pools, `bge-small-en-v1.5`, direct-VAD appraisal (gpt-5-mini),
paired bootstrap n=10,000 seed=0.
**Artifacts:** `benchmarks/esmemeval/results.json` / `results.md` / `results.protocol.json`

---

## Data correction (2026-07-07)

The **first** scored run (committed in the original closure and released in v0.15.0)
was affected by a harness bug: `AFTQueryAppraisedEsmemAdapter.ingest` closed a
temporary engine that shared the LLM appraiser, closing its httpx client. With
`fallback_on_error=True`, all ~1,133 **retrieve-time query appraisals silently fell
back to the keyword appraiser instead of the pre-registered direct-VAD schema**
(the 401 encode-time session appraisals ran before the close and were unaffected, so
D1/D2 remained valid). Fixed in PR #112 (ingest no longer closes the shared
appraiser); the study was re-run with genuine direct-VAD query appraisal.

**Impact — verdict unchanged, magnitude modestly smaller.** Direct-VAD query affect
is slightly less harmful than the keyword fallback on this corpus, but AFT still loses
decisively: u_nDCG@4 AFT **0.133** (was 0.120) vs cosine 0.284; **Δ=−0.150** (was −0.164)
[−0.165, −0.136], p_one=1.000, d=−0.622, MDE 0.018 — still a powered inverted FAIL.
D1 AUC 0.950 (was 0.971), D2 63.0% (was 68.2%) — both still support "appraisal faithful,
corpus affect-discriminative". The corrected run's neutral-query fraction (45.5%, |v|<0.2)
matches the original post-hoc (45.2%), which had used a separate healthy appraiser and was
therefore already correct. All numbers below and in downstream docs are the corrected
direct-VAD values. **MADial-Bench (Addendum X) is unaffected** (persistent engine, no
close in ingest).

## Verdict

**Hx2 FAIL — and inverted, again.** `aft_query_appraised` does not beat `naive_cosine`
on upstream-verbatim nDCG@4; cosine is _significantly better_, by twice the Addendum X
margin:

- Δ (appraised − cosine) = **−0.150** [−0.165, −0.136], p_one = 1.0000, Cohen's d = −0.622
- u_nDCG@4: cosine **0.284** vs AFT **0.133** (standard-formula nDCG@4: 0.528 vs 0.250);
  the deficit is consistent across all 31 pre-declared grid contrasts (every Δ negative,
  every 95% CI fully below zero), across all four interpreted capabilities
  (Δ from −0.131 to −0.166), and is worst at rank 1 (nDCG@1 0.378 vs 0.151)
- Zero-gold-inclusive variant (N=1,427): u_nDCG@4 0.225 vs 0.106 — same picture
- **Power:** MDE at 80% power = 0.018 ≪ observed |Δ| = 0.150 → a _powered negative_ by
  an order of magnitude. All three pre-declared pass conditions fail.
- Sanity vs published baseline: our cosine arm's u_Recall@4 = 0.595 is in line with the
  upstream session-level 65.0% (they used `bge-m3`; we pre-registered `bge-small-en-v1.5`).

## Diagnostics — appraisal faithful; the low-D2 prior was wrong

- **D1 (appraisal fidelity vs third-party labels): AUC = 0.950** [0.901, 0.988]
  (positive vs negative session labels, n=7/376; mean appraised valence +0.633 vs
  −0.175). Above the 0.75 flag threshold with the CI entirely above it — despite the
  tiny positive class, the mechanism was fairly tested; no "appraisal-limited" flag.
- **D2 (corpus affect-discriminativeness): 63.0%** of queries have a gold-set mean
  valence displaced >0.2 from the seeker's bank mean — **contrary to the ex-ante
  low-D2 prior** (the near-uniformly negative _labels_ hid real continuous-valence
  variation within the negative range). By the pre-declared three-reading scheme this
  selects reading (iii): a second third-party failure **in an affect-discriminative
  regime** — the provenance bound hardens.

## Post-hoc exploratory (labeled as such, not pre-registered)

Re-appraising all 1,133 queries and 401 sessions with the same schema:

- corr(query valence, gold-set mean valence) r = **+0.25** — weakly _aligned_, not
  counter-congruent (Addendum X: 40.0% opposite-sign gold sets; here **13.1%**);
- **45.2% of queries are affectively neutral** (|valence| < 0.2) — QA questions about
  facts, sequences, and consistency mostly carry no affect;
- mean query valence −0.166 vs mean gold valence −0.130 — both mildly negative, aligned.

**Interpretation (affect-orthogonal gold).** The Addendum X construct explanation does
NOT transfer: nothing here rewards counter-congruent recall. ES-MemEval's gold is
**content-determined** — the right session is the one where the queried fact was said,
whatever its affect. On such a task the affect channel contributes ranking variance
with no gold-directed signal (nearly half the queries are affectively neutral, and the
weak positive congruence that exists is already captured by semantics), while the
composite score pays roughly half its weight for it. Long-transcript documents compress
the cosine signal into a narrow range, so the affect noise dominates ordering — hence a
deficit twice X's, at every rank. This is a **second, distinct failure mode** on
third-party data: X = affect channel actively opposed (counter-congruent construct);
X2 = affect channel uninformative (affect-orthogonal gold).

## Bound update (this is the durable claim)

The retrieve-time query-appraisal advantage (Addendum T) remains bounded on three
measured axes, with the **provenance** axis now substantially hardened:

1. **Regime** (Addendum U/T2A): affect-discriminative queries only; null on naturalistic
   dialogue.
2. **Provenance** (X + X2): positive evidence exists only on author-crafted corpora.
   Both released third-party retrieval-native corpora produce powered _inverted_ FAILs
   (Δ=−0.083 and Δ=−0.150), each in an affect-discriminative regime (D2 76.9% / 63.0%),
   each with faithful appraisal (D1 0.996 / 0.950).
3. **Construct** (X, refined by X2): the failure mode is corpus-dependent —
   counter-congruent supportive recall on MADial-Bench, affect-orthogonal
   content-determined gold on ES-MemEval. What the two share: the corpus's gold
   relation was not authored around mood-congruence, and then the affect channel does
   not pay for its weight.

Honest scoping per Branch B: the D2 criterion alone does not predict third-party
success — an affect-discriminative corpus (in the D2 sense) can still have
content-determined gold. The regime claim's operative boundary is thus better stated as
"the gold relation itself must be affect-conditioned", which to date has only been
observed in author-crafted corpora. Neither Addendum T (curated PASS) nor the X
construct analysis is invalidated.

## Pre-declared decisions on exploratory arms

- `aft_full_stack` (real session dates + default decay): **dropped**, as pre-allowed.
  With the primary arm significantly under cosine at every rank, activating ACT-R decay
  on 2024–2025 date gaps cannot change the verdict and the timestamp rewrite adds
  protocol surface for no inferential value (same logic as Addendum X).
- No `mem0` arm was pre-registered (X precedent: per-corpus adapter surgery disallowed).

## Follow-ups (not scheduled here)

- **Support-mode retrieval profile** (from Addendum X) is _unaffected_ by X2: X2's
  failure mode is orthogonality, not counter-congruence, so a support-mode profile
  would not have rescued it. The theory-level fix X2 suggests instead is
  **affect-gating**: engage the affect channel only when the query carries affect
  (45.2% here do not) — note Addendum Q found state-based gating recovers the always-on
  penalty but does not produce gains; a query-affect-conditioned gate is the untested
  variant.
- Third released third-party corpus: none currently available (HLME unreleased,
  ENPMR-Bench repo 404 — see the X pre-registration audit); the provenance bound can
  only be revisited when new retrieval-native emotional corpora are released.

## Propagation

- `docs/research/08_limitations.md` §2.4 — Addendum X2 update paragraph (provenance
  bound hardened; two distinct failure modes).
- `docs/research/claim_validation_matrix.json` — `cross_domain_affect_replication`
  wording + evidence extended with Addendum X2; mirrored verbatim in
  `docs/research/09_current_evidence.md`.
- `paper/main.tex` — limitations + conclusion updated; addenda range A–X2; arXiv bundle
  regenerated.
- `ROADMAP.md` — Addendum X2 recorded; `CHANGELOG.md` — Unreleased/Research entry.
- Residual surface (separate PR): `README.md`, `benchmarks/README.md` (table + ladder),
  `docs/research/index.md`, problem register, `10_scientific_quality_bar.md`.
