# Pre-registration Addendum X2 — Hx2: Third-party longitudinal QA retrieval (ES-MemEval/EvoEmo)

**Status:** PRE-REGISTERED (2026-07-07) — committed before any scored run.
**Date (pre-reg):** 2026-07-07
**Dataset:** ES-MemEval v1.0.0 / EvoEmo corpus (WWW 2026; Chen, Lu, Shen & Zhang,
arXiv:2602.01885, ACM DOI 10.1145/3774904.3792143) — third-party, peer-reviewed,
CC-BY-4.0. Zenodo deposit DOI `10.5281/zenodo.18338564` (repo snapshot
`slptongji/ES-MemEval`, GitHub tree `6926242`). File: `data/evo_emo.json`
(sha256 `f30698e87fddaeff51270a666c654da604f487a3456ec60d2b6ae08a6fecd420`), 18 seekers,
401 support sessions (13–33 per seeker, timestamps `YYYY-MM-DD`, span ≈ 15 months),
1,427 QA questions in the released artifact. **Artifact-vs-paper discrepancy, declared
ex-ante:** the paper reports 1,209 QA; the released v1.0.0 JSON contains 1,427. We treat
the released, hash-pinned artifact as ground truth.
**Embedder:** `BAAI/bge-small-en-v1.5` (continuity with Addendum X; head-to-head arm
comparison, not reproduction of the upstream absolute numbers, which used `bge-m3`).
**LLM:** direct-VAD appraisal (`DIRECT_VAD_SCHEMA`, Addendum V) resolved from
`EMOTIONAL_MEMORY_LLM_*` (`.env`). ≈ 1,534 calls (401 session documents at encode +
1,133 in-family queries), LRU-cached.
**Parent closures:** `preregistration_addendum_x_madialbench_third_party_closure.md`
(Hx1 FAIL inverted, construct boundary) · `preregistration_addendum_t_query_appraisal_closure.md`
(Ht1 PASS, curated) · `preregistration_addendum_t2a_naturalistic_query_appraisal_closure.md`
(Ht2a FAIL, naturalistic) · Addendum U (author-crafted benchmarks ~62.5% AFT-favorable).

---

## Motivation

Addendum X produced a powered inverted FAIL on the first third-party corpus
(MADial-Bench): cosine beat oracle-free query-appraised AFT despite near-perfect
appraisal fidelity (D1 AUC=0.996) and an affect-discriminative corpus (D2=76.9%).
The closure attributed the inversion to **construct mismatch**: MADial rewards
counter-congruent supportive recall. X2 is the replication already reserved in that
closure — the second and only other released third-party corpus.

ES-MemEval differs from MADial on exactly the axis that matters: its retrieval task is
**longitudinal QA over emotional-support history** (information extraction, temporal
reasoning, conflict detection, user modeling), not emotion-triggered supportive recall.
Gold evidence marks _where the queried fact was said_, not _what would comfort the user
now_. If the X inversion was truly construct-specific, it should NOT replicate here; if
AFT still loses, the third-party bound hardens regardless of construct.

**Ex-ante regime prior (declared before any appraisal runs):** the corpus is
ESConv-seeded and near-uniformly negative — under the label mapping in D1 below, 393/401
sessions are negative and ~8 positive. We therefore expect **low D2**
(weak affect-discriminativeness), i.e. this corpus likely sits closer to the T2A
naturalistic regime than to MADial. The three readings are fixed now:

1. **Hx2 PASS** → first third-party PASS; external-validity leg for the regime claim.
2. **Hx2 FAIL with low D2** → consistent with the regime bound (U/T2A) on third-party
   data; does not independently confirm the MADial construct boundary.
3. **Hx2 FAIL with high D2** → second third-party failure in a discriminative regime;
   the provenance bound ("author-crafted only") hardens substantially.

Both FAIL readings and the PASS are informative and pre-declared.

---

## Protocol (replicates the upstream `qa_retrieval` session-level evaluation)

Upstream reference: `src/lib/qa_retrieval/qa_retrieval_experiment.py` and
`src/exe/qa_retrieval/qa_retrieval_session.py` in the pinned snapshot.

- **Memory unit — session document (both arms, identical):** one document per support
  session, text = the full transcript rendered as `"<speaker>: <utterance>"` lines
  (seeker name / `supporter`, upstream `SessionWiseMemoryInplaceStrategy._as_text`).
  The session `emotion`, `topic`, `summary`, `observation` annotations and the timestamp
  are **not** injected into the document text — third-party affect labels stay out of the
  semantic channel of both arms (same rule as Addendum X).
- **Queries:** the QA `question` text, verbatim. No time header (upstream adds none for
  qa_retrieval).
- **Gold (session-level, binary):** upstream maps each evidence ref `"<session_id>:<turn>"`
  to the session document containing that turn (`message_indices` intersection). We
  replicate at session granularity: gold = set of session ids referenced by the question's
  evidence. Event-timeline refs (`p*_event_*`, 490 refs) never resolve to session
  documents — identical to upstream, where such questions score 0 on both arms.
- **In-family queries (N=1,133):** questions with ≥1 resolvable session-level gold.
  Excluded (pre-declared): 260 questions with no evidence field (abstention by design),
  24 with event-only evidence, and 10 with unresolvable/malformed refs — 294 zero-gold
  questions total, which upstream scores as 0 for every system and which would add
  identical constant zeros to both arms of a paired test. The zero-gold-inclusive grid
  (N=1,427) is reported as a secondary, non-gating quantity.
- **Candidate pool (replicates upstream, shared by both arms):** per query, 50 candidates
  = all gold sessions + all remaining same-seeker sessions + cross-seeker sessions
  sampled at random to fill to 50, then shuffled (upstream `provided_candidates=50`,
  `_build_candidates_cross_user`). RNG: Python `Random`, one master seed 0, one derived
  seed per query in file order — fixed here, ex-ante. **The pool is constructed once per
  query and shared verbatim by both arms** (upstream re-derives it per system from the
  same seeded procedure; sharing is the paired-design equivalent).
- **Timestamps / recency:** time-invariance config-side via
  `DecayConfig(base_decay=0, arousal_modulation=0, retrieval_boost=0)` (Addendum X
  Amendment A1 pattern, declared here from the start). Session dates are not used in the
  primary arms; s5 remains arousal-gated as part of the affect channel under test.
- **Affective state:** engine state reset to baseline before each query; no `observe()`
  calls; no oracle affect anywhere in the primary arms.

## Arms

1. `naive_cosine` — cosine similarity over session-transcript embeddings within the
   query's 50-candidate pool. Baseline.
2. `aft_query_appraised` — **primary arm**, fully production-reachable and oracle-free:
   encode-time affect from direct-VAD appraisal of the session transcript
   (`LLMAppraisalEngine` + `DIRECT_VAD_SCHEMA`); at retrieve, the query text is appraised
   with the same schema and passed via the public `retrieve(..., query_affect=)` API
   (s3 override, no state mutation). Ranking restricted to the query's candidate pool.
3. `aft_full_stack` — **exploratory, not in family, pre-declared droppable**: as (2) but
   with default `DecayConfig` and encode timestamps from the session `timestamp` field
   (public-API rewrite via `export_memories()`/`import_memories()`), retrieval date =
   the latest session date per seeker. Dropped without penalty if it requires private-API
   surgery (Addendum X precedent).

No `mem0` arm: the X closure dropped it because the adapter required per-corpus surgery;
the same condition applies here a fortiori (per-query candidate pools).

## Hypotheses / quantities

- **Hx2 (primary, the only family member).** `aft_query_appraised` **nDCG@4** >
  `naive_cosine` nDCG@4, per-query paired over the N=1,133 in-family queries, one-tailed
  (directional+). k=4 is the upstream headline (their Table 4: session-level NDCG@4=55.9%
  with bge-m3). **Metric formula replicated verbatim from upstream** `QaRetrievalExperiment.ndcg`,
  including its non-standard rank offset (DCG enumerates from `log2(4)` while IDCG starts
  at `log2(2)`, so attainable nDCG < 1). The distortion is identical for both arms and
  preserves ordering; comparability with the published baseline requires the verbatim
  formula. Standard-formula nDCG is reported as a secondary check.
- **Secondary (reported, non-gating).** Upstream-verbatim Recall/nDCG @{2,4,6} (their
  reported grid); standard-formula MAP/MRR/nDCG/Recall/Precision @{1,3,4,5,10}; the
  zero-gold-inclusive N=1,427 variant; per-capability breakdown (information extraction /
  temporal reasoning / conflict detection / user modeling; abstention has only 3 in-family
  queries and is reported but not interpreted). Δ per metric with 95% bootstrap CI.
- **Diagnostic D1 (appraisal fidelity vs third-party labels).** AUC of encode-time
  direct-VAD valence separating positive-labeled from negative-labeled sessions.
  Label mapping fixed ex-ante on the session `emotion` field: positive =
  {`relief`, `hope`, `hopeful`, `hopefulness`, `pride`}; excluded as ambiguous/compound =
  {`nostalgia`, `mixed emotions`, `conflicted`, `misunderstanding`, any multi-emotion or
  comma-compound label}; all remaining single labels = negative. **Fragility note,
  ex-ante:** the positive class has ~7–8 sessions; D1 is reported with a bootstrap CI and
  is interpreted as descriptive. The appraisal-limited flag (below) still uses the 0.75
  point threshold, read jointly with the CI.
- **Diagnostic D2 (regime check).** Share of in-family queries whose gold sessions' mean
  appraised valence differs from the same seeker's bank mean valence by >0.2 (same
  threshold as Addendum X). This quantifies the low-discriminativeness prior above.

## Statistical analysis plan (pre-declared)

- **Primary metric:** per-query upstream-verbatim nDCG@4 (binary session-level relevance).
- **Test:** paired bootstrap difference, n=10,000, seed=0, one-tailed
  (`benchmarks/common/statistics.paired_bootstrap_diff`); Cohen's d on paired differences.
- **Family correction:** none needed — single primary hypothesis (m=1). All other
  quantities are descriptive/diagnostic and cannot gate the verdict.
- **N:** 1,133 in-family queries. Power note: at per-query nDCG SD ≈ 0.35, the minimal
  detectable Δ at 80% power (one-tailed α=.05) is ≈ 0.026 — an order of magnitude below
  the |Δ| observed in X (0.083). The closure must report the observed SD and implied MDE.

## Decision rule (pre-declared, ex-ante)

`aft_query_appraised` **passes Hx2** iff, vs `naive_cosine`:

1. p < 0.05 (one-tailed, paired bootstrap) on aggregate upstream-verbatim nDCG@4.
2. Δ (appraised − cosine) > 0.
3. 95% bootstrap CI does not cross 0 (all-positive).

Marginal handling: `0.04 < p < 0.05` → "PASS marginal", flagged. No post-hoc threshold
adjustment; no post-hoc metric switching; the result stands as measured.

### Branch A — PASS

First third-party PASS for the oracle-free query-appraisal mechanism.
`claim_validation_matrix.json`: new claim `third_party_retrieval` = `controlled_evidence`,
scoped to longitudinal support-history QA; 08_limitations §2.4 provenance bound amended;
paper gains the external-validity sentence Branch A of Addendum X had reserved.

### Branch B — FAIL

The reading depends on D2, per the ex-ante prior above: low D2 → regime-bound
confirmation on third-party data (the T2A boundary, now externally replicated); high
D2 → the provenance bound hardens (two third-party failures in discriminative regimes).
Either way `claim_validation_matrix.json` `cross_domain_affect_replication` wording is
extended with the measured result, and 08_limitations §2.4 gains the X2 sentence.
Neither outcome invalidates T (curated PASS) or the X construct analysis.

If D1 AUC < 0.75 (read jointly with its CI given the ~8-session positive class), the
closure must flag "appraisal-limited" alongside the verdict.

---

## Scope (explicit)

**In scope:** the two primary arms on ES-MemEval EN (N=1,133 in-family queries,
session-level documents, 50-candidate pools), `bge-small-en-v1.5`, direct-VAD
encode+query appraisal, exploratory arm 3, diagnostics D1/D2.

**Out of scope:** the summarization, dialogue-generation, and abstention _tasks_ of
ES-MemEval (retrieval only); tuning of any retrieval weight, schema, or prompt on this
corpus (zero-shot transfer only); the support-mode retrieval profile (future work — it
must not be tuned on either third-party corpus); human evaluation (#27); push to
`origin/main` without explicit authorisation.

## Execution (planned harness, committed before the run in a separate PR)

```bash
make bench-x2-esmem                                            # full run (requires API key)
uv run python -m benchmarks.esmemeval.runner --dry-run         # smoke run (10 queries, no LLM)
```

Harness: `benchmarks/esmemeval/` (loader pins the sha256 above and fails on mismatch;
asserts 18 seekers / 401 sessions / 1,427 questions / 1,133 in-family; runner reuses
`benchmarks/common/statistics`; metrics module replicates the upstream
`QaRetrievalExperiment.recall`/`ndcg` formulas verbatim, with unit tests against
hand-computed examples; dry-run writes `results.dry.*`, never the scored artifacts).

**Pre-registration integrity:** this document is committed before the harness executes a
single scored run; the closure reports realized per-arm metric grids, Δ/CI/p, Cohen's d,
D1/D2, the MDE, and the Hx2 verdict.

---

## Amendment A1 (2026-07-07, pre-run, before any harness execution)

Made while implementing the harness, before any scored or smoke run. One
descriptive correction:

1. **Regime-prior counts made exact under the D1 mapping.** The Motivation
   section's "393/401 negative and ~8 positive" figure was computed with a
   coarser rule (compound labels counted as negative). Under the D1 mapping
   actually pre-registered (compound/ambiguous labels excluded), the exact
   class counts on the pinned artifact are **7 positive / 376 negative /
   18 excluded**. The prior's substance (near-uniformly negative bank, low-D2
   expectation, three fixed readings) is unchanged.

No hypothesis, metric, decision rule, N, or statistical plan changed.
