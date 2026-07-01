# Pre-registration Addendum X — Hx1: Third-party emotion-triggered retrieval (MADial-Bench)

**Status:** EXECUTED (2026-07-02) — **Hx1 FAIL** (Δ=−0.083 [−0.123, −0.043], p*one=0.9998;
cosine significantly ahead despite faithful appraisal D1 AUC=0.996 and an
affect-discriminative corpus D2=76.9% — construct mismatch: the benchmark rewards
counter-congruent supportive recall). See
`preregistration_addendum_x_madialbench_third_party_closure.md`.
**Date (pre-reg):** 2026-07-02
**Dataset:** MADial-Bench EN (NAACL 2025, Long 499; He et al., arXiv:2409.15240) — third-party,
peer-reviewed, MIT license. Repo `hejunqing/MADial-Bench`, pinned commit
`572e3a10d6d01852a65e4508e0b3ab2a00d0710c` (2025-04-30). Files:
`data/en/MADial-Bench-en-dialogue.json` (sha256 `ba987172b4a720dd108c9a3b04855b8489ecde96fdf5d484d6522602ec6f4a31`, 160 dialogues,
JSONL) and `data/en/MADial-Bench-en-memory.json` (sha256 `d384b2d35e01ed364165001911af7f8193c808d168b106dea7b23435c7717aa5`,
160 memories, JSONL, ids 1–160 contiguous, each `{time, scene, emotion, event}`).
**Embedder:** `BAAI/bge-small-en-v1.5` (matches the `realistic_recall_v2` headline runs; EN corpus)
**LLM:** direct-VAD appraisal (`DIRECT_VAD_SCHEMA`, Addendum V) resolved from
`EMOTIONAL_MEMORY_LLM*\*` (`.env`). ~320 calls (160 memories at encode + 160 queries), cached.
**Parent closures:** `preregistration_addendum_t_query_appraisal_closure.md`(Ht1 PASS, curated) ·`preregistration_addendum_t2a_naturalistic_query_appraisal_closure.md`(Ht2a FAIL, naturalistic) ·`benchmarks/circularity/` Addendum U (author-crafted benchmark ~62.5% AFT-favorable)

---

## Motivation

Every positive retrieval result to date sits on **author-crafted** corpora (`affect_reference_v1`,
`realistic_recall_v2`), whose affect labels were written with AFT in mind (Addendum U:
~62.5% of queries AFT-favorable by construction). The one naturalistic re-test (T2A,
DailyDialog) FAILED, bounding the query-appraisal mechanism to the affect-discriminative
regime. What has never been tested: a corpus that is **affect-discriminative by third-party
construction** — emotionally triggered memory recall as defined by external authors, with
gold relevance annotations we did not create.

MADial-Bench is exactly that: memory-anchored dialogues where assistant recall is triggered
by the user's emotional state (grounded in interpersonal emotion-regulation strategies and a
two-phase theory of human recall), per-memory `emotion` annotations, ranked gold memory sets
per dialogue, and published embedding baselines whose authors conclude that _"text similarity
retrieval is inadequate for the memory recall process"_. It is peer-reviewed (NAACL 2025),
MIT-licensed, and data-complete in the public repo.

**The question:** does AFT with retrieve-time query appraisal (the production-reachable
Addendum T mechanism, fully oracle-free) beat cosine on a third-party affect-discriminative
corpus? A PASS breaks the circularity objection with external data; a FAIL tightens the
regime bound to "author-crafted only", which would substantially weaken the production
claim. Both outcomes are informative and pre-declared.

Dataset-selection audit (2026-07-02, before any run): HLME/MemEmo (arXiv:2602.23944) —
not publicly released; ENPMR-Bench (arXiv:2605.27240) — release URL in paper returns 404
("upon acceptance"); ES-MemEval/EvoEmo (WWW 2026, CC-BY-4.0, Zenodo 10.5281/zenodo.18338564) —
released but QA-shaped (evidence passages, not a native retrieval gold set), reserved as the
follow-up replication corpus (Addendum X2, separate pre-registration). MADial-Bench is the
only released retrieval-native candidate.

---

## Protocol (replicates the MADial-Bench evaluation, from the repo's `Embedding.py`)

- **Queries (N=160, one per dialogue):** query text = `"<Time>: 2024-06-16\n"` + all dialogue
  turns with index `i < test_turn[0]` (context up to the FIRST test turn, exclusive), verbatim
  concatenation as in the repo. Gold = the dialogue's `relevant-id` set, binary relevance
  (their metric implementation ignores gold ranking; we replicate this).
- **Memory bank:** all 160 memories, no user filtering (the EN memory file carries no user-id;
  the repo's own `calculate_similarity` retrieves over the full bank). Gold sets are naturally
  user-disjoint (verified: zero overlap between user 1 and user 2 gold ids).
- **Document text (both arms, identical):** the `event` field only. The `emotion`/`scene`/`time`
  labels are NOT injected into the document text — this keeps the third-party affect label out
  of the semantic channel of both arms. (The repo's baselines embed the serialized dict
  including the emotion label; our published-baseline comparison is therefore indicative only.)
- **Timestamps / recency (amended pre-run, 2026-07-02 — see Amendment A1):** dataset dates are
  NOT used in the primary arms. Time-invariance is implemented config-side via
  `DecayConfig(base_decay=0, arousal_modulation=0, retrieval_boost=0)`, which makes stored
  strength independent of elapsed time (equivalent to uniform encode timestamps, and robust to
  encode-loop duration). Note an implementation fact the original text got wrong: s5 is the
  _decayed consolidation strength_, whose initial value is arousal-gated
  (`consolidation_strength()`, inverted-U, McGaugh 2004) — so even time-invariant, s5 remains
  a function of the memory's appraised arousal. This is part of AFT's affect channel and the
  production default; it stays active. What this bullet removes is only the _time/file-order_
  confound, not the arousal-gated strength.
- **Affective state:** engine state reset to baseline before each query; no `observe()` calls.
  No oracle affect anywhere in the primary arms.

## Arms

1. `naive_cosine` — cosine similarity over `event`-text embeddings. Baseline.
2. `aft_query_appraised` — **primary arm**, fully production-reachable: encode-time affect
   from direct-VAD appraisal of the memory `event` text (`LLMAppraisalEngine` +
   `DIRECT_VAD_SCHEMA`); at retrieve, `retrieve_with_query_appraisal()` appraises the query
   text (same schema) — s3 override via the public `query_affect` API, no state mutation.
   Uniform timestamps.
3. `aft_full_stack` — **exploratory, not in family**: as (2) but with default `DecayConfig`
   and encode timestamps from the memory `time` field, retrieval date 2024-06-16 (the
   dataset's own `<Time>` header), activating ACT-R decay on real third-party dates.
   Conditional on public-API feasibility (timestamp rewrite via `export_memories()` /
   `import_memories()`); dropped without penalty if it would require private-API surgery.
4. `mem0` — **exploratory, not in family**: Mem0 adapter from the v2 SOTA harness if it runs
   unmodified on this corpus (symmetric, no oracle); dropped without penalty if the adapter
   requires per-corpus surgery (any such surgery would itself bias the comparison).

## Hypotheses / quantities

- **Hx1 (primary, the only family member).** `aft_query_appraised` **nDCG@5** > `naive_cosine`
  nDCG@5. One-tailed (directional+).
- **Secondary (reported, non-gating).** Recall@5, MAP@5, MRR@5, and the full
  MAP/MRR/nDCG/Recall/Precision @1/3/5/10 grid (their metric implementations, replicated) for
  both arms; Δ per metric with 95% bootstrap CI.
- **Diagnostic D1 (appraisal fidelity vs third-party labels).** AUC of encode-time direct-VAD
  valence separating `emotion=Happy` (n=109) from negative-emotion memories
  (Anxious/Sad/Angry/Disappointed/Fear/Frustrated, n≈45); plus mean valence per label group.
  Analogous to T2A's fidelity diagnostic: distinguishes "mechanism failed" from
  "appraisal failed".
- **Diagnostic D2 (regime check).** Share of queries where the gold set's mean appraised
  valence differs from the bank mean by >0.2 — an affect-discriminativeness estimate of the
  corpus, comparable to Addendum U's 62.5% figure.

## Statistical analysis plan (pre-declared)

- **Primary metric:** per-query nDCG@5 (binary relevance, their formula).
- **Test:** paired bootstrap difference, n=10,000, seed=0, one-tailed
  (`benchmarks/common/statistics.paired_bootstrap_diff`); Cohen's d on paired differences.
- **Family correction:** none needed — single primary hypothesis (m=1). All other quantities
  are descriptive/diagnostic and cannot gate the verdict.
- **N:** 160 queries. Power note: at nDCG SD ≈ 0.35 (typical for @5 binary), the minimal
  detectable Δ at 80% power (one-tailed α=.05) is ≈ 0.07 — smaller true effects may return
  an inconclusive FAIL; the closure must report the observed SD and the implied MDE.

## Decision rule (pre-declared, ex-ante)

`aft_query_appraised` **passes Hx1** iff, vs `naive_cosine`:

1. p < 0.05 (one-tailed, paired bootstrap) on aggregate nDCG@5.
2. Δ (appraised − cosine) > 0.
3. 95% bootstrap CI does not cross 0 (all-positive).

Marginal handling: `0.04 < p < 0.05` → "PASS marginal", flagged. No post-hoc threshold
adjustment; no post-hoc metric switching; the result stands as measured.

### Branch A — PASS

First evidence that the AFT advantage survives on a **third-party** affect-discriminative
corpus, oracle-free, production-reachable end to end. Paper: circularity bound (Addendum U,
08_limitations §2.4) is amended — the regime claim no longer rests solely on author-crafted
data; headline gains an external-validity leg. `claim_validation_matrix.json`: new claim
`third_party_retrieval` = `controlled_evidence` (scoped to emotion-triggered recall).

### Branch B — FAIL

Paper: the affect-discriminative regime advantage is **not yet shown outside author-crafted
corpora** — the bound sharpens from "regime-bound" to "regime- and provenance-bound"; this
must be stated in 08_limitations §2.4 and the abstract's production-reachability sentence
must be qualified accordingly. `claim_validation_matrix.json`: A2 note updated with the
measured third-party null. Honest scoping either way; neither outcome invalidates T (curated)
or T2A (naturalistic FAIL).

If D1 AUC < 0.75 (appraisal cannot separate Happy from negative on this corpus), the closure
must flag "appraisal-limited" alongside the verdict — the mechanism was not fairly tested and
a schema/prompt follow-up (not a re-run of Hx1) is the next step.

---

## Scope (explicit)

**In scope:** the two primary arms on MADial-Bench EN (160 queries, full bank, event-only
documents), `bge-small-en-v1.5`, direct-VAD encode+query appraisal, exploratory arms 3–4,
diagnostics D1/D2.

**Out of scope:** the ZH split (appraisal prompt is EN-validated only); tuning of any
retrieval weight, schema, or prompt on this corpus (zero-shot transfer only — any tuning
would recreate the circularity this study exists to break); ES-MemEval/EvoEmo (Addendum X2,
separate pre-registration); human evaluation (#27); push to `origin/main` without explicit
authorisation.

## Execution (planned harness, committed before the run in a separate PR)

```bash
make bench-x-madial                                            # full run (requires API key)
uv run python -m benchmarks.madialbench.runner --dry-run       # smoke run (10 queries, no LLM)
```

Harness: `benchmarks/madialbench/` (loader pins the two sha256 above and fails on mismatch;
runner reuses `benchmarks/common/statistics`; metrics module replicates the repo's
`embedding_score_new.py` formulas verbatim, with unit tests against hand-computed examples).

**Pre-registration integrity:** this document is committed before the harness executes a
single scored run; the closure reports realized per-arm metric grids, Δ/CI/p, Cohen's d,
D1/D2, the MDE, and the Hx1 verdict.

---

## Amendment A1 (2026-07-02, pre-run, before any harness execution)

Made while implementing the harness, before any scored or smoke run. Two corrections:

1. **Timestamps bullet corrected.** The original text claimed uniform encode timestamps make
   s5 "constant across memories". Factually wrong: s5 is the decayed _consolidation strength_,
   whose initial value is arousal-gated (`consolidation_strength()`, inverted-U). The intent
   (remove the time/file-order confound) is now implemented via
   `DecayConfig(base_decay=0, arousal_modulation=0, retrieval_boost=0)` — declared here
   ex-ante, not fitted to any result; the arousal-gated strength remains active as part of
   the affect channel under test.
2. **Arm 3 feasibility scoped.** `encode()` takes no timestamp parameter; the real-dates
   exploratory arm is conditional on public-API timestamp rewrite
   (`export_memories()`/`import_memories()`) and is dropped without penalty otherwise.

No hypothesis, metric, decision rule, N, or statistical plan changed.
