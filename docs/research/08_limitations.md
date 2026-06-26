# Known Limitations of Affective Field Theory

> This section documents the acknowledged limits of the current implementation.
> Transparency about limits is integral to any claim of scientific rigor.

---

## 1. Limits of the Affective Model

### 1.1 Affect dimensionality

Since v0.8.2, `CoreAffect` is a **3D PAD** (Pleasure-Arousal-Dominance) value
object (Mehrabian & Russell 1974), implemented as a continuous coordinate
`(valence, arousal, dominance)`. The dominance dimension was promoted from a
`MoodField` heuristic to a first-class field in commit `8b9ddbe`. Remaining
limitations:

- **Contested circumplex structure**: Russell (2003) himself revised the
  theory toward "core affect" as a contextual construction. The current model
  uses the classic 1980 form + Mehrabian dominance extension, not the
  constructionist revisions.
- **Limited emotional granularity**: the Plutchik mapping to 8 primary emotions
  is a discretization of the continuous space. Complex emotions (nostalgia,
  Schadenfreude, awe) have no direct representation.
- **Dominance estimation**: `KeywordAppraisalEngine` infers dominance from
  `coping_potential` vocabulary; `LLMAppraisalEngine` delegates to the Scherer
  CPM prompt. Cross-cultural validity of dominance estimates is not evaluated.

### 1.2 Language dependence

- **Validated robustly in English only**: the headline realistic-recall advantage
  (SBERT d=0.49, e5-small-v2 d=0.31, N=200) is robust and multi-embedder. Italian
  (Hd2_IT, N=80) shows a significant hit@k advantage but top1 non-significant;
  a pre-registered power top-up to N=120 on multilingual-e5-small returns
  Δ=+0.058 (p=0.276) — FAIL at declared power. Spanish (Hd2_ES, SBERT, N=80):
  Δ=+0.138, p=0.045 — a single directional positive, not power-replicated;
  me5 FAIL at both N=80 (Δ=+0.113, p=0.110) and N=120 (Δ=0.000, p=1.00).
  See power top-up closure `benchmarks/preregistration_addendum_hd2_powertopup_closure.md`.
- **`KeywordAppraisalEngine` is English-only**: the `LLMAppraisalEngine` handles
  multilingual input, but appraisal quality varies by language.
- **Cross-lingual validation: EN/IT/ES/FR confirmed, broader scope open**: English
  (robust), French (Hm1 PASS, me5 N=120), and Spanish-SBERT exploratory (N=80) are
  validated; Italian and Spanish me5 FAIL at N=120 power. Generalisation to other
  languages is not established. Larger multilingual embedders (e.g., `BGE-M3`) are a
  natural next step.

---

## 2. Limits of Empirical Validation

### 2.1 No validation with human users

The 127 psychological fidelity tests validate that the system behaves
_coherently with the theories_ it implements (for example, that retrieval is
mood-congruent and that decay follows a power law). They do not validate that
the system's behavior corresponds to how _real human beings_ form and retrieve
emotional memories.

This is the critical distinction between **intra-theoretical validation**
(tested) and **ecological validation** (not tested). A system that faithfully
implements a wrong theory passes all fidelity tests.

### 2.2 Synthetic comparative benchmarks

The repository includes a controlled comparative benchmark today
(`benchmarks/comparative/`) on `affect_reference_v1`, a synthetic 258-example
affect-labeled dataset with 4 mood-congruent retrieval queries. This benchmark
is useful for measuring whether AFT changes ranking in the theoretically
expected direction, but it has important limits:

- **Small and synthetic setup**: it does not represent multi-session
  conversations or realistic agentic tasks.
- **Protocol oriented to affect-aware retrieval**: it measures recall@k per
  affective quadrant, not downstream answer quality.
- **Explicit affect for affect-aware adapters**: AFT receives the query's
  affective context via `valence/arousal`; generalist baselines can ignore it
  entirely.
- **Non-uniform latencies**: the numbers mix local stores, optional
  dependencies, and systems designed for different purposes.

Accordingly, current comparative results should be read as **early controlled
evidence**, not as proof of general superiority over production-grade memory
systems.

### 2.3 Public datasets still small

The repository now ships two public dataset families:

- `affect_reference_v1.jsonl`: synthetic benchmark targeted at mood-congruent
  retrieval
- `realistic_recall_v1.json`: multi-session replay benchmark, still small
  and scripted

This improves reproducibility relative to earlier versions. The realistic
benchmark now rejects trivial queries (`candidate_count <= top_k`) and uses
`top1_accuracy` as the primary metric rather than relying on `hit@k` alone.
Even so, it is not yet a standardized, large, genuinely multi-turn dataset
capable of supporting strong between-system comparisons on emotional memory.

Comparisons with `Mem0` and `LangMem` are no longer hypothetical: the
repository ships adapters and reproducible results on the controlled
benchmark. These comparisons, however, remain limited by the synthetic
protocol, by the different functional surfaces of the systems, and by the
absence of human eval or realistic downstream tasks. `Letta` remains
unevaluated without access to the cloud service / API key.

The multi-session replay benchmark (v1.4) has been expanded to 50 scenarios /
100 queries with a rebalanced challenge-type distribution. AFT separates
clearly from a pure `recency` baseline and leads `naive_cosine` on aggregate
top1 (0.70 vs 0.50, N = 100, `sbert-bge`). On the `semantic_confound` subset
(N = 30), AFT top1 = 0.73 vs naive 0.47, Δ = +0.27 [0.10, 0.43], p_adj =
0.006 — the first per-challenge result to survive Holm correction. Other
per-subset results remain below the significance threshold. The earlier hash
embedder regression (AFT 0.12 vs naive 0.25) is confirmed as an embedder
artefact in `benchmarks/realistic/challenge_subset_pairwise.json`. The
benchmark still does not support strong claims of general superiority over
semantic-only retrieval in fully naturalistic multi-turn scenarios.

### 2.4 Oracle-affect circularity in Hd\* studies

The primary architecture comparison studies (Hd1, Hd2, Hd2_IT, Hd2_ES) compare
`aft_noAppraisal` — AFT using **preset valence/arousal values from the benchmark
dataset** — against `naive_cosine` which has no access to any affective signal.

This means the Hd\* numbers do **not** test whether AFT with automatic appraisal
beats naive cosine. They test: "given a perfect oracle of affect, does the AFT
retrieval architecture (mood field, momentum, resonance, decay) utilize that oracle
better than a pure cosine baseline?"

The dataset's affect values were hand-crafted by the author with AFT theory in mind
(valence/arousal chosen to discriminate the target memory). This introduces two
distinct concerns:

- **Operationalization gap**: real-world AFT deployments use `LLMAppraisalEngine`
  or `KeywordAppraisalEngine`, not preset oracle values. The Ha2/Hb2 results show
  that even keyword appraisal collapses performance (Δ = −0.39). The architecture
  advantage under automatic appraisal is now established as **negative** — see the
  Addendum G / P results below.
- **Dataset circularity**: affect labels were designed to favor affective
  discrimination. Scenarios where valence is not discriminative would reduce the
  oracle advantage. The proportion of AFT-favorable vs. neutral scenarios has not
  been audited.

**Addendum G (executed, FAIL).** The architecture comparison was run with
`LLMAppraisalEngine` on an affect-free dataset (`realistic_recall_v3_noAF`, 50
scenarios, 200 queries, sbert-bge, gpt-5-mini): dual-path AFT did **not** beat naive
cosine (top1 0.315 vs 0.325, Δ=−0.010, p_one=0.367).

**Addendum O + P (recalibration, then re-run — still FAIL).** WP-1a diagnosed the
appraisal signal as _mis-calibrated_, not absent (valence Pearson r=0.81). Addendum O
numerically recalibrated the Scherer SEC→valence/arousal projection (model M1, valence
bias +0.200→+0.072 on held-out scenarios — a calibration PASS). Addendum P then re-ran
Hg1 with M1 on a leakage-free affect-free dataset disjoint from the calibration data
(`realistic_recall_v4_noAF`, 40 scenarios, 160 queries, frozen before the run): naive
cosine was _significantly_ ahead — top1 0.800 vs 0.887 (Δ=−0.0875 [−0.144, −0.031],
p*one=0.0018, d=−0.242). Exploratory contrasts show the signal is real but not enough:
LLM affect beats fixed-neutral (Hp2, +0.056, p=0.030) and the deferred dual-path
schedule is essential (Hp3, +0.512, d=0.95; synchronous appraisal collapses to 0.287).
A better-calibrated affect signal is a \_net distractor* on affect-free queries where
semantics alone is already highly discriminative.

**Conclusion.** Hd\* results should be read as "architecture potential under oracle
affect", not "architecture advantage in production". The oracle-affect scope is a hard
boundary; calibration quality does not, by itself, convert into an affect-free retrieval
gain. See `benchmarks/preregistration_addendum_p_hg1_rerun_closure.md`.

**State-injection boundary (Addendum Q, 2026-06-11).** The oracle-affect boundary is
sharper than calibration: in the Hd-family protocols each query carries a `state` field
injected into the engine _before_ retrieval, i.e. the benchmark performs the
query↔state alignment. Addendum Q tested whether the session trajectory supplies that
alignment for free on affect-discriminative queries (LLM-inferred affect, gated
front-router, `realistic_recall_v5_gate`): it does not — the affect channel loses to
cosine even there (tiebreak 0.160 vs 0.280, Hq1 FAIL), and even a perfect gate stays
below cosine (Hq4, Δ=−0.045, p=0.0024). Gating only neutralizes the always-on penalty
(Hq2 PASS, +0.080; gated == cosine on affect-free queries). Because retrieval signals
are state-based and the query is never appraised, query-driven affect questions are
architecturally out of reach for the current signal set; the affect-routing line is
closed. See `benchmarks/preregistration_addendum_q_affect_gating_closure.md`.

### 2.5 Resonance magnitude amplification on e5-small-v2

S3 ablation (`no_resonance` variant, e5-small-v2) found that removing the
`ResonanceLink` layer **improves** top1 accuracy: Δ = +0.075 [0.040, 0.115],
p_boot=0.0002, p_adj=0.001 (Holm-corrected, n_bootstrap=10000). Per-challenge
decomposition (Addendum I, `benchmarks/preregistration_addendum_i.md`) confirms
this is **magnitude amplification, not sign reversal**: Δ≥0 on all five challenge
types, concentrated in `semantic_confound` (e5 Δ=+0.125 vs SBERT Δ=+0.025).

The SBERT embedder shows Δ = +0.030 (raw p_boot=0.022, family-corrected NS
p_holm=0.109) for the same ablation — a non-significant result under Holm
correction. The effect is thus **embedder-dependent**: statistically significant
on e5 but not on SBERT.

**Hi3 confirmatory closure (2026-05-06, N=500, seed=1, Holm m=3):** the
cross-embedder amplification on `semantic_confound` is confirmed at the
pre-registered threshold. Primary Hi3 **PASS** (Δ=+0.090 [0.030, 0.160],
d=0.257, Holm-adj p=0.0234). Secondary Hi3_recency PASS (Δ=+0.070, p_adj=0.0234)
extends the amplification to `recency_confound`; Hi3_arc FAIL (Δ=+0.010,
p_adj=0.380) scopes it: the embedder gap does not extend to `affective_arc`
queries. The v2 finding is not a sample-size artefact. Closure document:
`benchmarks/preregistration_addendum_i_closure.md`.

Mechanism remains exploratory: link-count instrumentation shows both embedders
saturate the top-5 link cap (mean ≈ 5.0), so link density alone does not
differentiate the embedders. The Hi2 spreading-activation over-fire hypothesis
is not yet localised at link-density granularity; link-type and link-strength
distributions remain unmeasured.

Potential explanations:

1. **Geometry incompatibility**: e5-small-v2's distance space clusters semantically
   related items more aggressively than SBERT. Spreading activation over this geometry
   may amplify noise rather than signal, causing resonance links to hurt rather than help.
2. **Resonance as amplifier**: if the spreading-activation BFS traverses links that
   connect semantically similar but contextually incorrect memories, it boosts
   distractors in e5's tighter cluster geometry.
3. **Benchmark-specific**: `realistic_recall_v2` contains challenge types (semantic
   confound, same-topic distractor) explicitly designed to create confusable neighbors.
   Resonance may be counterproductive when the primary challenge is distractor
   rejection rather than recall.

This finding does not refute resonance's theoretical role, but it indicates that
the interaction between the spreading-activation mechanism and the embedder's
distance geometry is non-trivial and embedder-dependent. It is disclosed in
`claim_validation_matrix.json` under the dedicated claims `resonance_amplification_e5`
(Hi3 PASS), `Hi3_recency` (PASS), and `Hi3_arc` (FAIL).

### 2.6 Controlled benchmark scope and confirmation bias

The `realistic_recall_v2` benchmark (50 scenarios, 200 queries, 5 challenge types × 40)
was designed to evaluate AFT on scenarios where affective signals are relevant. Four of
the five challenge types are **by construction pro-AFT**:

- `affective_arc` — valence discriminates the emotionally salient target from neutral alternatives
- `momentum_alignment` — emotional momentum is the primary differentiating signal
- `semantic_confound` — surface cosine fails specifically where affect differs
- `same_topic_distractor` — same-topic distractors are partly rejected via affect

Only `recency_confound` is valence-neutral (the discrimination criterion is temporal, not
affective). The aggregate advantage (Δ=+0.205 SBERT, Δ=+0.155 e5) should therefore be
read as **"AFT advantage when affective context is discriminative"**, not as general
superiority across all retrieval scenarios.

This is complementary to the LoCoMo negative result (Gate 1 FAIL, F1=0.168 vs 0.271):
on factual open-domain QA where affect is not discriminative, AFT provides no advantage.
The two results are consistent — AFT's benefit is scope-dependent. Scenarios where
valence and arousal do not distinguish the target from distractors are under-represented
in v2; their effect on aggregate top1 is unknown.

### 2.7 Human-eval pipeline not yet run with real ratings

The repository now ships an executable pipeline in `benchmarks/human_eval/`
that generates packets, rating templates, and aggregated summaries, with a
pilot v1 locked on `aft` vs `naive_cosine` across 10 scenarios. This closes
the procedural gap but not the empirical one:

- **No real raters included in the repository**: the pilot has not yet been
  run with human participants.
- **Empty templates do not count as evidence**: the pipeline now rejects
  `ratings.jsonl` files left in placeholder state.
- **No checked-in summary counts as a result**: `summary.json` and
  `summary.md` are not part of the evidence surface until the pilot is run
  with completed ratings.
- **Inter-rater agreement is wired but not yet measured**: Krippendorff's
  alpha (ordinal) is computed automatically when `ratings.jsonl` contains
  ratings from at least 2 raters. The target is alpha ≥ 0.67 with 3+
  raters; no real ratings have been collected yet.

### 2.8 LoCoMo per-task weight tuning closed (Add. J Hj1 FAIL)

Following the LoCoMo Gate-1 negative result (G2 FAIL, F1=0.168 vs naive_rag=0.271
on the full benchmark), Addendum J (`benchmarks/preregistration_addendum_j.md`)
pre-registered a per-category Pareto sweep over 10 weight configurations × a
200-QA stratified subsample (seed=42) to test whether per-task `base_weights`
tuning could close the gap. **Hj1 FAIL** (2026-05-06): no AFT configuration
reaches `naive_rag` parity on any of the four LoCoMo categories
(`single_hop`, `multi_hop`, `temporal`, `open_domain`).

The best configuration (W2: `sem=0.50, mood=0.30, no resonance`) improves
over the AFT default by Δ_F1=+0.044 (0.1765 vs 0.1323) and over W0 by
Δ_judge_acc=+0.085, but still trails `naive_rag` (F1=0.2092) by Δ=−0.033.
Aggregating across 10 sweeps and 4 categories produced no Pareto frontier
where AFT dominates the baseline, even on categories where affect could
plausibly help (`temporal` arc-like queries).

This forecloses the per-task weight-tuning hypothesis as a path to closing G2.
The remaining options are architectural rather than parametric:

- **Per-category routing**: classify queries by type and switch between AFT and
  naive cosine retrieval based on whether affect is discriminative.
- **Query-type classifier**: detect "factual open-domain" vs "affective arc"
  queries upstream of the 6-signal scorer and downweight non-discriminative
  signals dynamically.
- **Hybrid retrieval**: combine naive top-k with AFT re-ranking only when the
  affect signal exceeds a confidence threshold.

These are larger architectural changes outside v0.10 scope. Full-N (all
1500 LoCoMo queries) replication of W2 was _not_ warranted given the 200-QA
result already closes Hj1. See `benchmarks/locomo/pareto_results.md` and
`benchmarks/preregistration_addendum_j_closure.md` for the full numerical record.

### 2.9 Cross-seed robustness: characterized (retrieval is near-deterministic)

The confirmatory studies pin a single random seed each (ablation `seed=0`, Hi3
`seed=1`, the Pareto sweep `seed=42`), and the confidence intervals reported
throughout are **bootstrap CIs resampled within a single run**, not variance
_across_ seeds. To check whether that single-seed convention hides cross-run
instability, `benchmarks/realistic/multiseed_runner.py` (`make bench-multiseed`)
re-runs the realistic replay benchmark across seeds `{0, 1, 7, 42, 123}`, each in
an **isolated subprocess** invoking the canonical runner, and reports the
cross-seed mean/stdev/min/max of `top1_accuracy` and of the AFT−baseline Δ.

Result (`benchmarks/realistic/multiseed_results.md`, `realistic_recall_v2`, hash
embedder): retrieval is **near-deterministic — not exactly deterministic**. The
RNG seed itself moves nothing; the genuine residual stochasticity lives in (a)
dataset _generation_ seeds (frozen and committed) and (b) bootstrap RNG (already
reflected in the reported CIs). But the per-query top-1 outcome is **not bit-stable
across fresh sweeps**: repeating `make bench-multiseed` six times gave
`retrieval_deterministic=True` in 5/6 runs and `False` in 1/6 (cross-seed stdev
0.0024), and the absolute `aft` mean drifted across sweeps (0.120–0.125).

> An earlier version of this section asserted "cross-seed stdev and spread are
> exactly 0.0000 — identical across all five seeds." A fresh validation pass
> falsified that headline; the corrected scope is **near-deterministic, with sub-CI,
> timing-driven variance on near-ties**.

The cause is a timing effect, not RNG. The engine stamps encode/retrieve with real
wall-clock time (`datetime.now(tz=UTC)`, see `engine.py`), and ACT-R decay is a
function of `now − encoded_at`. Because each seed's subprocess is launched at a
slightly different instant, a ranking that is already at a numerical tie can tip
between seeds **even with subprocess isolation** — the isolation removes RNG
coupling but not wall-clock timing. (Running several benchmarks back-to-back inside
a single process amplifies the same effect.) This is _correct production behaviour_
(decay should track real time), not a library defect, and the resulting variance
stays within the reported bootstrap CIs — AFT still clears `naive_cosine` by far
more than the spread (Δ ≈ +0.075). A fully time-deterministic benchmark would
require threading an injected clock through `encode`/`retrieve`; that is
deliberately not done here, as it would expand the core API and disturb the 127
fidelity benchmarks for a sub-CI, benchmark-only effect. Finally, the sweep would
surface larger variance for a _stochastic_ embedder or a regenerated dataset;
extending it to those cases is straightforward future work.

---

## 3. Architectural Limits

### 3.1 Appraisal depends on an external LLM

`LLMAppraisalEngine` requires a call to an external LLM (OpenAI-compatible)
to produce an `AppraisalVector`. This introduces:

- **Latency**: 200–2000 ms per encoding in slow-path mode.
- **Cost**: dependent on provider and model.
- **Non-determinism**: the same text can produce different appraisals across
  successive calls.
- **Network dependency**: encoding does not work offline without the
  `KeywordAppraisalEngine` fallback.

The `KeywordAppraisalEngine` is a rule-based fallback with limited coverage
(~50 keyword classes). It does not generalize to open domains.

### 3.2 Affective state: local persistence yes, distributed sharing still limited

`AffectiveState` is no longer only in-process. The repository now supports
dedicated `AffectiveStateStore` implementations, with local backends such as
`InMemoryAffectiveStateStore` and `SQLiteAffectiveStateStore`, plus an
optional `RedisAffectiveStateStore` backend for shared state.

Important architectural limits remain:

- **No production-grade validation of the shared backend**: Redis is
  available as an optional backend, but comprehensive validation on real
  multi-worker deployments is still missing.
- **No joint memory+state transaction**: `MemoryStore` and
  `AffectiveStateStore` remain separated by design; there is no atomic
  cross-store semantics yet.
- **Limited distributed synchronization**: the Redis backend covers
  affective-state persistence, not distributed vector memory.

### 3.3 Thread-safety limited to a single connection

`SQLiteStore` uses a single `sqlite3.Connection` object with `threading.RLock`.
This serializes accesses but does not allow reader parallelism. For high
throughput on a SQLite store, a connection-pool architecture would be more
scalable.

### 3.4 No "enterprise" vector store

The available adapters are `InMemoryStore` (RAM, non-persistent),
`SQLiteStore` (local file, scalable to ~10^6 memories), `QdrantStore`
(v0.9), and `ChromaStore` (v0.9). There are no adapters for Weaviate or
Pinecone. The `MemoryStore` Protocol is duck-typed and contributions are
welcome.

---

## 4. Theoretical Limits

### 4.1 Partial operationalization of Heidegger

`MoodField` operationalizes _Stimmung_ (background emotional tonality) as an
EMA over valence–arousal with decay parameters. This is a necessary
computational simplification of a phenomenological concept that in Heidegger
is pre-cognitive and structurally tied to being-in-the-world. The mapping is
_inspired_, not _faithful_.

### 4.2 AFT novelty credentials are not peer-reviewed

As of the v0.9 release, Affective Field Theory has not been formally
published in a peer-reviewed venue. Originality claims (in particular the
unified integration of Russell + Scherer + Heidegger + Hebb + Collins &
Loftus in a single computational model for LLM memory) are plausible but
have not yet been validated by the scientific community.

---

## 5. Future Work

### Recently shipped (v0.9)

| Feature                                           | Status          |
| ------------------------------------------------- | --------------- |
| Qdrant vector-database adapter (`[qdrant]` extra) | ✅ shipped v0.9 |
| ChromaDB adapter (`[chroma]` extra)               | ✅ shipped v0.9 |
| OpenTelemetry spans (`[otel]` extra)              | ✅ shipped v0.9 |

### Open limits (not tied to a specific release)

| Limit                                                 | Indicative horizon |
| ----------------------------------------------------- | ------------------ |
| Broader, comparative affect-aware realistic benchmark | ongoing research   |
| Execute the human-eval pilot with real ratings        | research track     |
| BYO appraisal schema (OCC, GRID, custom taxonomies)   | v0.10              |

---

_Document added in v0.5.1. Last updated: v0.9 (2026-05-06)._
