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

- **Validated on English, Italian, and Spanish**: the realistic-recall benchmark
  has been run on EN (v2, SBERT/e5), IT (v2_it, me5), and ES (v2_es, sbert/me5)
  slices. Hd2 hypothesis passes on EN and IT; ES shows a split: SBERT PASS
  (Δ=0.138, p=0.045) but me5 borderline FAIL (Δ=0.113, p=0.110).
- **`KeywordAppraisalEngine` is English-only**: the `LLMAppraisalEngine` handles
  multilingual input, but appraisal quality varies by language.
- **No formal cross-lingual validation beyond EN/IT/ES**: generalisation to
  other languages is not established. Larger multilingual embedders
  (e.g., `BGE-M3`) are a natural next step.

---

## 2. Limits of Empirical Validation

### 2.1 No validation with human users

The 126 psychological fidelity tests validate that the system behaves
*coherently with the theories* it implements (for example, that retrieval is
mood-congruent and that decay follows a power law). They do not validate that
the system's behavior corresponds to how *real human beings* form and retrieve
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

### 2.4 Oracle-affect circularity in Hd* studies

The primary architecture comparison studies (Hd1, Hd2, Hd2_IT, Hd2_ES) compare
`aft_noAppraisal` — AFT using **preset valence/arousal values from the benchmark
dataset** — against `naive_cosine` which has no access to any affective signal.

This means the Hd* numbers do **not** test whether AFT with automatic appraisal
beats naive cosine. They test: "given a perfect oracle of affect, does the AFT
retrieval architecture (mood field, momentum, resonance, decay) utilize that oracle
better than a pure cosine baseline?"

The dataset's affect values were hand-crafted by the author with AFT theory in mind
(valence/arousal chosen to discriminate the target memory). This introduces two
distinct concerns:

- **Operationalization gap**: real-world AFT deployments use `LLMAppraisalEngine`
  or `KeywordAppraisalEngine`, not preset oracle values. The Ha2/Hb2 results show
  that even keyword appraisal collapses performance (Δ = −0.39). The architecture
  advantage under automatic appraisal is not yet established.
- **Dataset circularity**: affect labels were designed to favor affective
  discrimination. Scenarios where valence is not discriminative would reduce the
  oracle advantage. The proportion of AFT-favorable vs. neutral scenarios has not
  been audited.

Addendum G (future study) will address this by running the architecture comparison
with `LLMAppraisalEngine` on a dataset designed without preset affect injection.
Until Addendum G is executed, Hd* results should be read as "architecture potential
under oracle affect", not "architecture advantage in production".

### 2.5 Resonance sign reversal on e5-small-v2

S3 ablation (`no_resonance` variant, e5-small-v2) found that removing the
`ResonanceLink` layer **improves** top1 accuracy: Δ = +0.085 [0.04, 0.13],
p_boot < 0.001, p_adj < 0.001 (Holm-corrected). This is the opposite of the
theoretical prediction (resonance should boost relevant memories).

The SBERT embedder shows Δ = +0.02 (NS, p=0.20) for the same ablation — a
directionally opposite but non-significant result. The sign reversal is
**statistically confirmed on e5 but not replicated on SBERT**.

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
`claim_validation_matrix.json` under `theory_faithful_operationalization.not_yet_shown`.

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

`MoodField` operationalizes *Stimmung* (background emotional tonality) as an
EMA over valence–arousal with decay parameters. This is a necessary
computational simplification of a phenomenological concept that in Heidegger
is pre-cognitive and structurally tied to being-in-the-world. The mapping is
*inspired*, not *faithful*.

### 4.2 AFT novelty credentials are not peer-reviewed

As of the v0.5.x release, Affective Field Theory has not been formally
published in a peer-reviewed venue. Originality claims (in particular the
unified integration of Russell + Scherer + Heidegger + Hebb + Collins &
Loftus in a single computational model for LLM memory) are plausible but
have not yet been validated by the scientific community.

---

## 5. Future Work

The following limits are explicitly planned but are not tied to a specific
release:

| Limit | Indicative horizon |
|---|---|
| Broader, comparative affect-aware realistic benchmark | post-0.6 |
| Qdrant / Chroma adapters | ✅ shipped v0.9 |
| Execute the human-eval pilot with real ratings | research track |
| Dominance as a primary dimension | research track |
| Distributed / enterprise memory store (Qdrant, Chroma, ...) | ✅ shipped v0.9 |

---

*Document added in v0.5.1. Update at every significant release.*
