# Known Limitations of Affective Field Theory

> This section documents the acknowledged limits of the current implementation.
> Transparency about limits is integral to any claim of scientific rigor.

---

## 1. Limits of the Affective Model

### 1.1 Affect dimensionality

The model uses Russell's 2-dimensional circumplex (valence–arousal). This is the
most widely cited and computationally tractable choice, but it has documented
limitations:

- **Does not capture dominance (full PAD)**: the PAD model of Mehrabian &
  Russell (1974) includes a third *dominance* dimension (control/submission).
  In the current version, `CoreAffect` only models valence and arousal;
  dominance is derived optionally via `AppraisalVector.dominance`. It is not
  yet a primary dimension of the field.
- **Contested circumplex structure**: Russell (2003) himself revised the
  theory toward "core affect" as a contextual construction. The current model
  uses the classic 1980 form, not the constructionist revisions.
- **Limited emotional granularity**: the Plutchik mapping to 8 primary emotions
  is a discretization of the continuous space. Complex emotions (nostalgia,
  Schadenfreude, awe) have no direct representation.

### 1.2 Language dependence

- **Optimized for English and Italian**: the `KeywordAppraisalEngine` uses
  English-language rules. The `LLMAppraisalEngine` depends on the underlying
  LLM — it works best with languages well-represented in the pretraining
  distribution.
- **No formal cross-lingual validation**: no benchmarks exist that test the
  psychological coherence of affective predictions across languages other
  than English.

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

### 2.4 Human-eval pipeline not yet run with real ratings

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

The available adapters are `InMemoryStore` (RAM, non-persistent) and
`SQLiteStore` (local file, scalable to ~10^6 memories). There are no
adapters for distributed systems such as Qdrant, Chroma, Weaviate, or
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
| Qdrant / Chroma adapters | architecture track |
| Execute the human-eval pilot with real ratings | research track |
| Dominance as a primary dimension | research track |
| Distributed / enterprise memory store (Qdrant, Chroma, ...) | product track |

---

*Document added in v0.5.1. Update at every significant release.*
