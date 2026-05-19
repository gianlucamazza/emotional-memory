# emotional_memory

[![CI](https://github.com/gianlucamazza/emotional-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/gianlucamazza/emotional-memory/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/gianlucamazza/emotional-memory/graph/badge.svg)](https://codecov.io/gh/gianlucamazza/emotional-memory)
[![PyPI](https://img.shields.io/pypi/v/emotional_memory)](https://pypi.org/project/emotional_memory/)
[![Python](https://img.shields.io/pypi/pyversions/emotional_memory)](https://pypi.org/project/emotional_memory/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19972258.svg)](https://doi.org/10.5281/zenodo.19972258)
[![Benchmarks](https://img.shields.io/badge/benchmarks-tracked-blue)](https://gianlucamazza.github.io/emotional-memory/dev/bench/)
[![SLSA 3](https://slsa.dev/images/gh-badge-level3.svg)](https://github.com/gianlucamazza/emotional-memory/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Emotional memory for LLMs based on **Affective Field Theory (AFT)** — a 5-layer model that encodes not just *what* happened, but *how it felt*, *how that feeling was moving*, and *what mood colored the moment*. Validated in English (N=200, SBERT), Italian and Spanish (N=80 Hd2; N=120 power top-up FAIL at me5), and French (N=120 me5, Δ=+0.18, p&lt;0.0001 — Addendum M PASS) on the realistic-recall benchmark.

<!-- ssot:positioning-start -->
## Why emotional_memory?

Most LLM memory libraries treat retrieval as semantic-only: vector similarity over text. Real human recall is also driven by **affective congruence** — we remember things that feel like how we feel now (Bower 1981), and emotionally-charged events consolidate more strongly than neutral ones (Cahill & McGaugh 1995).

`emotional_memory` adds an explicit affective layer on top of standard semantic retrieval, grounded in 40+ years of affective-science literature and validated against 20 published psychological phenomena (126 fidelity tests).

### How it compares

| Library | Memory model | Affective retrieval | Reconsolidation | Decay model | Psychological fidelity tests |
|---|---|---|---|---|---|
| **emotional_memory** | 5-layer AFT (semantic + valence/arousal + momentum + mood + appraisal) | ✅ mood-congruent + APE-gated | ✅ Nader & Schiller 2000 | ACT-R power-law + arousal modulation | 126 tests, 20 phenomena |
| MemGPT / Letta | Hierarchical context (working + archival) | ❌ | ❌ | None | — |
| mem0 | Fact extraction + vector store | ❌ | ❌ | None | — |
| A-MEM | Atomic notes + dynamic links | ❌ | ❌ | None | — |
| LangMem | Hot/cold memory tiers | ❌ | ❌ | Time-based eviction | — |
| Generative Agents (Park et al.) | Importance + recency + relevance | Partial (importance only) | ❌ | Exponential | — |

This is **not** a replacement for those tools — `emotional_memory` is a focused primitive that can plug into any of them via the `MemoryStore` protocol (LangChain adapter included).

### 30-second example

```python
from emotional_memory import EmotionalMemory, InMemoryStore, CoreAffect

em = EmotionalMemory(store=InMemoryStore(), embedder=MyEmbedder())

em.set_affect(CoreAffect(valence=-0.6, arousal=0.7))   # stressed
em.encode("The deployment failed at 3am.")
em.encode("Beautiful sunset on the lake.")

em.set_affect(CoreAffect(valence=-0.5, arousal=0.6))   # similar mood later
results = em.retrieve("yesterday", top_k=2)
# → deployment memory ranks higher than sunset, even with equal semantic distance.
```
<!-- ssot:positioning-end -->

## Installation

```bash
uv pip install emotional-memory
uv pip install "emotional-memory[sentence-transformers]"  # real semantic embeddings (recommended)
uv pip install "emotional-memory[sqlite]"                 # SQLite persistence via sqlite-vec
uv pip install "emotional-memory[qdrant]"                 # Qdrant vector database
uv pip install "emotional-memory[chroma]"                 # ChromaDB vector database
uv pip install "emotional-memory[otel]"                   # OpenTelemetry tracing (no-op without this extra)
uv pip install "emotional-memory[redis]"                  # shared affective-state persistence via Redis
uv pip install "emotional-memory[viz]"                    # matplotlib visualization
uv pip install "emotional-memory[dotenv]"                 # .env file loading via python-dotenv
```

For development:

```bash
git clone https://github.com/gianlucamazza/emotional-memory
cd emotional-memory
make install
# optional local demo stack:
make install-demo
```

## Quickstart

```bash
uv pip install "emotional-memory[sentence-transformers]"
```

```python
from emotional_memory import EmotionalMemory, InMemoryStore, CoreAffect
from emotional_memory.embedders import SentenceTransformerEmbedder

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=SentenceTransformerEmbedder(),  # all-MiniLM-L6-v2 by default
)

# Set current emotional state
em.set_affect(CoreAffect(valence=0.8, arousal=0.6))

# Encode memories — each one captures the full affective context
em.encode("Just shipped the feature after three hard weeks.")
em.encode("Team celebration in the office.", metadata={"source": "slack"})

# Retrieve — ranked by semantic relevance AND emotional congruence
results = em.retrieve("difficult project success", top_k=3)
for mem in results:
    print(mem.content, mem.tag.core_affect)

# Or inspect why a memory ranked where it did
explained = em.retrieve_with_explanations("difficult project success", top_k=1)
top = explained[0]
print(top.score)
print(top.breakdown.raw_signals)
```

**Bring your own embedder** — any object with `.embed(text) -> list[float]` works:

```python
class MyEmbedder:
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

Or subclass `SequentialEmbedder` and implement only `embed()` — `embed_batch()` is provided.

### Async

```python
import asyncio
from emotional_memory import EmotionalMemory, InMemoryStore, as_async
from emotional_memory.embedders import SentenceTransformerEmbedder

sync_em = EmotionalMemory(store=InMemoryStore(), embedder=SentenceTransformerEmbedder())
em = as_async(sync_em)  # wraps sync components with asyncio.to_thread bridges

async def main():
    await em.encode("Meeting went surprisingly well today.")
    results = await em.retrieve("work meeting", top_k=3)

asyncio.run(main())
```

For native async embedders or stores, construct `AsyncEmotionalMemory` directly with
`SyncToAsyncEmbedder`, `SyncToAsyncStore`, or your own `AsyncEmbedder`/`AsyncMemoryStore`.

## Affective Field Theory

AFT models emotion as a **field** — distributed, dynamic, multi-layer — rather than a discrete label or a single coordinate. Five layers are captured at encoding time:

| Layer | Model | What it captures |
|---|---|---|
| **CoreAffect** | Russell-Mehrabian PAD model | Continuous `(valence, arousal, dominance)` — the emotional substrate |
| **AffectiveMomentum** | Spinoza — affect as transition | Velocity and acceleration of affect change |
| **MoodField** | Heidegger — *Stimmung* as attunement | Slow-moving global mood with inertia (EMA) |
| **AppraisalVector** | Scherer/Lazarus/Stoics | Emotion derived from evaluation: novelty, goal-relevance, coping, norm-congruence, self-relevance |
| **ResonanceLinks** | Aristotle/Hume/Bower/Collins & Loftus/Hebb | Associative bidirectional graph: semantic, emotional, temporal, causal, contrastive links; multi-hop spreading activation + Hebbian co-retrieval strengthening |

Full theoretical foundations: [`docs/research/`](docs/research/index.md)

## API Overview

### `EmotionalMemory`

```python
em = EmotionalMemory(
    store: MemoryStore,
    embedder: Embedder,
    appraisal_engine: AppraisalEngine | None = None,  # optional: auto-appraise via LLM
    config: EmotionalMemoryConfig | None = None,
    state_store: AffectiveStateStore | None = None,   # optional: persist affective state
)
```

| Method | Description |
|---|---|
| `encode(content, appraisal=None, metadata=None) -> Memory` | Encode content with full AFT pipeline |
| `observe(content, appraisal=None, metadata=None) -> EmotionalTag` | Update affective state without storing a retrievable memory |
| `encode_batch(contents, metadata=None) -> list[Memory]` | Batch encode with `embed_batch()`, per-item appraisal |
| `retrieve(query, top_k=5) -> list[Memory]` | Emotionally-weighted retrieval + reconsolidation |
| `retrieve_with_explanations(query, top_k=5) -> list[RetrievalExplanation]` | Same retrieval pipeline plus a structured score breakdown |
| `elaborate(memory_id) -> Memory \| None` | Run full appraisal on a fast-path (`pending_appraisal=True`) memory and blend core_affect |
| `elaborate_pending() -> list[Memory]` | Elaborate all pending fast-path memories in one call |
| `delete(memory_id)` | Remove a memory from the store |
| `get(memory_id) -> Memory \| None` | Look up a single memory by ID |
| `list_all() -> list[Memory]` | Return all stored memories |
| `len(engine) -> int` | Number of memories in the store |
| `prune(threshold=0.05) -> int` | Delete memories below effective strength threshold; returns count removed |
| `export_memories() -> list[dict]` | Serialise all memories to JSON-safe dicts (backup / migration) |
| `import_memories(data, overwrite=False) -> int` | Restore from `export_memories()` output; returns count written |
| `get_state() -> AffectiveState` | Current affective state (read-only copy) |
| `set_affect(core_affect)` | Manually inject a CoreAffect |
| `reset_state()` | Reset runtime affective state to the initial baseline |
| `save_state() -> dict` | Serialise affective state for persistence |
| `load_state(data)` | Restore previously saved affective state |
| `persist_state() -> dict` | Persist the current affective state to the configured state store |
| `restore_persisted_state() -> bool` | Restore the last persisted affective state from the configured state store |
| `clear_persisted_state()` | Clear the configured persisted affective-state snapshot |
| `get_current_mood(now=None) -> MoodField` | Read-only mood with time regression |
| `close()` | Release store resources (e.g. SQLite connection); also via `with` |

Both engines support context managers for automatic resource cleanup:

```python
with EmotionalMemory(store=SQLiteStore("mem.db"), embedder=MyEmbedder()) as em:
    em.encode("Session start")
    results = em.retrieve("relevant context")
# SQLiteStore.close() called automatically
```

Use `retrieve()` for normal recall paths. Use `retrieve_with_explanations()` when
you need the ranking-time decomposition (`semantic`, `mood`, `affect`, `momentum`,
`recency`, `resonance`) for debugging, evaluation, or UI inspection.

### `AsyncEmotionalMemory`

Same method signatures as `EmotionalMemory`. Coroutines: `encode`, `observe`, `encode_batch`, `retrieve`,
`retrieve_with_explanations`, `elaborate`, `elaborate_pending`, `delete`, `get`, `list_all`,
`count`, `prune`, `export_memories`, `import_memories`, `persist_state`, `restore_persisted_state`,
`clear_persisted_state`, `close`. State accessors (`get_state`, `set_affect`, `reset_state`,
`save_state`, `load_state`, `get_current_mood`) remain synchronous.

Supports `async with` for automatic resource cleanup:

```python
async with AsyncEmotionalMemory(store=..., embedder=...) as em:
    await em.encode("Session start")
    results = await em.retrieve("relevant context")
```

```python
from emotional_memory import AsyncEmotionalMemory, SyncToAsyncEmbedder, SyncToAsyncStore
```

Bridge adapters: `SyncToAsyncEmbedder`, `SyncToAsyncStore`, `SyncToAsyncAppraisalEngine` wrap
any sync implementation. `SyncToAsyncStore` also proxies `close()` to the underlying store.
`as_async(engine)` wraps a complete `EmotionalMemory` in one call.

### Key config classes

- `EmotionalMemoryConfig` — top-level config (nested configs below + top-level flags):
  - `dual_path_encoding: bool = False` — LeDoux 1996 fast/slow path (encode first, elaborate later)
  - `elaboration_learning_rate: float = 0.7` — blend ratio when `elaborate()` runs full appraisal (70% appraised / 30% raw)
  - `auto_categorize: bool = False` — run Plutchik categorization on every encode
  - `enable_appraisal: bool = True` — use appraisal engine if configured (ablation flag)
  - `enable_mood_signal: bool = True` — include mood-congruence in retrieval scoring (ablation flag)
  - `enable_momentum: bool = True` — include momentum alignment in retrieval scoring (ablation flag)
  - `enable_resonance: bool = True` — build and use resonance graph (ablation flag)
  - `enable_reconsolidation: bool = True` — APE-gated reconsolidation on retrieve (ablation flag)
- `RetrievalConfig` — weights, APE threshold, reconsolidation learning rate
- `ResonanceConfig` — similarity threshold, max links, semantic/emotional/temporal weights, candidate multiplier, `propagation_hops`, `hebbian_increment`, configurable link-classification thresholds
- `DecayConfig` — power-law decay parameters, arousal modulation, floor values
- `MoodDecayConfig` — time-based mood regression (half-life, inertia scale, baselines)
- `AdaptiveWeightsConfig` — smooth mood-adaptive retrieval weight tuning (sigmoid/Gaussian gates)
- `LLMAppraisalConfig` — LLM appraisal engine settings (system prompt, cache size, fallback behaviour, `appraisal_schema`)
- `QueryClassifierConfig` — query-type routing mode (`heuristic` / `llm`) + per-type weight override table

### Query routing

`EmotionalMemory` supports per-query-type adaptive weights via a pluggable classifier. The built-in
`HeuristicQueryClassifier` uses keyword patterns to detect temporal, multi-hop, single-hop and
open-domain questions and selects a matching weight profile from a routing table:

```python
from emotional_memory import (
    EmotionalMemory, EmotionalMemoryConfig,
    HeuristicQueryClassifier, LOCOMO_ROUTING,
    InMemoryStore,
)
from emotional_memory.retrieval import QueryClassifierConfig, RetrievalConfig

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=my_embedder,
    config=EmotionalMemoryConfig(
        retrieval=RetrievalConfig(
            query_classifier=QueryClassifierConfig(
                mode="heuristic",
                routed_weights=LOCOMO_ROUTING,
            )
        )
    ),
    query_classifier=HeuristicQueryClassifier(),
)
# retrieve now selects weights based on detected query type
results = em.retrieve("When did Alice first mention the project?")
```

`LOCOMO_ROUTING` is the pre-built routing table derived from the Addendum J Pareto sweep.
For LLM-backed classification use `LLMQueryClassifier` with the same `QueryClassifierConfig`.

### Interfaces (bring your own)

If your embedder has no native batching, subclass `SequentialEmbedder` — `embed_batch()` is
provided automatically:

```python
from emotional_memory import SequentialEmbedder

class MyEmbedder(SequentialEmbedder):
    def embed(self, text: str) -> list[float]:
        return my_model.encode(text).tolist()
```

Otherwise implement the full `Embedder` protocol:

```python
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

class MemoryStore(Protocol):
    def save(self, memory: Memory) -> None: ...
    def get(self, memory_id: str) -> Memory | None: ...
    def update(self, memory: Memory) -> None: ...
    def delete(self, memory_id: str) -> None: ...
    def list_all(self) -> list[Memory]: ...
    def search_by_embedding(self, embedding: list[float], top_k: int) -> list[Memory]: ...
    def __len__(self) -> int: ...

class AffectiveStateStore(Protocol):
    def save(self, state: AffectiveState) -> None: ...
    def load(self) -> AffectiveState | None: ...
    def clear(self) -> None: ...
```

Async variants (`AsyncEmbedder`, `AsyncMemoryStore`, `AsyncAppraisalEngine`) are defined in
`interfaces_async.py`. `AsyncMemoryStore` uses `count() -> int` instead of `__len__` since
dunder methods cannot be coroutines.

**Stores included:**
- `InMemoryStore` — dict-backed, brute-force cosine search (no extra deps)
- `SQLiteStore` — persistent SQLite + sqlite-vec ANN search (`uv pip install "emotional-memory[sqlite]"`)
- `QdrantStore` — Qdrant vector database, embedded or server mode (`uv pip install "emotional-memory[qdrant]"`)
- `ChromaStore` — ChromaDB vector database, ephemeral or persistent (`uv pip install "emotional-memory[chroma]"`)

**Affective-state stores included** (persist the runtime mood/momentum state across sessions):
- `InMemoryAffectiveStateStore` — in-process only, no extra deps
- `SQLiteAffectiveStateStore` — durable across restarts (`[sqlite]` extra)
- `RedisAffectiveStateStore` — shared across processes/services (`[redis]` extra)

Pass one as `state_store=` to `EmotionalMemory(...)` to enable cross-session mood continuity.

### Appraisal Engines

```python
class AppraisalEngine(Protocol):
    def appraise(self, event_text: str, context: dict | None = None) -> AppraisalVector: ...
```

Pass an `appraisal_engine` to `EmotionalMemory` to auto-generate `AppraisalVector` during encode.

**`LLMAppraisalEngine`** — wrap any LLM SDK in a single callable:

```python
from emotional_memory import LLMAppraisalEngine

def my_llm(prompt: str, json_schema: dict) -> str:
    # call openai / anthropic / local model here
    return response_text

engine = LLMAppraisalEngine(llm=my_llm)
em = EmotionalMemory(store=..., embedder=..., appraisal_engine=engine)
```

**`KeywordAppraisalEngine`** — regex-based fallback, zero external dependencies, ships with
default rules covering success, failure, novelty, danger, and social norms:

```python
from emotional_memory import KeywordAppraisalEngine
engine = KeywordAppraisalEngine()  # or pass custom KeywordRule list
```

**BYO appraisal schema (`AppraisalSchema`)** — swap the default Scherer CPM prompt for any
appraisal taxonomy (OCC, GRID, or custom) without forking the library:

```python
from emotional_memory import LLMAppraisalEngine, LLMAppraisalConfig, AppraisalSchema
from emotional_memory.appraisal_schema import AppraisalDimension

my_schema = AppraisalSchema(
    name="occ",
    dimensions=[
        AppraisalDimension(name="desirability", range=(-1.0, 1.0), description="…"),
        AppraisalDimension(name="likelihood",   range=( 0.0, 1.0), description="…"),
    ],
)
config = LLMAppraisalConfig(appraisal_schema=my_schema)
engine = LLMAppraisalEngine(llm=my_llm, config=config)
```

The engine validates `AppraisalVector` outputs against the declared schema dimensions.
`SCHERER_CPM_SCHEMA` (the 5-dimension default) is exported from `emotional_memory` directly.

## Visualization

The optional `viz` extra provides 8 plotting functions for inspecting and presenting the model's internals. Each function accepts an optional `ax` parameter for subplot composition and returns a `matplotlib.Figure`.

```python
from emotional_memory.visualization import plot_circumplex, plot_decay_curves
```

### Valence-Arousal Circumplex

Memories plotted on the Russell-Mehrabian PAD model (valence-arousal plane), colored by consolidation strength.

![Circumplex](docs/images/circumplex.png)

### Decay Curves (ACT-R Power Law)

Family of curves showing how arousal (McGaugh 2004) and retrieval count (spacing effect) modulate memory decay.

![Decay Curves](docs/images/decay_curves.png)

### Yerkes-Dodson Inverted-U

Consolidation strength peaks near effective arousal 0.7, then drops — the classic Yerkes-Dodson curve.

![Yerkes-Dodson](docs/images/yerkes_dodson.png)

### 6-Signal Retrieval Breakdown

Radar chart of the six retrieval signals: semantic similarity, mood congruence, affect proximity, momentum alignment, recency, and resonance boost.

![Retrieval Radar](docs/images/retrieval_radar.png)

### Mood Field Evolution

Time series of valence, arousal, and dominance with dashed baselines showing the regression attractors.

![Mood Evolution](docs/images/mood_evolution.png)

### Adaptive Retrieval Weights

Heatmap showing how retrieval weights shift across different mood states (valence x arousal grid).

![Adaptive Weights](docs/images/adaptive_weights_heatmap.png)

### Resonance Network

Directed graph with memories as nodes and edges colored by link type (semantic, emotional, temporal, causal, contrastive).

![Resonance Network](docs/images/resonance_network.png)

### Appraisal Radar (Scherer CPM)

Spider chart of the 5 Stimulus Evaluation Check dimensions.

![Appraisal Radar](docs/images/appraisal_radar.png)

### Generating images

```bash
make docs-images   # regenerate all PNGs in docs/images/
make research-figures   # regenerate benchmark evidence figures
```

### Evidence Figures

The research figures below are generated from committed benchmark JSON artefacts,
not from rerunning long studies.

![Realistic Replay Overview](docs/images/research/research_realistic_v2_overview.png)

![Realistic Replay Challenge Breakdown](docs/images/research/research_realistic_v2_challenges.png)

![S3 Ablation Study](docs/images/research/research_ablation_s3.png)

![Multilingual Slices (IT + ES + FR)](docs/images/research/research_multilingual.png)

![LoCoMo Negative Result](docs/images/research/research_locomo_negative.png)

## Comparison with Existing Systems

| | **emotional-memory (AFT)** | **Mem0** | **Letta** | **Zep** | **LangChain Memory** |
|---|---|---|---|---|---|
| **License** | MIT | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT |
| **Persistence** | InMemory / SQLite / Qdrant / Chroma | Qdrant, Chroma, Pinecone, PG, MongoDB | PostgreSQL / SQLite | Neo4j (self-hosted) / Cloud | In-memory / custom |
| **BYO embedder** | ✅ any `Embedder` protocol | ✅ (OpenAI default) | ⚠️ partial | ⚠️ partial | ✅ |
| **Emotion model** | ✅ 5-layer AFT (valence, arousal, dominance, mood, appraisal, resonance) | ❌ | ❌ | ❌ | ❌ |
| **Reconsolidation** | ✅ APE-gated lability window | ✅ auto update/remove | ✅ tool-call edit | ✅ edge invalidation | ❌ |
| **Persistent mood state** | ✅ MoodField (Heidegger EMA) | ❌ | ❌ | ❌ | ❌ |
| **LLM-agnostic** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **LangChain integration** | ✅ `EmotionalMemoryChatHistory` | ✅ official | ✅ tools interop | ✅ ZepVectorStore | ✅ native |
| **Internal fidelity tests** | ✅ 126 cases, 20 phenomena ([bench-fidelity](benchmarks/fidelity/)) | — | — | — | — |
| **External benchmark** | ✅ LoCoMo (FAIL: F1 0.168 vs 0.271; Pareto Hj1 FAIL) | ✅ LoCoMo, LongMemEval, BEAM | ✅ LoCoMo, DMR | ✅ DMR, LongMemEval | ❌ |
| **Codebase size** | ~4.8k LOC (src/) | >50k LOC | >50k LOC | >50k LOC | ~5k LOC |

**Key differentiator**: emotional-memory makes affect a first-class, multi-layer part
of encoding and retrieval. Compared with the general-purpose memory systems in this
table, it emphasizes mood-congruent retrieval, appraisal-conditioned tagging, and
APE-gated reconsolidation rather than generic conversational recall alone.

**Benchmark caveat**: AFT fidelity tests validate psychological invariants
(intra-theory). The comparative benchmark in this repo is a controlled synthetic
retrieval probe, not a general downstream evaluation of production memory systems.

## Current validation status

> **Methodological boundary**: results labelled **A)** below inject preset
> valence/arousal values at encode time (oracle affect). The LLM/keyword
> appraisal pipeline is bypassed. Results labelled **B)** ran without
> oracle affect — either naturalistic or appraisal-driven. These two
> regimes measure different things; they should not be conflated.
> The field `requires_oracle_affect` in
> [`docs/research/claim_validation_matrix.json`](docs/research/claim_validation_matrix.json)
> encodes this boundary machine-readably for every claim.

### A) Synthetic affect-controlled benchmarks (oracle affect provided)

- **Theory fidelity — 126 fidelity test cases across 20 phenomena**: the
  implementation behaves coherently with the theories it operationalizes.
  Phenomena include mood-congruent recall (Bower 1981), arousal floor
  (McGaugh 2004), ACT-R power-law decay, Hebbian strengthening, spacing
  effect, spreading activation, and more.
- **Realistic multi-session replay (v2, N=200)**: AFT outperforms `naive_cosine`
  on both embedder classes: SBERT bge-small-en-v1.5 — top1 0.53 vs 0.33,
  Δ=+0.21 [0.15,0.27], p<0.001, d=0.49; e5-small-v2 — top1 0.50 vs 0.34,
  Δ=+0.16 [0.09,0.22], p<0.001, d=0.31. Architecture attribution confirmed
  (appraisal confound ruled out, Gate 3 CLOSED).
- **SOTA comparison (v2, N=200, gpt-4.1-mini)**: AFT top1=0.535 vs Mem0=0.330,
  LangMem=0.365, naive_cosine=0.325. Δ vs cosine: +0.210 [+0.155,+0.270],
  p<0.001, d=0.512; non-overlapping CIs. Neither Mem0 nor LangMem beats cosine
  on this benchmark. **Asymmetry**: Mem0 outperforms AFT on the simpler
  `affect_reference_v1` probe (recall@5=1.00 vs 0.85) at 25× higher latency.
- **Italian multilingual slice (G6, 20 scenarios / 80 queries)**:
  SBERT: AFT hit@k=0.34, naive_cosine=0.19, Δ=+0.15 [p=0.0005].
  me5: AFT hit@k=0.42, naive_cosine=0.26, Δ=+0.16 [p=0.001].
  Spanish-SBERT (N=80 exploratory): Δ=+0.138 [p=0.045]. me5 runs at N=120
  (declared power) FAIL for both languages (Branch C closure 2026-05-07).
- **French multilingual slice (Addendum M, Branch A PASS, 2026-05-16)**:
  30 native-FR hand-authored scenarios, 120 queries, me5, 2-session design.
  AFT top1=0.31 vs naive_cosine=0.12, Δ=+0.18 [0.11, 0.26], p<0.0001, Hedges g=0.424.
  Prior expectation: FAIL. `cross_domain_affect_replication` → `controlled_evidence`.
  Closes WS3b. See `benchmarks/preregistration_addendum_m_fr_closure.md`.
- **Resonance amplification Hi3 (N=500)**: e5-small-v2 shows larger resonance
  interference than SBERT on semantic_confound queries (Δ=+0.090, d=0.257,
  Holm-adj p=0.023 — PASS) and recency_confound (Δ=+0.070, p=0.023 — PASS).
  Hi3_arc FAIL (Δ=+0.010, p=0.38).

### B) End-to-end naturalistic benchmarks (no oracle affect)

- **LLM appraisal end-to-end (Hg1 — FAIL, falsified)**: with
  `LLMAppraisalEngine` (gpt-5-mini) and no preset affect, AFT dual-path
  top1=0.315 vs naive_cosine=0.325 (Δ=−0.010, p=0.367). Synchronous
  appraisal is actively harmful: `aft_llm_sync`=0.130 vs `aft_neutral`=0.315
  (−18.5 pp). The oracle-affect advantage does not transfer to automatic
  appraisal under this protocol.
- **LoCoMo external QA benchmark (Gate 1 FAIL)**: on LoCoMo (1986 QA pairs,
  10 conversations), AFT F1=0.168 vs naive_rag=0.271 (−10.3 pp). Affective
  weighting does not improve factual open-domain QA. Add. J Pareto sweep
  (10 weight configs × 200-QA) confirms the gap is not closable via
  `base_weights` tuning (Hj1 FAIL).
- **DailyDialog ecological replication (Hk1 — FAIL)**:
  N=120 synthetic personas, 396 queries, multilingual-e5-small. AFT
  top1=0.212 vs naive_cosine=0.220 (Δ=−0.008, p_holm=1.000, d=−0.015).
  Only `affective_trajectory` queries show an underpowered positive trend
  (Δ=+0.103, d=0.186, N=39). Naturalistic short-turn dialogue does not
  show the AFT advantage; the 2-session realistic replay format does (FR PASS above).
- **Human / ecological validation**: not yet established. Kit ready at
  `benchmarks/human_eval/`; zero ratings collected.

See [Current Evidence](docs/research/09_current_evidence.md) for the study ladder
and the current claim-to-evidence matrix. The canonical machine-readable source
for those public scientific claims lives in
[`docs/research/claim_validation_matrix.json`](docs/research/claim_validation_matrix.json).

## Benchmarks

### Psychological fidelity (126 parametrized test cases, 20 phenomena)

The library validates 20 phenomena from the affective science literature via 126 parametrized test cases (run `pytest --collect-only benchmarks/fidelity/` to enumerate them):

| Phenomenon | Reference | Cases | Test file |
|---|---|---|---|
| Mood-congruent recall | Bower 1981 | 3 | [test_mood_congruent.py](benchmarks/fidelity/test_mood_congruent.py) |
| Emotional enhancement | Cahill & McGaugh 1995 | 3 | [test_emotional_enhancement.py](benchmarks/fidelity/test_emotional_enhancement.py) |
| Yerkes-Dodson inverted-U | Yerkes & Dodson 1908 | 12 | [test_yerkes_dodson.py](benchmarks/fidelity/test_yerkes_dodson.py) |
| Spacing effect | Ebbinghaus 1885 | 7 | [test_spacing_effect.py](benchmarks/fidelity/test_spacing_effect.py) |
| Arousal floor | McGaugh 2004 | 7 | [test_arousal_floor.py](benchmarks/fidelity/test_arousal_floor.py) |
| Reconsolidation (APE) | Nader & Schiller 2000 | 5 | [test_reconsolidation.py](benchmarks/fidelity/test_reconsolidation.py) |
| State-dependent retrieval | Godden & Baddeley 1975 | 3 | [test_state_dependent.py](benchmarks/fidelity/test_state_dependent.py) |
| Affective momentum | Spinoza, Ethics III | 9 | [test_momentum.py](benchmarks/fidelity/test_momentum.py) |
| Mood-adaptive weights | Heidegger, Being & Time §29 | 14 | [test_mood_adaptive.py](benchmarks/fidelity/test_mood_adaptive.py) |
| Appraisal-to-affect mapping | Scherer CPM 2009 | 11 | [test_appraisal_affect.py](benchmarks/fidelity/test_appraisal_affect.py) |
| Spreading activation | Collins & Loftus 1975 | 5 | [test_spreading_activation.py](benchmarks/fidelity/test_spreading_activation.py) |
| Hebbian co-retrieval strengthening | Hebb 1949 | 4 | [test_hebbian_strengthening.py](benchmarks/fidelity/test_hebbian_strengthening.py) |
| ACT-R power-law decay | Anderson 1983 / McGaugh 2004 | 5 | [test_decay_power_law.py](benchmarks/fidelity/test_decay_power_law.py) |
| PAD dominance | Mehrabian & Russell 1974 | 8 | [test_pad_dominance.py](benchmarks/fidelity/test_pad_dominance.py) |
| Emotional retrieval vs. cosine | Bower 1981 / Russell 1980 / Nader 2000 | 3 | [test_emotional_vs_cosine.py](benchmarks/fidelity/test_emotional_vs_cosine.py) |
| Design gap regression | (various) | 3 | [test_design_gaps.py](benchmarks/fidelity/test_design_gaps.py) |
| Dual-path encoding | LeDoux 1996 | 6 | [test_dual_path_encoding.py](benchmarks/fidelity/test_dual_path_encoding.py) |
| Emotion categorization | Plutchik 1980 | 10 | [test_emotion_categorization.py](benchmarks/fidelity/test_emotion_categorization.py) |
| Affective prediction error | Schultz 1997 / Pearce-Hall 1980 | 5 | [test_prediction_error.py](benchmarks/fidelity/test_prediction_error.py) |
| APE-gated reconsolidation window | Nader & Schiller 2000 | 3 | [test_reconsolidation_window.py](benchmarks/fidelity/test_reconsolidation_window.py) |

Run with: `make bench-fidelity`

For the comparative protocol and interpretation rules, see
[benchmarks/comparative/protocol.md](benchmarks/comparative/protocol.md).

### Performance (hash-based embedder, InMemoryStore)

| Operation | N | Mean | OPS |
|---|---|---|---|
| Encode (single) | 1 | 1.7 ms | 590/s |
| Encode (batch of 100) | 100 | 9.9 ms/op | 101/s |
| Encode w/ resonance graph | 500 | 4.0 ms | 250/s |
| Retrieve top-5 | 100 | ~2 ms | ~500/s |
| Retrieve top-5 | 1 000 | ~12 ms | ~85/s |
| Retrieve top-5 | 10 000 | ~120 ms | ~8/s |
| Retrieve (top-k 1–25) | 1 000 | 10–18 ms | 55–100/s |
| Retrieve + reconsolidation | 200 | 2.6 ms | 385/s |

`InMemoryStore.search_by_embedding` uses vectorized matrix multiplication (numpy),
making retrieval O(n · d) in a single batch rather than n individual cosine calls.
Retrieval uses two-pass scoring (spreading activation); when no resonance links are
active the second pass is skipped. For stores > 10 000 memories, use `SQLiteStore`
(sqlite-vec ANN) or a vector database implementing the `MemoryStore` protocol.

Run with: `make bench-perf`

### Appraisal quality (LLM prompt validation)

15 natural-language phrases with expected directional outcomes against Scherer's 5 dimensions:

| Phrase category | Key assertions |
|---|---|
| Personal loss ("I got fired") | `goal_relevance < -0.2`, `coping_potential < 0.6` |
| Achievement ("Got promoted") | `goal_relevance > 0.2`, `norm_congruence > 0.0` |
| Moral violation ("Coworker stole credit") | `norm_congruence < -0.2`, `goal_relevance < 0.0` |
| Grief, danger, betrayal, relief, … | dimension-specific directional bounds |

Assertions use wide bands (e.g. `> 0.3`, `< -0.2`) and evaluate the median over 3 LLM calls to tolerate non-determinism. Designed to catch systematic prompt regressions, not exact calibration.

Run with: `EMOTIONAL_MEMORY_LLM_API_KEY=... make bench-appraisal`

Works with any OpenAI-compatible endpoint (Ollama, vLLM, LiteLLM, …) via `EMOTIONAL_MEMORY_LLM_BASE_URL`.

## Production readiness

`emotional-memory` is production-hardened for teams that need supply-chain assurances:

| Signal | Status |
|---|---|
| **PyPI releases** | Trusted Publishing (OIDC, no long-lived tokens) |
| **SLSA provenance** | Level 3 — build-provenance attestation on every release |
| **SBOM** | CycloneDX JSON generated and attested per release (`dist/sbom.cdx.json`) |
| **PEP 740 attestations** | Signed attestations verifiable via `gh attestation verify` |
| **SAST** | CodeQL workflow on every push/PR to `main` |
| **Workflow security** | All third-party GitHub Actions SHA-pinned; zizmor static analysis in CI |
| **Dependency audit** | pip-audit in CI on every push; no known CVEs |
| **Coverage** | ≥80% branch coverage enforced; informational target 90% |
| **Type safety** | mypy strict + basedpyright (secondary) on every PR |
| **Conventional commits** | PR title enforced (amannn/action-semantic-pull-request) |

```bash
# Verify provenance of a released wheel locally:
gh attestation verify emotional_memory-0.11.0-py3-none-any.whl \
  --repo gianlucamazza/emotional-memory
```

## mem0 integration

`EmotionalMemoryMem0Backend` exposes the mem0 `Memory` API (`add` / `search` / `get_all` /
`delete` / `delete_all` / `reset` / `close`) backed by the full AFT retrieval pipeline.
No runtime `mem0ai` dependency is required — it's always available:

```bash
uv pip install "emotional-memory[sentence-transformers]"
```

```python
from emotional_memory import EmotionalMemory, InMemoryStore
from emotional_memory.embedders import SentenceTransformerEmbedder
from emotional_memory.integrations import EmotionalMemoryMem0Backend

em = EmotionalMemory(store=InMemoryStore(), embedder=SentenceTransformerEmbedder())
backend = EmotionalMemoryMem0Backend(em, default_user_id="alice")

backend.add([{"role": "user", "content": "I had a wonderful day at the park."}])
results = backend.search("outdoors positive experiences")
print(results["results"][0]["memory"])
```

The backend stores memories verbatim. For LLM-based fact extraction, chain a real `mem0.Memory`
instance as a pre-processor and store its extracted facts here. See
[the mem0 tutorial](docs/tutorials/mem0.md) for the chain pattern.

## LangChain integration

`EmotionalMemoryChatHistory` is a drop-in replacement for any LangChain chat history object.
It backs the transcript with an `EmotionalMemory` instance so the affective state evolves
naturally as the conversation unfolds, while letting you control which messages become
retrievable memories.

```bash
uv pip install "emotional-memory[langchain,sentence-transformers]"
```

```python
from emotional_memory import EmotionalMemory, InMemoryStore
from emotional_memory.embedders import SentenceTransformerEmbedder
from emotional_memory.integrations import (
    EmotionalMemoryChatHistory,
    recommended_conversation_policy,
)

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=SentenceTransformerEmbedder(),
)
history = EmotionalMemoryChatHistory(em, message_policy=recommended_conversation_policy)

# Works anywhere BaseChatMessageHistory is accepted:
history.add_user_message("I'm anxious about the deadline.")
history.add_ai_message("Let's break the work into smaller steps.")

print(history.messages)   # [HumanMessage(...), AIMessage(...)]

# The underlying engine has tracked the affective state:
state = em.get_state()
print(f"valence={state.core_affect.valence:.2f}  arousal={state.core_affect.arousal:.2f}")
```

With `recommended_conversation_policy`, user messages become retrievable memories, assistant
messages update affective state without being stored, and control commands such as
`recall ...` are ignored by retrieval. The adapter uses dependency injection — pass a
fully-configured `EmotionalMemory` so you control the store backend and embedder.
`clear()` removes stored memories, clears the transcript, and resets affective state.

## Logging & Observability

The library uses the standard `logging` module. Enable debug output to trace the full pipeline:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# or just for emotional_memory:
logging.getLogger("emotional_memory").setLevel(logging.DEBUG)
```

Debug events include: encode start/stored/resonance, retrieve start/done, reconsolidation
triggers, LLM appraisal cache hits, and fallback activations.

A convenience helper configures the root logger with sensible defaults, optional
JSON formatting, and environment-variable level control:

```python
from emotional_memory import configure_logging

configure_logging(level="DEBUG")  # or "INFO", "WARNING", "ERROR"
# JSON output for production pipelines:
configure_logging(level="INFO", json_format=True)
```

Set the level via environment variable without code changes:

```bash
EMOTIONAL_MEMORY_LOG_LEVEL=DEBUG uv run python my_script.py
```

### OpenTelemetry tracing

Install the optional `[otel]` extra to get distributed spans on every engine operation:

```bash
uv pip install "emotional-memory[otel]"
```

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

provider = TracerProvider()
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
# wire to your OTLP/Jaeger/Zipkin backend instead of InMemorySpanExporter

from opentelemetry import trace
trace.set_tracer_provider(provider)

# now all em.encode(), em.retrieve(), em.prune(), etc. emit spans automatically
```

Root spans are emitted for `encode`, `retrieve`, `encode_batch`, `elaborate`, `observe`, and
`prune`. Child spans cover individual `embed` and `store.search_by_embedding` calls.
**Without the `[otel]` extra, all tracing is zero-overhead no-op.**

## Examples

The [`examples/`](examples/) directory contains runnable scripts covering the full API.
All scripts are self-contained and use a deterministic `HashEmbedder` so they run without
any ML dependencies.

| Script | Description | Extra |
|--------|-------------|-------|
| `basic_usage.py` | Encode/retrieve, reconsolidation, resonance links | — |
| `advanced_config.py` | ACT-R decay, mood regression, adaptive weights | — |
| `appraisal_engines.py` | Keyword, static, and custom appraisal rules | — |
| `reconsolidation.py` | Two-retrieval lability window (Nader & Schiller 2000) | — |
| `async_usage.py` | `as_async()`, `SyncToAsync*` adapters, `encode_batch` | — |
| `persistence.py` | SQLiteStore, save/load state, export/import, prune | `[sqlite]` |
| `emotional_journal.py` | Multi-session journaling with full lifecycle | `[sqlite]` |
| `llm_appraisal.py` | LLM-backed appraisal via OpenAI-compatible API | `openai` |
| `httpx_llm_integration.py` | httpx LLMCallable, `.env` config, 7 API deep-dives | `httpx` |
| `sentence_transformers_embedder.py` | `SequentialEmbedder` with real embeddings | `sentence-transformers` |
| `visualization.py` | All 8 matplotlib plot types | `[viz]` |
| `resonance_network.py` | Resonance graph and link-type distribution | `[viz]` |
| `retrieval_signals.py` | 6-signal decomposition, radar chart, weight heatmap | `[viz]` |

Run any script: `uv run python examples/<script>.py`

## Development

```bash
make check                    # lint + typecheck + test
make cov                      # tests with branch coverage report
make bench                    # fidelity + performance benchmarks

# Real-LLM tests (require API key):
make llm-config                # print resolved LLM config (no secrets)
make test-llm                 # end-to-end integration tests
make bench-appraisal          # Scherer CPM prompt quality benchmarks
```

### LLM test environment variables

| Variable | Default | Purpose |
|---|---|---|
| `EMOTIONAL_MEMORY_LLM_API_KEY` | — | API key (required) |
| `EMOTIONAL_MEMORY_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `EMOTIONAL_MEMORY_LLM_MODEL` | `gpt-5-mini` | Model |
| `EMOTIONAL_MEMORY_LLM_REASONING_EFFORT` | `""` | Reasoning budget for o-series / gpt-5 models (`minimal` / `low` / `medium` / `high`); omitted when empty |
| `EMOTIONAL_MEMORY_LLM_OUTPUT_MODE` | `plain` | LLM response mode: `plain` or `json_object` |
| `EMOTIONAL_MEMORY_LLM_TIMEOUT_SECONDS` | `30` | HTTP timeout for OpenAI-compatible calls |
| `EMOTIONAL_MEMORY_LLM_REPEATS` | `3` | Repeats per phrase in quality benchmarks |

## Citing

If you use `emotional-memory` in research, please cite:

```bibtex
@software{mazza_emotional_memory_2026,
  author    = {Mazza, Gianluca},
  title     = {{emotional-memory: Affective Field Theory for LLM Memory}},
  year      = {2026},
  version   = {0.11.0},
  doi       = {10.5281/zenodo.20070143},
  url       = {https://github.com/gianlucamazza/emotional-memory},
  license   = {MIT},
}
```

Concept DOI (all versions): [10.5281/zenodo.19972258](https://doi.org/10.5281/zenodo.19972258)

## License

MIT — see [LICENSE](LICENSE)
