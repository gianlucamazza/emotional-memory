# Module Overview

This page is the narrative map of the codebase: what each module does and how the
pieces compose into the encode / retrieve pipeline. For the symbol-level reference
(classes, methods, signatures) follow the links into the auto-generated
[API Reference](../api/engine.md). For the theory behind each layer see
[Research](../research/index.md).

## The 5-layer emotional model

`emotional_memory` implements **Affective Field Theory (AFT)** — emotion modelled as a
distributed, dynamic, multi-layer *field* rather than a discrete label. Five layers are
captured at encoding time:

| Layer | Class | Theory | Reference |
|-------|-------|--------|-----------|
| 1 | `CoreAffect` | Russell 1980 valence-arousal circumplex (+ PAD dominance) | [Affect](../api/affect.md) |
| 2 | `AffectiveMomentum` | Spinozist velocity + acceleration over a 3-point history | [Affect](../api/affect.md) |
| 3 | `MoodField` | Heidegger's slow-moving PAD mood background (EMA) | [Mood](../api/mood.md) |
| 4 | `AppraisalVector` | Scherer's 5 Stimulus Evaluation Checks → `CoreAffect` | [Appraisal](../api/appraisal.md) |
| 5 | `ResonanceLink` | Bidirectional associative graph; spreading activation (Collins & Loftus 1975) + Hebbian strengthening (Hebb 1949) | [Resonance](../api/resonance.md) |

The runtime affective state that ties the first three layers together lives in
[`AffectiveState`](../api/state.md); the Plutchik categorization of an
`(valence, arousal)` pair into a labelled emotion lives in
[`categorize`](../api/categorize.md).

## Orchestration

| Module | Purpose | API |
|--------|---------|-----|
| `engine.py` | `EmotionalMemory`, `EmotionalMemoryConfig` — sync orchestrator for the full encode/retrieve/observe/elaborate/prune pipeline | [Engine (sync)](../api/engine.md) |
| `async_engine.py` | `AsyncEmotionalMemory` — async facade mirroring the sync engine | [Engine (async)](../api/async_engine.md) |
| `models.py` | `Memory`, `EmotionalTag` — Pydantic models for stored memories and their affective annotations | [Models](../api/models.md) |
| `retrieval.py` | 6-signal composite scoring, `build_retrieval_plan()`, retrieval config and explanation types | [Retrieval](../api/retrieval.md) |
| `decay.py` | ACT-R power-law decay with arousal modulation (McGaugh 2004) and spacing effect | [Decay](../api/decay.md) |
| `appraisal_schema.py` | Pluggable appraisal-theory schemas (Scherer CPM, OCC, GRID, custom) | [Appraisal Schema](../api/appraisal_schema.md) |
| `query_classifier.py` | Pluggable query-type routing driving per-type retrieval weights | [Query Classifier](../api/query_classifier.md) |
| `telemetry.py` | `traced_span()` — OpenTelemetry spans, no-op without the `[otel]` extra | [Telemetry](../api/telemetry.md) |
| `visualization.py` | 8 matplotlib plotting functions (optional `[viz]` extra) | [Visualization](../api/visualization.md) |

## Extension points (protocols)

`Embedder`, `MemoryStore`, and `AffectiveStateStore` are `typing.Protocol`s — duck-typed,
no inheritance required. Bring your own implementation, or use the bundled ones:

| Module | Purpose | API |
|--------|---------|-----|
| `interfaces.py` | `Embedder`, `MemoryStore`, `AffectiveStateStore` protocols + `SequentialEmbedder` base class | [Interfaces](../api/interfaces.md) |
| `interfaces_async.py` | `AsyncEmbedder`, `AsyncMemoryStore`, `AsyncAppraisalEngine` protocols | [Interfaces](../api/interfaces.md) |
| `embedders/` | `SentenceTransformerEmbedder` (production embedder, `[sentence-transformers]` extra) | [Embedders](../api/embedders.md) |
| `stores/` | `SQLiteStore` (sqlite-vec ANN), `QdrantStore`, `ChromaStore` — plus the in-memory reference store | [Stores](../api/stores.md) |
| `state_stores/` | `InMemoryAffectiveStateStore`, `SQLiteAffectiveStateStore`, `RedisAffectiveStateStore` — persist runtime affect across sessions | [State Stores](../api/state_stores.md) |
| `appraisal_llm.py` | `LLMAppraisalEngine` (LLM-backed, cached) + `KeywordAppraisalEngine` (rule-based fallback) | [Appraisal](../api/appraisal.md) |
| `integrations/` | LangChain ([tutorial](../tutorials/langchain.md)) and mem0 ([tutorial](../tutorials/mem0.md)) adapters | — |

## Key data flow

**encode** — `AppraisalVector → CoreAffect → AffectiveState.update() → EmotionalTag → [Plutchik label if auto_categorize] → embed → store → resonance links (forward + backward)`.

**encode (dual-path, LeDoux 1996)** — with `dual_path_encoding=True` and an appraisal engine: the fast path skips appraisal, uses the raw `core_affect`, and marks `pending_appraisal=True`; a later `elaborate(memory_id)` runs full appraisal and blends core affect (default 70% appraised / 30% raw).

**retrieve** — `embed query → build_retrieval_plan() → Pass 1 (6-signal score, no spreading) → seed set → spreading_activation() (multi-hop BFS) → Pass 2 (activation boost) → per-memory APE computation → APE-gated reconsolidation → Hebbian strengthening on co-retrieved links → top-k`. Use `retrieve_with_explanations()` to expose the per-signal breakdown.

**observe** — update the affective state from content *without* storing a retrievable memory (useful for assistant turns or system events).

**prune** — iterate all memories, compute effective strength via [`decay`](../api/decay.md), delete those below the threshold.

**state persistence** — `save_state()`/`load_state()` round-trip the affective state (including momentum history) as a JSON-safe dict; `persist_state()`/`restore_persisted_state()` delegate to the configured `AffectiveStateStore`.

## See also

- [Mental Model](../mental_model.md) — a 5-minute developer walkthrough of the same pipeline
- [Getting Started](../getting-started.md) — install and run a first example
- [Research](../research/index.md) — the theoretical foundations of each layer
