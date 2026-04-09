# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`AsyncEmotionalMemory`** — async-native facade mirroring `EmotionalMemory`; all I/O methods
  (`encode`, `retrieve`, `encode_batch`, `delete`) are coroutines; state accessors remain sync
- **Async protocols** in `interfaces_async.py`: `AsyncEmbedder`, `AsyncMemoryStore` (uses
  `count()` instead of `__len__`), `AsyncAppraisalEngine` — all `@runtime_checkable`
- **Sync-to-async bridge adapters** in `async_adapters.py`: `SyncToAsyncEmbedder`,
  `SyncToAsyncStore`, `SyncToAsyncAppraisalEngine`, and `as_async()` convenience wrapper
- **`SQLiteStore`** in `stores/sqlite.py` — persistent `MemoryStore` backed by SQLite +
  sqlite-vec for ANN vector search; install via `pip install emotional_memory[sqlite]`;
  context-manager support; lazy vector index creation
- **`LLMAppraisalEngine`** — provider-agnostic LLM-backed appraisal via user-supplied
  `LLMCallable` protocol; LRU cache (configurable size), fallback-on-error, markdown fence
  extraction, `LLMAppraisalConfig`
- **`KeywordAppraisalEngine`** — rule-based appraisal fallback using `KeywordRule` regex
  patterns with dimension score deltas; ships with defaults covering success, failure,
  novelty, danger, and social norms
- **`save_state()` / `load_state()`** on `EmotionalMemory` and `AsyncEmotionalMemory` —
  serialise and restore the full `AffectiveState` (core affect, momentum history, Stimmung)
  as a JSON-safe dict, enabling session persistence
- **`get_current_stimmung(now)`** — read-only Stimmung inspection with time-based regression
  applied on-the-fly without mutating engine state
- **`StimmungDecayConfig`** — exponential Stimmung regression toward PAD baselines, modulated
  by inertia; configurable half-life and inertia scale; applied via `StimmungField.regress()`
- **`AdaptiveWeightsConfig`** — continuous sigmoid/Gaussian modulation of retrieval weights
  replacing hard thresholds; `_smooth_gate()` helper for tanh-based gate functions
- **`ResonanceConfig.candidate_multiplier`** — pre-filter resonance candidates in large stores
  to avoid loading all memories during encode
- **Context passthrough** — `appraise(content, context=metadata)` now forwarded in both
  `encode()` and `encode_batch()` paths, enabling LLM appraisal engines to use memory metadata

## [0.1.0] - 2026-04-09

### Added

- **Affective Field Theory (AFT)** — original 5-layer emotional model for LLM memory systems
- `CoreAffect` — continuous valence/arousal circumplex (Barrett/Russell 1980)
- `AffectiveMomentum` — time-normalised velocity and acceleration of affect transitions (Spinoza)
- `StimmungField` — slow-moving global mood with inertia and PAD-based dominance update,
  evolved via EMA (Heidegger §29 / Mehrabian & Russell 1974)
- `AppraisalVector` — emotion derived from 5-dimension cognitive evaluation with `to_core_affect()`
  mapping (Scherer CPM 2009 / Lazarus / Stoics)
- `ResonanceLink` — associative memory graph with semantic, emotional, temporal, causal, and
  contrastive link types (Aristotle / Bower 1981 spreading activation)
- `EmotionalTag` — immutable snapshot of all 5 layers at encoding time + consolidation metadata
- `EmotionalMemory` — main facade:
  - `encode(content, appraisal, metadata)` — single-item encode with full AFT pipeline
  - `encode_batch(contents, metadata)` — batched encode via `embed_batch()`, per-item appraisal
  - `retrieve(query, top_k)` — two-pass spreading activation with Stimmung-adaptive weights
  - `delete(memory_id)` — remove a memory from the store
  - `get_state()` / `set_affect()` — read and write the runtime affective state
- `InMemoryStore` — dict-backed `MemoryStore` with brute-force cosine search
- `Embedder` and `MemoryStore` — `typing.Protocol` interfaces for dependency injection (PEP 544)
- Power-law memory decay (ACT-R, Anderson 1983), arousal-modulated, with configurable `power`
  exponent and high-arousal floor (Merleau-Ponty body memory)
- Mood-congruent retrieval: 6-signal weighted scoring (semantic, stimmung-congruence,
  affect-proximity, momentum-alignment, recency, resonance-boost)
- Stimmung-adaptive retrieval weights (Heidegger: mood is the ground of disclosure)
- Two-pass spreading activation: first pass seeds active memory IDs for resonance boost
- Embedding pre-filter: `candidate_multiplier` limits scoring candidates in large stores
- Reconsolidation with lability window: tag updated on high APE only within
  `reconsolidation_window_seconds` of previous retrieval (Nader & Schiller 2000)
- `DecayConfig.power` — configurable power-law scaling exponent
- 296 tests: 219 unit/integration + 77 psychological fidelity benchmarks
- 14 performance benchmarks (encode throughput, retrieve latency, memory footprint, resonance build)
- PEP 561 typed (`py.typed` marker), mypy strict, 98% branch coverage
- CI: GitHub Actions matrix (Python 3.11-3.14), Codecov upload, benchmark regression tracking
- PyPI release workflow (OIDC trusted publishing)
- Pre-commit hooks: ruff check + format
