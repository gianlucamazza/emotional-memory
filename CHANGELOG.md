# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2026-04-12

### Fixed

- **CHANGELOG accuracy** — v0.4.0 incorrectly described `reconsolidate()` as using a
  "sigmoid-scaled adaptive learning rate"; the actual formula is linear: `alpha = min(ape * lr, 0.5)`.
  Pearce-Hall associability is handled exclusively by `update_prediction()`, not `reconsolidate()`.
- **Dead `adapt_rate` parameter removed** from `reconsolidate()` (`retrieval.py`) — the
  `adapt_rate=True` Pearce-Hall branch was unreachable dead code; no engine ever called it with
  `True`. Removing it eliminates a misleading public signature.
- **`stores/__init__.py` `__all__`** — added `SQLiteStore` so `from emotional_memory.stores import *`
  exports it correctly when `sqlite-vec` is installed.
- **Duplicate `AffectiveState` in docs** — removed the redundant `:::` directive from
  `docs/api/affect.md`; the class is documented exclusively in `docs/api/state.md`.
- **CHANGELOG v0.1.0/v0.2.0 terminology** — replaced stale "Stimmung" references with
  "MoodField"/"mood" throughout the historical changelog entries for consistency.
- **Module docstring artifacts** — removed internal "Step N:" prefixes from module-level
  docstrings in `engine.py`, `decay.py`, `retrieval.py`, `resonance.py`, `state.py`.

### Added

- **`elaborate()` / `elaborate_pending()` async tests** — 11 new tests in `test_async_engine.py`
  covering both methods (clear pending flag, blend affect, persist to store, window, edge cases).
- **`SyncToAsyncStore` direct tests** — `update()` and `search_by_embedding()` adapter methods
  now have dedicated unit tests.
- **NaN embedding warning tests** — sync and async engines both verified to emit `warnings.warn`
  when the embedder returns NaN values.
- **Reconsolidation window expiry test** — explicit test for the branch that clears
  `window_opened_at` when the lability window has elapsed.
- **Async `import_memories(overwrite=True)` test** — overwrite path previously untested.
- **Async `auto_categorize` during encode** — verified `emotion_label` is attached when the flag
  is set, and absent when it is not.
- **Concurrency tests** — threading test for independent sync engines; `asyncio.gather` test
  for independent async engines; concurrent read test for `SQLiteStore`.
- **`SQLiteStore` edge cases** — `update()` with a changed embedding vector replaces the vec
  table row correctly; dimension mismatch raises `sqlite3.OperationalError`.
- **Fidelity benchmark: Hebbian co-retrieval strengthening** (`test_hebbian_strengthening.py`) —
  4 tests validating Hebb (1949): co-retrieval increases link strength, monotonic growth over
  rounds, zero-increment leaves strength unchanged, strength capped at 1.0.
- **Fidelity benchmark: ACT-R power-law decay** (`test_decay_power_law.py`) — 5 tests
  verifying Anderson (1983) + McGaugh (2004): strictly decreasing strength, log-log linearity
  R² > 0.99, arousal slows decay, high-arousal floor respected, low-arousal can fall below floor.
- **Fidelity benchmark: PAD dominance** (`test_pad_dominance.py`) — 8 tests (+ parametrised)
  validating Mehrabian & Russell (1974): positive×high-arousal raises dominance, negative×high
  lowers it, low arousal stays near neutral, dominance clamped to [0, 1], formula verified
  numerically.

### Documentation

- **README** — `EmotionalMemory` API table now includes `elaborate()` and `elaborate_pending()`;
  `AsyncEmotionalMemory` coroutine list now includes `elaborate`, `elaborate_pending`, `count`.

## [0.4.0] - 2026-04-12

### Added

- **Discrete emotion categorization** (`categorize.py`) — `EmotionLabel`, `categorize_affect()`,
  `label_tag()`: maps continuous (valence, arousal) coordinates to Plutchik's 8 primary emotions
  with intensity tiers (low/moderate/high) via angular sector lookup in the Russell circumplex;
  optional dominance parameter disambiguates fear vs anger (Mehrabian & Russell 1974)
- **`auto_categorize` config flag** — when `True`, every `encode()` / `encode_batch()` call
  automatically attaches an `EmotionLabel` to the stored `EmotionalTag`
- **Dual-speed encoding** (LeDoux, 1996) — `dual_path_encoding` config flag enables fast
  thalamo-amygdala path (`pending_appraisal=True`, no appraisal call); `elaborate(memory_id)` runs
  the slow thalamo-cortical appraisal later and blends affect (70% appraised / 30% raw);
  `elaborate_pending()` processes all outstanding fast-path memories in one call
- **Adaptive prediction error** (Schultz 1997, Pearce-Hall 1980) — `compute_ape()` computes
  affective prediction error against `expected_affect` (EMA prediction) when available; called on
  every retrieval so the prediction model learns continuously; `update_prediction()` applies
  Pearce-Hall associability: large errors increase the learning rate, small errors decrease it
- **APE-gated reconsolidation window** — `window_opened_at` field on `EmotionalTag` separates
  window-opening (requires APE above threshold) from `last_retrieved` (any retrieval); fixes the
  prior behaviour where any retrieval could open the lability window
- 76 new tests across `tests/test_categorize.py`, `tests/test_prediction.py`,
  `tests/test_dual_path.py` and 4 new fidelity benchmarks in `benchmarks/fidelity/`

### Changed

- `reconsolidate()` now applies a linearly-scaled alpha (`min(ape * learning_rate, 0.5)`) so
  larger prediction errors produce proportionally larger core affect updates, capped at 50% per
  retrieval (Schultz 1997); Pearce-Hall associability is handled separately by `update_prediction()`
- `encode_batch()` now honours `dual_path_encoding` and `auto_categorize` flags, consistent with
  the single-item `encode()` path

## [0.3.0] - 2026-04-12

### Added

- **Spreading activation** (Collins & Loftus, 1975) — `spreading_activation()` in `resonance.py`
  performs BFS-based multi-hop propagation through the associative link graph; activation decays
  multiplicatively per hop and uses max-aggregation to prevent path-count inflation; configurable
  via `ResonanceConfig.propagation_hops` (1–5, default 2)
- **Bidirectional resonance links** — encoding a memory now creates backward links on all target
  memories so activation flows in both directions through the network; the weakest existing link is
  evicted if the target is already at `max_links`
- **Hebbian co-retrieval strengthening** (Hebb, 1949) — `hebbian_strengthen()` in `resonance.py`
  increments the strength of every link shared between memories returned in the same retrieval call
  ("neurons that fire together wire together"); increment configurable via
  `ResonanceConfig.hebbian_increment` (default 0.05, capped at 1.0)
- **Configurable link-classification thresholds** — causal, contrastive, and temporal thresholds
  that were previously hardcoded magic numbers in `_classify_link_type()` are now named fields on
  `ResonanceConfig`: `contrastive_temporal_threshold`, `contrastive_valence_threshold`,
  `causal_temporal_threshold`, `causal_semantic_threshold`
- **Vectorized `InMemoryStore.search_by_embedding`** — rewrites the per-memory Python loop with a
  NumPy batch matrix multiply + `np.argpartition` (O(n)) for top-k selection; significant speedup
  for stores > 500 memories
- `spreading_activation` and `hebbian_strengthen` exported from the top-level package (`__all__`)

### Breaking Changes

- **`StimmungField` → `MoodField`** — import from `emotional_memory.mood`; the old
  `emotional_memory.stimmung` module is removed entirely
- **`StimmungDecayConfig` → `MoodDecayConfig`** — same module move
- **`EmotionalMemoryConfig.stimmung_alpha` → `mood_alpha`**
- **`EmotionalMemoryConfig.stimmung_decay` → `mood_decay`**
- **`EmotionalTag.stimmung_snapshot` → `mood_snapshot`**
- **`AffectiveState.stimmung` → `mood`**
- **`get_current_stimmung()` → `get_current_mood()`** on both `EmotionalMemory` and
  `AsyncEmotionalMemory`
- **`make_emotional_tag()` parameter `stimmung` → `mood`**
- **`EmotionalTag` is now frozen** (`model_config = ConfigDict(frozen=True)`) — consistent
  with all other value objects; mutating tag fields now raises `ValidationError`

### Fixed

- **Decay formula boost** — `compute_effective_strength()` no longer returns a value above
  the initial `consolidation_strength` for very small elapsed times (power-law exponent can
  produce values > 1 when `elapsed < 1 s`)
- **Calm-event floor** — `consolidation_strength()` now has a minimum of `0.1`; memories
  encoded under low-arousal states are no longer immediately prunable
- **`RetrievalConfig.base_weights` length** — a Pydantic `field_validator` now raises
  `ValidationError` if the list does not contain exactly 6 elements
- **`ResonanceLink.strength` range** — field is now declared with `ge=0.0, le=1.0`
- **`as_async()` documentation** — docstring now correctly states that state is copied at
  wrap time; the two engines are independent afterwards

### Changed

- Docstrings reworked for theoretical honesty: "implements X" → "inspired by X" where the
  code is a simplification (Scherer CPM note added; Heidegger reference demoted to loose
  inspiration in `mood.py`)
- `appraisal.py` module docstring notes that the CPM evaluation is a simultaneous linear
  combination, not the original sequential model

## [0.2.0] - 2026-04-10

### Added

- **13 runnable examples** covering the full public API — `basic_usage`, `advanced_config`,
  `appraisal_engines`, `async_usage`, `emotional_journal`, `httpx_llm_integration`,
  `llm_appraisal`, `persistence`, `reconsolidation`, `resonance_network`, `retrieval_signals`,
  `sentence_transformers_embedder`, `visualization`; each is self-contained and always runnable
  without ML dependencies
- **Visualization module** (`visualization.py`) — 8 matplotlib plot functions: circumplex,
  decay curves, Yerkes-Dodson, retrieval radar, mood evolution, adaptive weights heatmap,
  resonance network, appraisal radar; install via `pip install emotional-memory[viz]`
- **`python-dotenv` optional extra** (`pip install emotional-memory[dotenv]`) and
  `make install-dotenv` Makefile target
- **`examples/httpx_llm_integration.py`** — SDK-agnostic LLM pipeline using raw httpx; covers
  `AffectiveMomentum`, `LLMCallable`, `ResonanceLink`, `SyncToAsyncAppraisalEngine`,
  `make_emotional_tag`, `consolidation_strength`, and `__version__` (previously uncovered)
- **`examples/emotional_journal.py`** — capstone multi-session journaling app combining
  `SQLiteStore`, `KeywordAppraisalEngine`, `MoodDecayConfig`, mood-congruent retrieval,
  reconsolidation, and `prune()`
- **MkDocs documentation site** with API reference (mkdocstrings) and research pages
- **`prune(threshold=0.05)`** on `EmotionalMemory` and `AsyncEmotionalMemory` — removes memories
  whose `compute_effective_strength()` has fallen below the given threshold; returns count removed
- **`export_memories()` / `import_memories(data, overwrite=False)`** on both engines — bulk
  serialise all memories to a list of JSON-safe dicts for backup or store migration;
  `import_memories` skips duplicate IDs by default, returns count written
- **`close()` and context-manager support** on both engines — `with EmotionalMemory(...) as em`
  and `async with AsyncEmotionalMemory(...) as em` propagate cleanup to the underlying store
  (calls `store.close()` when available, no-ops otherwise)
- **`SequentialEmbedder`** base class in `interfaces.py` — subclass and implement `embed()`;
  `embed_batch()` is provided automatically as a sequential fallback; exported from top-level `__init__`
- **`SQLiteStore` re-export** — now importable as `from emotional_memory import SQLiteStore`
  (when `sqlite-vec` is installed); also re-exported from `emotional_memory.stores`
- **Structured logging** — `engine.py`, `async_engine.py`, and `appraisal_llm.py` emit `DEBUG`
  log records at key pipeline points (encode start/stored/resonance, retrieve start/done,
  reconsolidate, cache hit/fallback) via `logging.getLogger(__name__)`
- **`__repr__`** on all non-Pydantic concrete classes — `EmotionalMemory`, `AsyncEmotionalMemory`,
  `InMemoryStore`, `SQLiteStore`, `LLMAppraisalEngine`, `KeywordAppraisalEngine`,
  `StaticAppraisalEngine`
- **`__slots__`** on all non-Pydantic classes — reduces per-instance memory footprint and
  prevents accidental attribute creation
- **Smoke test for `examples/basic_usage.py`** (`tests/test_examples.py`) — executed via
  `runpy.run_path` to catch silent breakage in the example script
- **LLM integration tests** (`tests/test_llm_integration.py`) — 5 end-to-end tests against a
  real OpenAI-compatible endpoint; gated behind `pytest.mark.llm` and API key env var
- **Appraisal quality benchmarks** (`benchmarks/appraisal_quality/`) — 15 natural-language
  phrases with directional assertions on Scherer's 5 dimensions; evaluates median over N repeats
- **numpy cosine similarity** — replaced pure-Python loop with `np.dot + np.linalg.norm`;
  added NaN guard returning 0.0 to prevent NaN propagation in scoring
- **Performance: hoisted `adaptive_weights()`** — computed once per `retrieve()` call instead
  of once per candidate per pass; `retrieval_score()` accepts `precomputed_weights` parameter
- **Performance: skip Pass 2** when no resonance links target the active memory set
- **Engine facade methods**: `get(memory_id)`, `list_all()`, `__len__()`/`count()` on both
  `EmotionalMemory` and `AsyncEmotionalMemory`
- **Input validation** on `encode_batch()` (metadata/contents length mismatch raises `ValueError`)
  and `retrieve()` (top_k < 1 raises `ValueError`)
- **CI jobs for optional extras** — dedicated sqlite-tests and viz-tests jobs install and
  exercise those extras explicitly so they are never silently skipped
- **`__init__.py` export smoke test** — verifies all `__all__` entries are importable
- **`AsyncEmotionalMemory`** — async-native facade mirroring `EmotionalMemory`; all I/O methods
  (`encode`, `retrieve`, `encode_batch`, `delete`) are coroutines; state accessors remain sync
- **Async protocols** in `interfaces_async.py`: `AsyncEmbedder`, `AsyncMemoryStore` (uses
  `count()` instead of `__len__`), `AsyncAppraisalEngine` — all `@runtime_checkable`
- **Sync-to-async bridge adapters** in `async_adapters.py`: `SyncToAsyncEmbedder`,
  `SyncToAsyncStore`, `SyncToAsyncAppraisalEngine`, and `as_async()` convenience wrapper
- **`SQLiteStore`** in `stores/sqlite.py` — persistent `MemoryStore` backed by SQLite +
  sqlite-vec for ANN vector search; install via `pip install emotional-memory[sqlite]`;
  context-manager support; lazy vector index creation
- **`LLMAppraisalEngine`** — provider-agnostic LLM-backed appraisal via user-supplied
  `LLMCallable` protocol; LRU cache (configurable size), fallback-on-error, markdown fence
  extraction, `LLMAppraisalConfig`
- **`KeywordAppraisalEngine`** — rule-based appraisal fallback using `KeywordRule` regex
  patterns with dimension score deltas; ships with defaults covering success, failure,
  novelty, danger, and social norms
- **`save_state()` / `load_state()`** on `EmotionalMemory` and `AsyncEmotionalMemory` —
  serialise and restore the full `AffectiveState` (core affect, momentum history, MoodField)
  as a JSON-safe dict, enabling session persistence
- **`get_current_mood(now)`** — read-only mood inspection with time-based regression
  applied on-the-fly without mutating engine state
- **`MoodDecayConfig`** — exponential mood regression toward PAD baselines, modulated
  by inertia; configurable half-life and inertia scale; applied via `MoodField.regress()`
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
- `MoodField` — slow-moving global mood with inertia and PAD-based dominance update,
  evolved via EMA (Heidegger §29 / Mehrabian & Russell 1974)
- `AppraisalVector` — emotion derived from 5-dimension cognitive evaluation with `to_core_affect()`
  mapping (Scherer CPM 2009 / Lazarus / Stoics)
- `ResonanceLink` — associative memory graph with semantic, emotional, temporal, causal, and
  contrastive link types (Aristotle / Bower 1981 spreading activation)
- `EmotionalTag` — immutable snapshot of all 5 layers at encoding time + consolidation metadata
- `EmotionalMemory` — main facade:
  - `encode(content, appraisal, metadata)` — single-item encode with full AFT pipeline
  - `encode_batch(contents, metadata)` — batched encode via `embed_batch()`, per-item appraisal
  - `retrieve(query, top_k)` — two-pass spreading activation with mood-adaptive weights
  - `delete(memory_id)` — remove a memory from the store
  - `get_state()` / `set_affect()` — read and write the runtime affective state
- `InMemoryStore` — dict-backed `MemoryStore` with brute-force cosine search
- `Embedder` and `MemoryStore` — `typing.Protocol` interfaces for dependency injection (PEP 544)
- Power-law memory decay (ACT-R, Anderson 1983), arousal-modulated, with configurable `power`
  exponent and high-arousal floor (Merleau-Ponty body memory)
- Mood-congruent retrieval: 6-signal weighted scoring (semantic, mood-congruence,
  affect-proximity, momentum-alignment, recency, resonance-boost)
- Mood-adaptive retrieval weights (Heidegger: mood is the ground of disclosure)
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

[Unreleased]: https://github.com/gianlucamazza/emotional-memory/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/gianlucamazza/emotional-memory/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/gianlucamazza/emotional-memory/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/gianlucamazza/emotional-memory/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/gianlucamazza/emotional-memory/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/gianlucamazza/emotional-memory/releases/tag/v0.1.0
