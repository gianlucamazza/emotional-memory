# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make check          # Full suite: lint + typecheck + test (run before commits)
make test           # Run tests only
make cov            # Tests with branch coverage (80% minimum enforced)
make typecheck      # mypy strict mode
make lint           # ruff check
make format         # ruff format
make bench-fidelity # Psychological invariant tests (123 tests in benchmarks/)
make bench-perf     # Performance benchmarks
make bench          # fidelity + performance benchmarks (combined)
make install-demo   # Install canonical local Gradio demo stack
make test-llm        # Real-LLM integration tests (requires EMOTIONAL_MEMORY_LLM_API_KEY)
make bench-appraisal # LLM appraisal quality benchmarks (requires EMOTIONAL_MEMORY_LLM_API_KEY)
make install        # Install package in editable mode with dev deps
make install-llm-test # Install llm-test dependencies (httpx)
make install-dotenv  # Install dotenv dependencies (python-dotenv)
make install-bench  # Install benchmark dependencies
make install-viz    # Install visualization dependencies (matplotlib)
make install-docs   # Install documentation dependencies (mkdocs + material + mkdocstrings)
make docs-images    # Generate docs/images/ PNGs from synthetic data
make docs           # Build static documentation site (docs/ → site/)
make docs-serve     # Serve documentation locally with live reload
make dist           # Build distribution packages (wheel + sdist)
make publish        # Build and publish to PyPI
```

Single test:
```bash
uv run pytest tests/test_engine.py::test_name -v
```

### LLM test environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `EMOTIONAL_MEMORY_LLM_API_KEY` | Yes | — | API key for LLM provider |
| `EMOTIONAL_MEMORY_LLM_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `EMOTIONAL_MEMORY_LLM_MODEL` | No | `gpt-5-mini` | Model to use |
| `EMOTIONAL_MEMORY_LLM_OUTPUT_MODE` | No | `plain` | Response mode: `plain` or `json_object` |
| `EMOTIONAL_MEMORY_LLM_TIMEOUT_SECONDS` | No | `30` | HTTP timeout in seconds |
| `EMOTIONAL_MEMORY_LLM_REPEATS` | No | `3` | Repeats per phrase in quality benchmarks |

## Architecture

This library implements **Affective Field Theory (AFT)** — a 5-layer emotional model for LLM memory. The main entry point is `EmotionalMemory` (engine.py), which orchestrates the full pipeline.

### 5-Layer Emotional Model

| Layer | Class | Theory |
|-------|-------|--------|
| 1 | `CoreAffect` (affect.py) | Russell 1980 valence-arousal circumplex |
| 2 | `AffectiveMomentum` (affect.py) | Spinozist velocity + acceleration over 3-point history |
| 3 | `MoodField` (mood.py) | Heidegger's slow-moving PAD mood background (EMA) |
| 4 | `AppraisalVector` (appraisal.py) | Scherer's 5 Stimulus Evaluation Checks → CoreAffect |
| 5 | `ResonanceLink` (resonance.py) | Bidirectional associative graph: 5 link types, top-5 per memory; spreading activation (Collins & Loftus 1975) + Hebbian strengthening (Hebb 1949) |

### Additional Modules

| Module | Purpose |
|--------|---------|
| `categorize.py` | `EmotionLabel`, `categorize_affect()`, `label_tag()` — Plutchik wheel: maps (valence, arousal) to 8 primary emotions with intensity tiers (Russell 1980 + Plutchik 1980) |
| `appraisal_llm.py` | `LLMAppraisalEngine` (LLM-backed, thread-safe LRU cache) + `KeywordAppraisalEngine` (rule-based fallback) |
| `async_engine.py` | `AsyncEmotionalMemory` — async facade, mirrors `EmotionalMemory` |
| `async_adapters.py` | `SyncToAsync*` bridge adapters + `as_async()` convenience wrapper |
| `interfaces_async.py` | `AsyncEmbedder`, `AsyncMemoryStore`, `AsyncAppraisalEngine` protocols |
| `interfaces.py` | `Embedder`, `MemoryStore`, `AffectiveStateStore` protocols + `SequentialEmbedder` base class |
| `state_stores/` | `InMemoryAffectiveStateStore`, `SQLiteAffectiveStateStore`, `RedisAffectiveStateStore` — pluggable backends for persisting the runtime affective state across sessions |
| `llm_http.py` | `OpenAICompatibleLLMConfig`, `make_httpx_llm()` — thin httpx-based LLM client for appraisal |
| `integrations/langchain.py` | LangChain memory integration (optional) |
| `stores/sqlite.py` | `SQLiteStore` — persistent store with sqlite-vec ANN search |
| `visualization.py` | 8 matplotlib plotting functions (optional `viz` extra) |

### Key Data Flow

**encode (single path)**: `AppraisalVector → CoreAffect → AffectiveState.update() → EmotionalTag → [label_tag() if auto_categorize] → embed → store → resonance links (forward + backward bidirectional)`

**encode (dual-path, LeDoux 1996)**: When `dual_path_encoding=True` and an appraisal engine is configured — fast path: skip appraisal, use raw `state.core_affect`, set `pending_appraisal=True`. Slow path: call `elaborate(memory_id)` later to run full appraisal and blend core_affect (70% appraised / 30% raw).

**retrieve**: `embed query → build_retrieval_plan() (pure, no side-effects) → Pass 1 (6-signal score, no spreading) → seed set → spreading_activation() (BFS multi-hop) → Pass 2 (activation_map boost) → per-memory: compute_ape() + update_prediction() → APE-gated reconsolidation → hebbian_strengthen() on co-retrieved links → return top-k`. Use `retrieve_with_explanations()` to expose `RetrievalExplanation` / `RetrievalBreakdown` / `RetrievalSignals` per result.

**observe**: update affective state from content without storing a retrievable memory. Useful for assistant turns or system events.

**async encode/retrieve**: Same pipeline as sync. Embed/store/appraise calls are awaited. CPU-bound scoring (retrieval_score, decay, resonance) runs synchronously inline.

**state persistence**: `save_state()` → `AffectiveState.snapshot()` → JSON-safe dict (includes private `_history`). `load_state(data)` → `AffectiveState.restore()`. Round-trip preserves momentum history. `persist_state()` / `restore_persisted_state()` / `clear_persisted_state()` delegate to the configured `AffectiveStateStore`. `reset_state()` resets runtime state to baseline without touching the store.

**prune**: `prune(threshold=0.05)` → iterate all memories, call `compute_effective_strength()`, delete those below threshold. Returns count removed. Async variant awaits each store call.

**export/import**: `export_memories()` → `[Memory.model_dump_json() | json.loads(...)]`. `import_memories(data, overwrite=False)` → `Memory.model_validate(item)` per dict, skip duplicates unless `overwrite=True`. Returns count written.

**resource cleanup**: `close()` delegates to `store.close()` if available (duck-typed via `getattr`). Both engines support context manager: `with EmotionalMemory(...) as em` / `async with AsyncEmotionalMemory(...) as em`.

### 6-Signal Retrieval Scoring (retrieval.py)

Composite score weighted by current mood (adaptive weights):
1. Semantic similarity (cosine)
2. Mood congruence (Bower 1981 mood-congruent retrieval)
3. Core affect proximity
4. Momentum alignment
5. Recency/decay (ACT-R power-law)
6. Resonance boost (spreading activation, Pass 2 only)

### Decay Engine (decay.py)

ACT-R power-law: `strength(t) = initial * elapsed^(-effective_decay)`, modulated by encoding arousal (McGaugh 2004) and retrieval count (spacing effect). High-arousal memories have a floor.

### Interfaces

`Embedder` and `MemoryStore` are `typing.Protocol` in `interfaces.py` — duck-typed, no inheritance required. `MemoryStore` requires `__len__`. `InMemoryStore` is the reference implementation.

`AffectiveStateStore` protocol is also in `interfaces.py`: `save(state)`, `load() → AffectiveState | None`, `clear()`. Implementations live in `state_stores/`.

`SequentialEmbedder` in `interfaces.py` is a concrete base class: subclass it and implement `embed()`; `embed_batch()` is provided as a sequential fallback.

Async protocols live in `interfaces_async.py`: `AsyncEmbedder`, `AsyncMemoryStore` (uses `count() -> int` instead of `__len__`), `AsyncAppraisalEngine`.

`AppraisalEngine` protocol is in `appraisal.py`. `LLMCallable` protocol is in `appraisal_llm.py`.

`get_current_mood(now)` on both engines reads the `MoodField` regressed via `MoodDecayConfig` without mutating state. Configured via `EmotionalMemoryConfig.mood_decay`.

`SQLiteStore` is exported from the top-level `__init__.py` and from `stores/__init__.py` when `sqlite-vec` is installed (guarded by `contextlib.suppress(ImportError)`).

`SQLiteAffectiveStateStore` and `InMemoryAffectiveStateStore` are exported from `state_stores/__init__.py`. `RedisAffectiveStateStore` requires the `redis` extra.

## Conventions

- **Immutability**: All value objects are Pydantic `frozen=True`. `update()` methods return new instances.
- **Protocols over ABCs**: Extend via duck-typed protocols, not inheritance.
- **Config-driven**: All behavior parameterized via nested config classes (`EmotionalMemoryConfig`, `DecayConfig`, `RetrievalConfig`, `ResonanceConfig`, `MoodDecayConfig`, `AdaptiveWeightsConfig`, `LLMAppraisalConfig`). New top-level flags: `dual_path_encoding` (bool), `elaboration_learning_rate` (float, blend ratio in `elaborate()`), `auto_categorize` (bool, run Plutchik categorization on encode).
- **Theory references**: Each component cites source papers — preserve these in docstrings/comments.
- **Validation**: Field clamping via Pydantic validators (e.g., valence ∈ [-1, +1], arousal ∈ [0, 1]).
- **`__slots__`**: All non-Pydantic classes define `__slots__` for memory efficiency and attribute safety.
- **`__repr__`**: All non-Pydantic concrete classes implement a useful `__repr__`.
- **Logging**: Modules with observable pipeline events (`engine.py`, `async_engine.py`, `appraisal_llm.py`) use `logging.getLogger(__name__)` at `DEBUG` level. `warnings.warn` is reserved for user-visible degradation (e.g., NaN embeddings).
- mypy strict is enforced — all new code must be fully annotated.
- Fidelity benchmarks in `benchmarks/fidelity/` validate psychological phenomena — run after logic changes to retrieval, decay, or resonance.
- Appraisal quality benchmarks in `benchmarks/appraisal_quality/` validate LLM prompt output — run after changes to the Scherer CPM prompt in `appraisal_llm.py`.
