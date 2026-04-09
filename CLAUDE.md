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
make bench-fidelity # Psychological invariant tests (77 tests in benchmarks/)
make bench-perf     # Performance benchmarks
```

Single test:
```bash
uv run pytest tests/test_engine.py::test_name -v
```

## Architecture

This library implements **Affective Field Theory (AFT)** — a 5-layer emotional model for LLM memory. The main entry point is `EmotionalMemory` (engine.py), which orchestrates the full pipeline.

### 5-Layer Emotional Model

| Layer | Class | Theory |
|-------|-------|--------|
| 1 | `CoreAffect` (affect.py) | Russell 1980 valence-arousal circumplex |
| 2 | `AffectiveMomentum` (affect.py) | Spinozist velocity + acceleration over 3-point history |
| 3 | `StimmungField` (stimmung.py) | Heidegger's slow-moving PAD mood background (EMA) |
| 4 | `AppraisalVector` (appraisal.py) | Scherer's 5 Stimulus Evaluation Checks → CoreAffect |
| 5 | `ResonanceLink` (resonance.py) | Associative graph: 5 link types, top-5 per memory |

### Additional Modules

| Module | Purpose |
|--------|---------|
| `appraisal_llm.py` | `LLMAppraisalEngine` (LLM-backed) + `KeywordAppraisalEngine` (rule-based fallback) |
| `async_engine.py` | `AsyncEmotionalMemory` — async facade, mirrors `EmotionalMemory` |
| `async_adapters.py` | `SyncToAsync*` bridge adapters + `as_async()` convenience wrapper |
| `interfaces_async.py` | `AsyncEmbedder`, `AsyncMemoryStore`, `AsyncAppraisalEngine` protocols |
| `stores/sqlite.py` | `SQLiteStore` — persistent store with sqlite-vec ANN search |

### Key Data Flow

**encode**: `AppraisalVector → CoreAffect → AffectiveState.update() → EmotionalTag → embed → store → resonance links`

**retrieve**: `embed query → Pass 1 (6-signal score, no spreading) → seed active set → Pass 2 (resonance boost) → reconsolidation check → return top-k`

**async encode/retrieve**: Same pipeline as sync. Embed/store/appraise calls are awaited. CPU-bound scoring (retrieval_score, decay, resonance) runs synchronously inline.

**state persistence**: `save_state()` → `AffectiveState.snapshot()` → JSON-safe dict (includes private `_history`). `load_state(data)` → `AffectiveState.restore()`. Round-trip preserves momentum history.

### 6-Signal Retrieval Scoring (retrieval.py)

Composite score weighted by current Stimmung (Heidegger adaptive weights):
1. Semantic similarity (cosine)
2. Stimmung congruence (Bower 1981 mood-congruent retrieval)
3. Core affect proximity
4. Momentum alignment
5. Recency/decay (ACT-R power-law)
6. Resonance boost (spreading activation, Pass 2 only)

### Decay Engine (decay.py)

ACT-R power-law: `strength(t) = initial * elapsed^(-effective_decay)`, modulated by encoding arousal (McGaugh 2004) and retrieval count (spacing effect). High-arousal memories have a floor.

### Interfaces

`Embedder` and `MemoryStore` are `typing.Protocol` in `interfaces.py` — duck-typed, no inheritance required. `MemoryStore` requires `__len__`. `InMemoryStore` is the reference implementation.

Async protocols live in `interfaces_async.py`: `AsyncEmbedder`, `AsyncMemoryStore` (uses `count() -> int` instead of `__len__`), `AsyncAppraisalEngine`.

`AppraisalEngine` protocol is in `appraisal.py`. `LLMCallable` protocol is in `appraisal_llm.py`.

`get_current_stimmung(now)` on both engines reads Stimmung regressed via `StimmungDecayConfig` without mutating state. Configured via `EmotionalMemoryConfig.stimmung_decay`.

## Conventions

- **Immutability**: All value objects are Pydantic `frozen=True`. `update()` methods return new instances.
- **Protocols over ABCs**: Extend via duck-typed protocols, not inheritance.
- **Config-driven**: All behavior parameterized via nested config classes (`EmotionalMemoryConfig`, `DecayConfig`, `RetrievalConfig`, `ResonanceConfig`, `StimmungDecayConfig`, `AdaptiveWeightsConfig`, `LLMAppraisalConfig`).
- **Theory references**: Each component cites source papers — preserve these in docstrings/comments.
- **Validation**: Field clamping via Pydantic validators (e.g., valence ∈ [-1, +1], arousal ∈ [0, 1]).
- mypy strict is enforced — all new code must be fully annotated.
- Fidelity benchmarks in `benchmarks/fidelity/` validate psychological phenomena — run after logic changes to retrieval, decay, or resonance.
