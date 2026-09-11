# Migration Guide

## v0.2.0 → v0.3.0

v0.3.0 renames all `Stimmung*` identifiers to English equivalents. The underlying
behaviour is unchanged — this is a pure rename with no logic differences.

### Find-and-replace

Run these substitutions across your codebase (imports, config dicts, attribute access):

| Old (v0.2.0) | New (v0.3.0) |
|---|---|
| `from emotional_memory.stimmung import StimmungField, StimmungDecayConfig` | `from emotional_memory.mood import MoodField, MoodDecayConfig` |
| `from emotional_memory import StimmungField, StimmungDecayConfig` | `from emotional_memory import MoodField, MoodDecayConfig` |
| `StimmungField` | `MoodField` |
| `StimmungDecayConfig` | `MoodDecayConfig` |
| `EmotionalMemoryConfig(stimmung_alpha=...)` | `EmotionalMemoryConfig(mood_alpha=...)` |
| `EmotionalMemoryConfig(stimmung_decay=...)` | `EmotionalMemoryConfig(mood_decay=...)` |
| `tag.stimmung_snapshot` | `tag.mood_snapshot` |
| `state.stimmung` | `state.mood` |
| `em.get_current_stimmung()` | `em.get_current_mood()` |
| `make_emotional_tag(..., stimmung=...)` | `make_emotional_tag(..., mood=...)` |
| `plot_stimmung_evolution(...)` | `plot_mood_evolution(...)` |

### `EmotionalTag` is now frozen

`EmotionalTag` now has `model_config = ConfigDict(frozen=True)`. Any code that
mutated tag fields directly will raise `ValidationError`:

```python
# v0.2.0 — worked silently
tag.consolidation_strength = 0.9

# v0.3.0 — raises ValidationError
# Solution: create a new tag or use model_copy()
tag = tag.model_copy(update={"consolidation_strength": 0.9})
```

### New exports (no action required)

Two new functions are now part of the public API:

```python
from emotional_memory import spreading_activation, hebbian_strengthen
```

These are used internally by the retrieval pipeline. You only need to import them
if you are building a custom retrieval flow on top of the resonance graph.

## Explainable retrieval API

Recent releases add a supported retrieval introspection path without changing the
default recall semantics.

### New public entrypoint

Use `retrieve_with_explanations()` when you need ranking-time diagnostics:

```python
from emotional_memory import EmotionalMemory, RetrievalExplanation

results: list[RetrievalExplanation] = em.retrieve_with_explanations(
    "project success",
    top_k=3,
)
```

Each `RetrievalExplanation` contains:

- `memory`: the retrieved `Memory` object after retrieval-side updates
- `score`: the final ranking score
- `breakdown`: the structured score decomposition
- `pass1_rank` / `pass2_rank`: how resonance changed ordering
- `activation_level`: spreading-activation contribution used in pass 2

### New public types

These types are now part of the supported top-level API:

```python
from emotional_memory import (
    RetrievalSignals,
    RetrievalBreakdown,
    RetrievalExplanation,
)
```

Use them for debugging, evaluation, visualization, and typed UI/reporting code.

### Retrieval semantics

Two details matter when migrating tooling around retrieval:

- `retrieve()` remains the normal runtime recall path and returns only `Memory` objects.
- `retrieve_with_explanations()` runs the same ranking pipeline, but also returns
  the ranking-time signal breakdown.

The `breakdown` describes the score before retrieval-side mutation. The returned
`memory` reflects the stored object after post-retrieval updates such as
reconsolidation and Hebbian strengthening.

### Stability boundary

The supported public introspection surface is:

- `EmotionalMemory.retrieve_with_explanations()`
- `AsyncEmotionalMemory.retrieve_with_explanations()`
- `RetrievalSignals`
- `RetrievalBreakdown`
- `RetrievalExplanation`

Lower-level planning helpers in `emotional_memory.retrieval` are intentionally not
promoted as stable top-level API. In particular, `build_retrieval_plan()` should be
treated as an internal engine helper unless the project explicitly documents it as
stable in a future release.

## v0.16.0 → v0.17.0

Additive. `retrieve_query_gated()` (sync and async) is the Addendum Y safe wrapper:
it appraises the query and routes `|valence| < tau` (default
`EmotionalMemoryConfig.query_affect_gate_tau=0.2`) to semantic-only retrieval,
else to `retrieve_with_query_appraisal`. Requires an `appraisal_engine`. Default
`retrieve()` behaviour is unchanged. See the
[query-appraisal tutorial](tutorials/query_appraisal_retrieval.md).

## v0.17.0 → v0.18.0

### SQLite vector index ranks by cosine

Databases created by v0.18+ use `sqlite-vec` `distance_metric=cosine`, matching
the retrieval scorer. Databases written by earlier versions used L2. Opening one
emits a `UserWarning`; call `store.rebuild_vector_index()` once to migrate (the
vector table is derived — embeddings also live in the memory rows).

### Keyword appraisal scaling

`KeywordAppraisalEngine` no longer averages in rules that leave a dimension at
its neutral default. Multi-rule inputs produce stronger (less attenuated)
appraisals. The published `aft_keyword_synchronous` (Hf1) numbers used the
pre-fix engine and are not regenerated.
