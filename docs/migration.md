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
