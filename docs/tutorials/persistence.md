# Persistence

By default `EmotionalMemory` uses `InMemoryStore`, which is ephemeral.  For
production use you want memories and affective state to survive process
restarts.  This tutorial covers:

1. **`SQLiteStore`** — durable on-disk storage with ANN search
2. **`save_state` / `load_state`** — affective state continuity
3. **`export_memories` / `import_memories`** — backup and migration
4. **`prune()`** — removing decayed memories

## Installation

```bash
pip install emotional-memory[sqlite]
```

`SQLiteStore` requires `sqlite-vec` for approximate-nearest-neighbour search.

## Session 1 — encode and close

```python
from emotional_memory import (
    CoreAffect, EmotionalMemory, EmotionalMemoryConfig,
    InMemoryStore, RetrievalConfig, SQLiteStore,
)

DB_PATH = "my_agent.db"

with EmotionalMemory(store=SQLiteStore(DB_PATH), embedder=MyEmbedder()) as em:
    em.set_affect(CoreAffect(valence=0.9, arousal=0.8))
    em.encode("Landed a major client — six-figure contract signed.",
               metadata={"category": "sales"})

    em.set_affect(CoreAffect(valence=-0.6, arousal=0.7))
    em.encode("Server outage lasted three hours on Black Friday.",
               metadata={"category": "ops"})

    print(f"Encoded {len(em)} memories.")

    # Save mood trajectory so the next session resumes in-context
    state_snapshot = em.save_state()

# Context manager calls close() — SQLite connection is flushed and closed.
```

!!! tip
    Always use `EmotionalMemory` as a context manager (`with ... as em:`) or
    call `em.close()` explicitly.  `SQLiteStore` holds an open connection; not
    closing it can leave the database in a locked state.

## Session 2 — reopen and restore

```python
import json, pathlib

# Persist the snapshot between sessions (file, Redis, database, etc.)
pathlib.Path("state.json").write_text(json.dumps(state_snapshot))

# --- next process start ---
state_snapshot = json.loads(pathlib.Path("state.json").read_text())

with EmotionalMemory(store=SQLiteStore(DB_PATH), embedder=MyEmbedder()) as em:
    print(f"Memories on disk: {len(em)}")

    em.load_state(state_snapshot)          # resume mood + momentum history
    sm = em.get_state().mood
    print(f"Restored mood: valence={sm.valence:.3f}  arousal={sm.arousal:.3f}")

    results = em.retrieve("client deal revenue", top_k=2)
    for i, mem in enumerate(results, 1):
        print(f"{i}. {mem.content[:60]}")
        print(f"   valence={mem.tag.core_affect.valence:+.2f}  "
              f"retrieval_count={mem.tag.retrieval_count}")
```

`load_state()` restores:

| Field | Description |
|---|---|
| `mood.valence` / `mood.arousal` | current PAD mood background |
| `momentum._history` | last 3 `CoreAffect` snapshots (velocity/acceleration) |
| `mood_decay` config | EMA parameters |

## Export and import — backup & migration

`export_memories()` serialises all memories as a list of JSON-safe dicts.
`import_memories()` re-creates them in any `MemoryStore`.

```python
# Export from SQLite
with EmotionalMemory(store=SQLiteStore(DB_PATH), embedder=MyEmbedder()) as em:
    exported = em.export_memories()
    pathlib.Path("backup.json").write_text(json.dumps(exported))

# Import into a fresh in-memory store (e.g. for unit tests)
data = json.loads(pathlib.Path("backup.json").read_text())
em_fresh = EmotionalMemory(store=InMemoryStore(), embedder=MyEmbedder())
count = em_fresh.import_memories(data)
print(f"Imported {count} memories.")

# import_memories is idempotent by default
count2 = em_fresh.import_memories(data)              # 0 — all duplicates skipped
count3 = em_fresh.import_memories(data, overwrite=True)  # re-writes all
```

Use this pattern to:

- **Back up** memories before schema migrations
- **Migrate** from `InMemoryStore` to `SQLiteStore` (or vice-versa)
- **Seed** a fresh agent with curated memories

## Pruning weak memories

`prune(threshold)` removes memories whose effective strength (ACT-R power-law
decay modulated by arousal) falls below `threshold`.  Keeps the store lean in
long-running agents.

```python
removed = em.prune(threshold=0.05)   # default production threshold
print(f"Removed {removed} weak memories.")
print(f"Remaining: {len(em)}")
```

`threshold=0.0` removes nothing (all memories have at least residual
strength); typical production values are `0.01`–`0.10`.

## Async variant

`AsyncEmotionalMemory.prune()` and `export_memories()` / `import_memories()`
work identically; `save_state()` and `load_state()` are synchronous on both
engines.  See [Async tutorial](async.md) for the full async API.

## See also

- [`EmotionalMemory` API reference](../api/engine.md)
- [`SQLiteStore`](../api/stores.md)
- [Async tutorial](async.md)
- `examples/persistence.py` in the repository
