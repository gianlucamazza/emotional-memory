# Persistence

By default `EmotionalMemory` uses `InMemoryStore`, which is ephemeral.  For
production use you want memories and affective state to survive process
restarts.  This tutorial covers:

1. **`SQLiteStore`** — durable on-disk storage with ANN search (local file)
2. **`QdrantStore`** / **`ChromaStore`** — vector-database backends for scale
3. **`SQLiteAffectiveStateStore`** — canonical affective state continuity
4. **`export_memories` / `import_memories`** — backup and migration
5. **`prune()`** — removing decayed memories

## Installation

```bash
uv pip install "emotional-memory[sqlite]"    # SQLite backend
uv pip install "emotional-memory[qdrant]"    # Qdrant backend
uv pip install "emotional-memory[chroma]"    # ChromaDB backend
```

`SQLiteStore` requires `sqlite-vec` for approximate-nearest-neighbour search.
`QdrantStore` and `ChromaStore` require embeddings to be provided on every
`save()` call (vector-first strict mode — no dict fallback).

For shared affective state across instances you can also use:

```bash
uv pip install "emotional-memory[redis]"
```

## Session 1 — encode and close

```python
from emotional_memory import (
    CoreAffect, EmotionalMemory, EmotionalMemoryConfig,
    InMemoryStore, RetrievalConfig, SQLiteAffectiveStateStore, SQLiteStore,
)

DB_PATH = "my_agent.db"
STATE_PATH = "my_agent.state.sqlite"

with EmotionalMemory(
    store=SQLiteStore(DB_PATH),
    embedder=MyEmbedder(),
    state_store=SQLiteAffectiveStateStore(STATE_PATH),
) as em:
    em.set_affect(CoreAffect(valence=0.9, arousal=0.8))
    em.encode("Landed a major client — six-figure contract signed.",
               metadata={"category": "sales"})

    em.set_affect(CoreAffect(valence=-0.6, arousal=0.7))
    em.encode("Server outage lasted three hours on Black Friday.",
               metadata={"category": "ops"})

    print(f"Encoded {len(em)} memories.")

# Context manager calls close() — SQLite connection is flushed and closed.
```

!!! tip
    Always use `EmotionalMemory` as a context manager (`with ... as em:`) or
    call `em.close()` explicitly.  `SQLiteStore` holds an open connection; not
    closing it can leave the database in a locked state.

## Session 2 — reopen and restore

```python
with EmotionalMemory(
    store=SQLiteStore(DB_PATH),
    embedder=MyEmbedder(),
    state_store=SQLiteAffectiveStateStore(STATE_PATH),
) as em:
    print(f"Memories on disk: {len(em)}")

    sm = em.get_state().mood
    print(f"Restored mood: valence={sm.valence:.3f}  arousal={sm.arousal:.3f}")

    results = em.retrieve("client deal revenue", top_k=2)
    for i, mem in enumerate(results, 1):
        print(f"{i}. {mem.content[:60]}")
        print(f"   valence={mem.tag.core_affect.valence:+.2f}  "
              f"retrieval_count={mem.tag.retrieval_count}")
```

The configured `state_store` restores:

| Field | Description |
|---|---|
| `mood.valence` / `mood.arousal` | current PAD mood background |
| `momentum._history` | last 3 `CoreAffect` snapshots (velocity/acceleration) |
| `mood_decay` config | EMA parameters |

## Manual snapshots still exist

`save_state()` and `load_state()` are still available when you want to move the
state snapshot yourself (for example, custom storage, fixtures, or migrations).

```python
snapshot = em.save_state()
em.load_state(snapshot)
```

## Optional shared state with Redis

When multiple engine instances need to share the same affective-state snapshot,
configure a `RedisAffectiveStateStore`:

```python
from emotional_memory import EmotionalMemory, RedisAffectiveStateStore

em = EmotionalMemory(
    store=SQLiteStore(DB_PATH),
    embedder=MyEmbedder(),
    state_store=RedisAffectiveStateStore(
        "redis://localhost:6379/0",
        key="demo-user:affective-state",
    ),
)
```

This keeps the `state_store` contract separate from `MemoryStore`: shared mood
state is supported, but memory storage and affective-state storage are still
independent backends.

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
work identically. Snapshot helpers remain available on both engines; the async
engine also exposes `await persist_state()` and `await restore_persisted_state()`
when a state store is configured. See [Async tutorial](async.md) for the full async API.

## See also

- [`EmotionalMemory` API reference](../api/engine.md)
- [`SQLiteStore`, `QdrantStore`, `ChromaStore`](../api/stores.md)
- [Async tutorial](async.md)
- `examples/persistence.py` in the repository
