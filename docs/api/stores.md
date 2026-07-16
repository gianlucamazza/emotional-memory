# Stores

## Memory Stores

### InMemoryStore

Reference store for tests and small single-process agents. `search_by_embedding`
is a full-scan cosine over a **lazily cached** embedding matrix (rebuilt after
save/update/delete). Suitable up to roughly thousands of vectors; for larger
N or durability see [Performance & Scaling](../guides/performance_scaling.md).

::: emotional_memory.stores.in_memory.InMemoryStore

### SQLiteStore

!!! note
    Requires the `sqlite` extra: `uv pip install "emotional-memory[sqlite]"`

::: emotional_memory.stores.sqlite.SQLiteStore

### QdrantStore

!!! note
    Requires the `qdrant` extra: `uv pip install "emotional-memory[qdrant]"`

::: emotional_memory.stores.qdrant.QdrantStore

### ChromaStore

!!! note
    Requires the `chroma` extra: `uv pip install "emotional-memory[chroma]"`

::: emotional_memory.stores.chroma.ChromaStore

## Affective State Stores

These backends persist the engine's runtime affective state (valence, arousal,
momentum, mood) across sessions.  Pass one as `state_store=` when constructing
`EmotionalMemory` or `AsyncEmotionalMemory`.

### InMemoryAffectiveStateStore

::: emotional_memory.state_stores.in_memory.InMemoryAffectiveStateStore

### SQLiteAffectiveStateStore

No extra dependencies required — uses the stdlib `sqlite3` module.

::: emotional_memory.state_stores.sqlite.SQLiteAffectiveStateStore

### RedisAffectiveStateStore

!!! note
    Requires the `redis` extra: `uv pip install "emotional-memory[redis]"`

::: emotional_memory.state_stores.redis.RedisAffectiveStateStore
