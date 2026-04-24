# Stores

## Memory Stores

### InMemoryStore

::: emotional_memory.stores.in_memory.InMemoryStore

### SQLiteStore

!!! note
    Requires the `sqlite` extra: `uv pip install "emotional-memory[sqlite]"`

::: emotional_memory.stores.sqlite.SQLiteStore

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
