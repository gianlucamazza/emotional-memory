# State Stores

State stores persist the runtime `AffectiveState` snapshot across sessions.
Pass one as the `state_store=` constructor argument of `EmotionalMemory` or
`AsyncEmotionalMemory` — it is **not** a field on `EmotionalMemoryConfig`.

The [`AffectiveStateStore`](interfaces.md) protocol (defined in `interfaces.py`)
requires three methods: `save(state)`, `load() → AffectiveState | None`, and
`clear()`.

`SQLiteAffectiveStateStore` uses the stdlib `sqlite3` module (no extra).
`RedisAffectiveStateStore` requires the `[redis]` extra.

See also: [State](state.md) for the `AffectiveState` data model, and
[Stores](stores.md) for `MemoryStore` backends.

## In-Memory (default)

::: emotional_memory.state_stores.in_memory.InMemoryAffectiveStateStore

## SQLite (persistent)

::: emotional_memory.state_stores.sqlite.SQLiteAffectiveStateStore

## Redis (distributed)

Requires the `[redis]` extra: `pip install 'emotional-memory[redis]'`

::: emotional_memory.state_stores.redis.RedisAffectiveStateStore
