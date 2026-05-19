# State Stores

State stores persist the runtime `AffectiveState` snapshot across sessions.
Plug one in via `EmotionalMemoryConfig.state_store` (or the `state_store=` parameter).

The `AffectiveStateStore` protocol (defined in `interfaces.py`) requires three methods:
`save(state)`, `load() → AffectiveState | None`, and `clear()`.

See also: [State](state.md) for the `AffectiveState` data model,
and [Interfaces](interfaces.md) for the protocol definition.

## In-Memory (default)

::: emotional_memory.state_stores.in_memory.InMemoryAffectiveStateStore

## SQLite (persistent)

::: emotional_memory.state_stores.sqlite.SQLiteAffectiveStateStore

## Redis (distributed)

Requires the `[redis]` extra: `pip install 'emotional-memory[redis]'`

::: emotional_memory.state_stores.redis.RedisAffectiveStateStore
