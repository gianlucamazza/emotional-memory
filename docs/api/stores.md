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

!!! warning "Databases written before v0.18"
    The `sqlite-vec` index is created with `distance_metric=cosine` so that ANN
    candidate prefiltering matches the cosine similarity used by retrieval
    scoring. Earlier versions used sqlite-vec's L2 default, which ranks
    differently whenever embeddings are not L2-normalised. Opening such a
    database emits a `UserWarning`; call `store.rebuild_vector_index()` once to
    migrate it (the vector table is derived data — embeddings are also stored in
    the memory rows, so nothing is lost).

::: emotional_memory.stores.sqlite.SQLiteStore

### QdrantStore

!!! note
    Requires the `qdrant` extra: `uv pip install "emotional-memory[qdrant]"`

::: emotional_memory.stores.qdrant.QdrantStore

### ChromaStore

!!! note
    Requires the HTTP-only `chroma` extra and a separately operated server:
    `uv pip install "emotional-memory[chroma]"`. Pass `host=`; embedded and
    file-backed modes are intentionally unavailable.

::: emotional_memory.stores.chroma.ChromaStore

Runtime mood/affect persistence is a separate protocol (`AffectiveStateStore`),
not a `MemoryStore`. See [State Stores](state_stores.md).
