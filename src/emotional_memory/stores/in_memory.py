"""InMemoryStore — dict-backed MemoryStore with brute-force cosine search."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from emotional_memory.models import Memory


class InMemoryStore:
    """Simple in-memory MemoryStore. Not persistent across restarts.

    ``search_by_embedding`` uses brute-force cosine similarity with a cached
    embedding matrix rebuilt lazily after mutations — suitable for small
    datasets and testing. For production at larger scale, use a vector
    database (``SQLiteStore``, ``QdrantStore``, ``ChromaStore``).
    """

    __slots__ = ("_dirty", "_ids", "_matrix", "_norms", "_store")

    def __init__(self) -> None:
        self._store: dict[str, Memory] = {}
        self._ids: list[str] = []
        self._matrix: NDArray[np.float64] | None = None
        self._norms: NDArray[np.float64] | None = None
        self._dirty: bool = True

    def save(self, memory: Memory) -> None:
        self._store[memory.id] = memory
        self._dirty = True

    def get(self, memory_id: str) -> Memory | None:
        return self._store.get(memory_id)

    def update(self, memory: Memory) -> None:
        self._store[memory.id] = memory
        self._dirty = True

    def delete(self, memory_id: str) -> None:
        self._store.pop(memory_id, None)
        self._dirty = True

    def list_all(self) -> list[Memory]:
        return list(self._store.values())

    def _rebuild_cache(self) -> None:
        """Rebuild the embedding matrix cache from current store contents."""
        candidates = [m for m in self._store.values() if m.embedding is not None]
        if not candidates:
            self._ids = []
            self._matrix = None
            self._norms = None
            self._dirty = False
            return

        self._ids = [m.id for m in candidates]
        self._matrix = np.asarray([m.embedding for m in candidates], dtype=np.float64)
        self._norms = np.linalg.norm(self._matrix, axis=1)
        self._dirty = False

    def search_by_embedding(self, embedding: list[float], top_k: int) -> list[Memory]:
        """Return top_k memories by cosine similarity to the query embedding.

        Memories without an embedding are skipped. Uses a cached embedding
        matrix (rebuilt lazily after save/update/delete) so repeated searches
        avoid re-stacking all vectors on every call.
        """
        if self._dirty:
            self._rebuild_cache()

        if not self._ids or self._matrix is None or self._norms is None:
            return []

        query = np.asarray(embedding, dtype=np.float64)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            # Preserve previous behaviour: first top_k embeddable memories
            return [self._store[mid] for mid in self._ids[:top_k]]

        with np.errstate(invalid="ignore", divide="ignore"):
            scores = (self._matrix @ query) / (self._norms * query_norm)
        scores = np.nan_to_num(scores, nan=0.0)

        n = len(self._ids)
        k = min(top_k, n)
        # np.argpartition is O(n) for finding top-k; sort only the k winners
        top_indices = np.argpartition(scores, n - k)[n - k :]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self._store[self._ids[i]] for i in top_indices]

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(count={len(self._store)})"
