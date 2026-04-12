"""InMemoryStore — dict-backed MemoryStore with brute-force cosine search."""

from __future__ import annotations

import numpy as np

from emotional_memory.models import Memory


class InMemoryStore:
    """Simple in-memory MemoryStore. Not persistent across restarts.

    search_by_embedding uses brute-force cosine similarity — suitable for
    small datasets and testing. For production, use a vector database.
    """

    __slots__ = ("_store",)

    def __init__(self) -> None:
        self._store: dict[str, Memory] = {}

    def save(self, memory: Memory) -> None:
        self._store[memory.id] = memory

    def get(self, memory_id: str) -> Memory | None:
        return self._store.get(memory_id)

    def update(self, memory: Memory) -> None:
        self._store[memory.id] = memory

    def delete(self, memory_id: str) -> None:
        self._store.pop(memory_id, None)

    def list_all(self) -> list[Memory]:
        return list(self._store.values())

    def search_by_embedding(self, embedding: list[float], top_k: int) -> list[Memory]:
        """Return top_k memories by cosine similarity to the query embedding.

        Memories without an embedding are skipped. Uses vectorized matrix
        multiplication to compute all cosine similarities in a single batch
        rather than calling cosine_similarity individually for each memory.
        """
        candidates = [m for m in self._store.values() if m.embedding is not None]
        if not candidates:
            return []

        query = np.asarray(embedding, dtype=np.float64)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return candidates[:top_k]

        # Stack all embeddings into a (n x d) matrix — single allocation
        matrix = np.asarray([m.embedding for m in candidates], dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1)  # shape (n,)

        # Avoid division by zero for zero-norm embeddings
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = (matrix @ query) / (norms * query_norm)
        scores = np.nan_to_num(scores, nan=0.0)

        n = len(candidates)
        k = min(top_k, n)
        # np.argpartition is O(n) for finding top-k; sort only the k winners
        top_indices = np.argpartition(scores, n - k)[n - k :]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [candidates[i] for i in top_indices]

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(count={len(self._store)})"
