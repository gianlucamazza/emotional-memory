"""InMemoryStore — dict-backed MemoryStore with brute-force cosine search."""

from __future__ import annotations

from emotional_memory._math import cosine_similarity
from emotional_memory.models import Memory


class InMemoryStore:
    """Simple in-memory MemoryStore. Not persistent across restarts.

    search_by_embedding uses brute-force cosine similarity — suitable for
    small datasets and testing. For production, use a vector database.
    """

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

        Memories without an embedding are skipped.
        """
        scored: list[tuple[float, Memory]] = []
        for memory in self._store.values():
            if memory.embedding is None:
                continue
            score = cosine_similarity(embedding, memory.embedding)
            scored.append((score, memory))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def __len__(self) -> int:
        return len(self._store)
