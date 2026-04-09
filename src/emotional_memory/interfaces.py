"""Abstract interfaces: Embedder and MemoryStore.

Both are typing.Protocol — implementors do not import or subclass from
here. Duck typing is sufficient.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from emotional_memory.models import Memory


@runtime_checkable
class Embedder(Protocol):
    """Converts text to a dense vector representation."""

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class MemoryStore(Protocol):
    """Persistent storage for Memory objects."""

    def save(self, memory: Memory) -> None: ...

    def get(self, memory_id: str) -> Memory | None: ...

    def update(self, memory: Memory) -> None: ...

    def delete(self, memory_id: str) -> None: ...

    def list_all(self) -> list[Memory]: ...

    def search_by_embedding(self, embedding: list[float], top_k: int) -> list[Memory]: ...
