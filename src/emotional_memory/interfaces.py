"""Abstract interfaces: Embedder and MemoryStore.

Both are typing.Protocol — implementors do not import or subclass from
here. Duck typing is sufficient.

``SequentialEmbedder`` is a convenience base class for implementations
that only define ``embed()``.  It provides a default ``embed_batch()``
that calls ``embed()`` sequentially.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from emotional_memory.models import Memory


@runtime_checkable
class Embedder(Protocol):
    """Converts text to a dense vector representation."""

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class SequentialEmbedder:
    """Base class for Embedder implementations that lack native batching.

    Subclass this and implement ``embed()``; ``embed_batch()`` is provided
    automatically by calling ``embed()`` for each item in sequence.

    Example::

        class MyEmbedder(SequentialEmbedder):
            def embed(self, text: str) -> list[float]:
                return my_model.encode(text).tolist()
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@runtime_checkable
class MemoryStore(Protocol):
    """Persistent storage for Memory objects."""

    def save(self, memory: Memory) -> None: ...

    def get(self, memory_id: str) -> Memory | None: ...

    def update(self, memory: Memory) -> None: ...

    def delete(self, memory_id: str) -> None: ...

    def list_all(self) -> list[Memory]: ...

    def search_by_embedding(self, embedding: list[float], top_k: int) -> list[Memory]: ...

    def __len__(self) -> int: ...
