"""Async protocol variants for Embedder, MemoryStore, and AppraisalEngine.

These mirror the synchronous protocols in ``interfaces.py`` and ``appraisal.py``
but with ``async def`` methods, enabling native async implementations backed by
async database drivers (asyncpg, motor, aiosqlite) or async LLM SDKs.

Note: ``__len__`` cannot be an async method in Python, so ``AsyncMemoryStore``
exposes ``count()`` instead for the async size check.

Usage::

    from emotional_memory.interfaces_async import AsyncEmbedder, AsyncMemoryStore
    from emotional_memory.async_engine import AsyncEmotionalMemory
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from emotional_memory.appraisal import AppraisalVector
from emotional_memory.models import Memory


@runtime_checkable
class AsyncEmbedder(Protocol):
    """Async variant of the ``Embedder`` protocol."""

    async def embed(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class AsyncMemoryStore(Protocol):
    """Async variant of the ``MemoryStore`` protocol.

    ``count()`` replaces ``__len__`` because ``__len__`` cannot be declared
    ``async`` in Python.
    """

    async def save(self, memory: Memory) -> None: ...

    async def get(self, memory_id: str) -> Memory | None: ...

    async def update(self, memory: Memory) -> None: ...

    async def delete(self, memory_id: str) -> None: ...

    async def list_all(self) -> list[Memory]: ...

    async def search_by_embedding(self, embedding: list[float], top_k: int) -> list[Memory]: ...

    async def count(self) -> int: ...


@runtime_checkable
class AsyncAppraisalEngine(Protocol):
    """Async variant of the ``AppraisalEngine`` protocol."""

    async def appraise(
        self, event_text: str, context: dict[str, Any] | None = None
    ) -> AppraisalVector: ...
