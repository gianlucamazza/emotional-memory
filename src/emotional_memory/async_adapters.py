"""Bridge adapters: wrap sync implementations for use with AsyncEmotionalMemory.

Each adapter uses ``asyncio.to_thread`` to offload blocking calls to a thread
pool, keeping the event loop free.  The CPU-bound scoring logic inside the
engine itself is synchronous and runs inline — only the I/O boundaries
(embed, store, appraise) are bridged.

Usage::

    from emotional_memory import EmotionalMemory
    from emotional_memory.async_adapters import as_async

    sync_engine = EmotionalMemory(store, embedder)
    async_engine = as_async(sync_engine)

    # or build manually:
    from emotional_memory.async_adapters import SyncToAsyncEmbedder, SyncToAsyncStore
    from emotional_memory.async_engine import AsyncEmotionalMemory

    async_engine = AsyncEmotionalMemory(
        store=SyncToAsyncStore(sync_store),
        embedder=SyncToAsyncEmbedder(sync_embedder),
    )
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from emotional_memory.appraisal import AppraisalVector, GenericAppraisalVector
from emotional_memory.interfaces import Embedder, MemoryStore
from emotional_memory.models import Memory

if TYPE_CHECKING:
    from emotional_memory.appraisal import AppraisalEngine
    from emotional_memory.async_engine import AsyncEmotionalMemory
    from emotional_memory.engine import EmotionalMemory


class SyncToAsyncEmbedder:
    """Wraps a synchronous ``Embedder`` for use with ``AsyncEmotionalMemory``."""

    __slots__ = ("_inner",)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(inner={self._inner!r})"

    def __init__(self, sync_embedder: Embedder) -> None:
        self._inner = sync_embedder

    async def embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._inner.embed, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._inner.embed_batch, texts)


class SyncToAsyncStore:
    """Wraps a synchronous ``MemoryStore`` for use with ``AsyncEmotionalMemory``."""

    __slots__ = ("_inner",)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(inner={self._inner!r})"

    def __init__(self, sync_store: MemoryStore) -> None:
        self._inner = sync_store

    async def save(self, memory: Memory) -> None:
        await asyncio.to_thread(self._inner.save, memory)

    async def get(self, memory_id: str) -> Memory | None:
        return await asyncio.to_thread(self._inner.get, memory_id)

    async def update(self, memory: Memory) -> None:
        await asyncio.to_thread(self._inner.update, memory)

    async def delete(self, memory_id: str) -> None:
        await asyncio.to_thread(self._inner.delete, memory_id)

    async def list_all(self) -> list[Memory]:
        return await asyncio.to_thread(self._inner.list_all)

    async def search_by_embedding(self, embedding: list[float], top_k: int) -> list[Memory]:
        return await asyncio.to_thread(self._inner.search_by_embedding, embedding, top_k)

    async def count(self) -> int:
        return await asyncio.to_thread(len, self._inner)

    async def close(self) -> None:
        """Proxy ``close()`` to the wrapped store if it supports it."""
        close = getattr(self._inner, "close", None)
        if callable(close):
            await asyncio.to_thread(close)


class SyncToAsyncAppraisalEngine:
    """Wraps a synchronous ``AppraisalEngine`` for use with ``AsyncEmotionalMemory``."""

    __slots__ = ("_inner",)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(inner={self._inner!r})"

    def __init__(self, sync_engine: AppraisalEngine) -> None:
        self._inner = sync_engine

    async def appraise(
        self, event_text: str, context: dict[str, Any] | None = None
    ) -> AppraisalVector | GenericAppraisalVector:
        return await asyncio.to_thread(self._inner.appraise, event_text, context)


def as_async(engine: EmotionalMemory) -> AsyncEmotionalMemory:
    """Wrap a synchronous ``EmotionalMemory`` into an async one.

    All I/O calls (embed, store, appraise) are delegated via
    ``asyncio.to_thread``.  The engine's affective state is **copied** at
    wrap time — the two engines are independent after this point and their
    states diverge as each processes new events.  The underlying store and
    embedder are shared (same object references), so concurrent writes from
    both engines will interleave.
    """
    from emotional_memory.async_engine import AsyncEmotionalMemory

    appraisal_async = (
        SyncToAsyncAppraisalEngine(engine._appraisal_engine)
        if engine._appraisal_engine is not None
        else None
    )
    async_engine = AsyncEmotionalMemory(
        store=SyncToAsyncStore(engine._store),
        embedder=SyncToAsyncEmbedder(engine._embedder),
        appraisal_engine=appraisal_async,
        config=engine._config,
        state_store=engine._state_store,
    )
    # Share the current AffectiveState reference as the async engine's starting
    # point.  AffectiveState is always *replaced* (never mutated) on every
    # update, so both engines immediately diverge once either processes an event.
    # There is no aliasing hazard because the object is effectively immutable.
    async_engine._state = engine._state
    return async_engine
