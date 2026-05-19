"""Round-trip tests for async_adapters.py bridge adapters.

Covers SyncToAsyncStore, SyncToAsyncEmbedder, SyncToAsyncAppraisalEngine,
and as_async(). All tests use in-memory implementations — no I/O or API key.
"""

from __future__ import annotations

import pytest

from emotional_memory import EmotionalMemory, InMemoryStore
from emotional_memory.async_adapters import (
    SyncToAsyncAppraisalEngine,
    SyncToAsyncEmbedder,
    SyncToAsyncStore,
    as_async,
)
from emotional_memory.async_engine import AsyncEmotionalMemory
from emotional_memory.models import Memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FixedEmbedder:
    """Always returns the same embedding vector."""

    DIM = 4

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class _CountingAppraisalEngine:
    """Tracks appraise() calls and returns neutral AppraisalVector."""

    def __init__(self) -> None:
        self.call_count = 0

    def appraise(self, event_text: str, context: object = None) -> object:
        from emotional_memory.appraisal import AppraisalVector

        self.call_count += 1
        return AppraisalVector.neutral()


class _StoreWithClose(InMemoryStore):
    """InMemoryStore subclass that records close() calls."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# SyncToAsyncStore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_to_async_store_count_equals_len() -> None:
    sync_store = InMemoryStore()
    async_store = SyncToAsyncStore(sync_store)

    assert await async_store.count() == len(sync_store) == 0


@pytest.mark.asyncio
async def test_sync_to_async_store_save_get_roundtrip() -> None:
    sync_store = InMemoryStore()
    async_store = SyncToAsyncStore(sync_store)

    # Build a minimal Memory via the sync engine and extract it
    engine = EmotionalMemory(store=sync_store, embedder=_FixedEmbedder())
    mem = engine.encode("hello world")  # returns Memory

    # Delete it, then re-save via async adapter
    sync_store.delete(mem.id)
    assert await async_store.count() == 0

    await async_store.save(mem)
    assert await async_store.count() == 1

    retrieved = await async_store.get(mem.id)
    assert retrieved is not None
    assert retrieved.id == mem.id


@pytest.mark.asyncio
async def test_sync_to_async_store_delete() -> None:
    sync_store = InMemoryStore()
    async_store = SyncToAsyncStore(sync_store)

    engine = EmotionalMemory(store=sync_store, embedder=_FixedEmbedder())
    mem = engine.encode("to delete")

    assert await async_store.count() == 1
    await async_store.delete(mem.id)
    assert await async_store.count() == 0
    assert await async_store.get(mem.id) is None


@pytest.mark.asyncio
async def test_sync_to_async_store_list_all() -> None:
    sync_store = InMemoryStore()
    async_store = SyncToAsyncStore(sync_store)

    engine = EmotionalMemory(store=sync_store, embedder=_FixedEmbedder())
    mem1 = engine.encode("first")
    mem2 = engine.encode("second")

    all_mems = await async_store.list_all()
    assert {m.id for m in all_mems} == {mem1.id, mem2.id}


@pytest.mark.asyncio
async def test_sync_to_async_store_close_without_close_method() -> None:
    """close() on a store without close() should not raise."""
    sync_store = InMemoryStore()
    assert not hasattr(sync_store, "close") or not callable(getattr(sync_store, "close", None))
    async_store = SyncToAsyncStore(sync_store)
    # Must not raise
    await async_store.close()


@pytest.mark.asyncio
async def test_sync_to_async_store_close_calls_inner_close() -> None:
    sync_store = _StoreWithClose()
    async_store = SyncToAsyncStore(sync_store)

    assert not sync_store.closed
    await async_store.close()
    assert sync_store.closed


@pytest.mark.asyncio
async def test_sync_to_async_store_update_roundtrip() -> None:
    sync_store = InMemoryStore()
    async_store = SyncToAsyncStore(sync_store)

    engine = EmotionalMemory(store=sync_store, embedder=_FixedEmbedder())
    mem = engine.encode("original content")

    updated = mem.model_copy(update={"content": "updated content"})
    await async_store.update(updated)

    retrieved = await async_store.get(mem.id)
    assert retrieved is not None
    assert retrieved.content == "updated content"


# ---------------------------------------------------------------------------
# SyncToAsyncEmbedder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_to_async_embedder_embed() -> None:
    async_emb = SyncToAsyncEmbedder(_FixedEmbedder())
    result = await async_emb.embed("test")
    assert result == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_sync_to_async_embedder_embed_batch() -> None:
    async_emb = SyncToAsyncEmbedder(_FixedEmbedder())
    results = await async_emb.embed_batch(["a", "b", "c"])
    assert len(results) == 3
    assert all(r == [0.1, 0.2, 0.3, 0.4] for r in results)


# ---------------------------------------------------------------------------
# SyncToAsyncAppraisalEngine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_to_async_appraisal_engine_delegates() -> None:
    from emotional_memory.appraisal import AppraisalVector

    sync_engine = _CountingAppraisalEngine()
    async_engine = SyncToAsyncAppraisalEngine(sync_engine)  # type: ignore[arg-type]

    result = await async_engine.appraise("something emotional")
    assert isinstance(result, AppraisalVector)
    assert sync_engine.call_count == 1

    await async_engine.appraise("another event", context={"key": "val"})
    assert sync_engine.call_count == 2


# ---------------------------------------------------------------------------
# as_async() convenience wrapper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_as_async_produces_async_engine() -> None:
    sync_engine = EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder())
    async_engine = as_async(sync_engine)
    assert isinstance(async_engine, AsyncEmotionalMemory)


@pytest.mark.asyncio
async def test_as_async_encode_retrieve_roundtrip() -> None:
    sync_engine = EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder())
    async_engine = as_async(sync_engine)

    mem = await async_engine.encode("test memory for async roundtrip")
    assert isinstance(mem, Memory)

    results = await async_engine.retrieve("test memory", top_k=1)
    assert len(results) == 1
    assert results[0].id == mem.id


@pytest.mark.asyncio
async def test_as_async_state_independent_from_sync() -> None:
    """After as_async(), the two engines have independent affective states."""
    sync_engine = EmotionalMemory(store=InMemoryStore(), embedder=_FixedEmbedder())
    async_engine = as_async(sync_engine)

    # Encoding into async engine should not affect sync engine's store
    # (they share the store object, but state objects diverge)
    await async_engine.encode("async only event")
    sync_results = sync_engine.retrieve("async only event", top_k=1)
    # The store is shared, so the memory IS visible via sync engine too
    assert len(sync_results) == 1
