"""Tests for AsyncEmotionalMemory and async adapters."""

import asyncio

import pytest
from conftest import DeterministicEmbedder, FixedEmbedder

from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal import AppraisalVector, StaticAppraisalEngine
from emotional_memory.async_adapters import (
    SyncToAsyncAppraisalEngine,
    SyncToAsyncEmbedder,
    SyncToAsyncStore,
    as_async,
)
from emotional_memory.async_engine import AsyncEmotionalMemory
from emotional_memory.engine import EmotionalMemory, EmotionalMemoryConfig
from emotional_memory.stimmung import StimmungDecayConfig
from emotional_memory.stores.in_memory import InMemoryStore

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sync_store() -> InMemoryStore:
    return InMemoryStore()


def _async_engine(embedder=None, appraisal_engine=None, config=None) -> AsyncEmotionalMemory:
    return AsyncEmotionalMemory(
        store=SyncToAsyncStore(_sync_store()),
        embedder=SyncToAsyncEmbedder(embedder or FixedEmbedder([1.0, 0.0])),
        appraisal_engine=appraisal_engine,
        config=config,
    )


# ---------------------------------------------------------------------------
# AsyncEmotionalMemory — encode
# ---------------------------------------------------------------------------


class TestAsyncEncode:
    async def test_returns_memory(self):
        em = _async_engine()
        m = await em.encode("hello world")
        assert m.content == "hello world"

    async def test_memory_has_embedding(self):
        em = _async_engine()
        m = await em.encode("test")
        assert m.embedding is not None

    async def test_memory_stored(self):
        store = _sync_store()
        em = AsyncEmotionalMemory(
            store=SyncToAsyncStore(store),
            embedder=SyncToAsyncEmbedder(FixedEmbedder([1.0, 0.0])),
        )
        m = await em.encode("stored?")
        assert store.get(m.id) is not None

    async def test_appraisal_engine_used(self):
        vector = AppraisalVector(
            novelty=0.9,
            goal_relevance=0.8,
            coping_potential=0.7,
            norm_congruence=0.6,
            self_relevance=0.5,
        )
        appraisal = SyncToAsyncAppraisalEngine(StaticAppraisalEngine(vector))
        em = _async_engine(appraisal_engine=appraisal)
        m = await em.encode("test")
        assert m.tag.appraisal is not None
        assert m.tag.appraisal.novelty == pytest.approx(0.9)

    async def test_state_updates_after_encode(self):
        em = _async_engine()
        initial = em.get_state()
        await em.encode("something happened")
        # State should have moved (stimmung updated)
        assert em.get_state() is not initial

    async def test_metadata_passed_as_context(self):
        captured: list[dict] = []

        class TrackingEngine:
            def appraise(self, text: str, context=None) -> AppraisalVector:
                captured.append(context or {})
                return AppraisalVector.neutral()

        class AsyncWrapper:
            def __init__(self, inner):
                self._inner = inner

            async def appraise(self, text: str, context=None) -> AppraisalVector:
                return await asyncio.to_thread(self._inner.appraise, text, context)

        em = _async_engine(appraisal_engine=AsyncWrapper(TrackingEngine()))
        await em.encode("test", metadata={"key": "val"})
        assert captured[0] == {"key": "val"}


# ---------------------------------------------------------------------------
# AsyncEmotionalMemory — retrieve
# ---------------------------------------------------------------------------


class TestAsyncRetrieve:
    async def test_retrieve_empty_returns_empty(self):
        em = _async_engine()
        results = await em.retrieve("query")
        assert results == []

    async def test_retrieve_returns_encoded_memory(self):
        em = _async_engine(embedder=DeterministicEmbedder())
        await em.encode("the quick brown fox")
        results = await em.retrieve("quick brown fox")
        assert len(results) == 1
        assert results[0].content == "the quick brown fox"

    async def test_retrieve_top_k_respected(self):
        em = _async_engine(embedder=DeterministicEmbedder())
        for i in range(5):
            await em.encode(f"memory {i}")
        results = await em.retrieve("memory", top_k=2)
        assert len(results) <= 2

    async def test_retrieval_count_incremented(self):
        em = _async_engine(embedder=DeterministicEmbedder())
        await em.encode("something")
        results = await em.retrieve("something")
        assert results[0].tag.retrieval_count == 1


# ---------------------------------------------------------------------------
# AsyncEmotionalMemory — encode_batch
# ---------------------------------------------------------------------------


class TestAsyncEncodeBatch:
    async def test_batch_returns_all(self):
        em = _async_engine()
        contents = ["a", "b", "c"]
        results = await em.encode_batch(contents)
        assert len(results) == 3
        assert {r.content for r in results} == set(contents)


# ---------------------------------------------------------------------------
# AsyncEmotionalMemory — delete
# ---------------------------------------------------------------------------


class TestAsyncDelete:
    async def test_delete_removes_memory(self):
        store = _sync_store()
        em = AsyncEmotionalMemory(
            store=SyncToAsyncStore(store),
            embedder=SyncToAsyncEmbedder(FixedEmbedder([1.0, 0.0])),
        )
        m = await em.encode("to delete")
        await em.delete(m.id)
        assert store.get(m.id) is None


# ---------------------------------------------------------------------------
# AsyncEmotionalMemory — state persistence
# ---------------------------------------------------------------------------


class TestAsyncStatePersistence:
    async def test_save_and_load_state(self):
        em = _async_engine()
        await em.encode("encode something to build state")
        snapshot = em.save_state()
        assert "_history" in snapshot

        em2 = _async_engine()
        em2.load_state(snapshot)
        assert em2.get_state().core_affect == em.get_state().core_affect
        assert em2.get_state().stimmung.valence == pytest.approx(
            em.get_state().stimmung.valence, abs=1e-9
        )

    async def test_momentum_preserved_after_restore(self):
        em = _async_engine()
        for v in [0.2, 0.5, 0.8]:
            em.set_affect(CoreAffect(valence=v, arousal=0.5))
        snapshot = em.save_state()

        em2 = _async_engine()
        em2.load_state(snapshot)
        next1 = em.get_state().update(CoreAffect(valence=0.9, arousal=0.5))
        next2 = em2.get_state().update(CoreAffect(valence=0.9, arousal=0.5))
        assert next1.momentum.d_valence == pytest.approx(next2.momentum.d_valence, abs=1e-6)


# ---------------------------------------------------------------------------
# AsyncEmotionalMemory — get_current_stimmung
# ---------------------------------------------------------------------------


class TestAsyncGetCurrentStimmung:
    async def test_returns_stimmung_without_decay(self):
        em = _async_engine()
        s = em.get_current_stimmung()
        assert s.valence == pytest.approx(0.0)

    async def test_returns_regressed_stimmung_with_decay(self):
        from datetime import UTC, datetime, timedelta

        cfg = EmotionalMemoryConfig(stimmung_decay=StimmungDecayConfig(base_half_life_seconds=1.0))
        em = _async_engine(config=cfg)
        em.set_affect(CoreAffect(valence=1.0, arousal=0.8))
        future = datetime.now(tz=UTC) + timedelta(hours=1)
        s = em.get_current_stimmung(now=future)
        assert s.valence < em.get_state().stimmung.valence


# ---------------------------------------------------------------------------
# SyncToAsync adapters
# ---------------------------------------------------------------------------


class TestSyncToAsyncAdapters:
    async def test_sync_to_async_embedder(self):
        adapter = SyncToAsyncEmbedder(FixedEmbedder([0.5, 0.5]))
        result = await adapter.embed("test")
        assert result == [0.5, 0.5]

    async def test_sync_to_async_embedder_batch(self):
        adapter = SyncToAsyncEmbedder(FixedEmbedder([1.0, 0.0]))
        results = await adapter.embed_batch(["a", "b"])
        assert len(results) == 2

    async def test_sync_to_async_store_save_get(self):
        from conftest import make_test_memory

        store = _sync_store()
        adapter = SyncToAsyncStore(store)
        m = make_test_memory("hello")
        await adapter.save(m)
        got = await adapter.get(m.id)
        assert got is not None
        assert got.content == "hello"

    async def test_sync_to_async_store_count(self):
        from conftest import make_test_memory

        store = _sync_store()
        adapter = SyncToAsyncStore(store)
        assert await adapter.count() == 0
        await adapter.save(make_test_memory("a"))
        assert await adapter.count() == 1

    async def test_sync_to_async_store_delete(self):
        from conftest import make_test_memory

        store = _sync_store()
        adapter = SyncToAsyncStore(store)
        m = make_test_memory("to delete")
        await adapter.save(m)
        await adapter.delete(m.id)
        assert await adapter.get(m.id) is None

    async def test_sync_to_async_store_list_all(self):
        from conftest import make_test_memory

        store = _sync_store()
        adapter = SyncToAsyncStore(store)
        await adapter.save(make_test_memory("a"))
        await adapter.save(make_test_memory("b"))
        all_mems = await adapter.list_all()
        assert len(all_mems) == 2

    async def test_sync_to_async_appraisal_engine(self):
        vector = AppraisalVector.neutral()
        adapter = SyncToAsyncAppraisalEngine(StaticAppraisalEngine(vector))
        result = await adapter.appraise("test event")
        assert result == vector


# ---------------------------------------------------------------------------
# as_async factory
# ---------------------------------------------------------------------------


class TestAsAsync:
    async def test_as_async_wraps_engine(self):
        store = _sync_store()
        sync_em = EmotionalMemory(store=store, embedder=FixedEmbedder([1.0, 0.0]))
        async_em = as_async(sync_em)
        m = await async_em.encode("wrapped")
        assert m.content == "wrapped"

    async def test_as_async_shares_state(self):
        store = _sync_store()
        sync_em = EmotionalMemory(store=store, embedder=FixedEmbedder([1.0, 0.0]))
        sync_em.set_affect(CoreAffect(valence=0.9, arousal=0.7))
        async_em = as_async(sync_em)
        assert async_em.get_state().core_affect.valence == pytest.approx(0.9)
