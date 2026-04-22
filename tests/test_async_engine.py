"""Tests for AsyncEmotionalMemory and async adapters."""

import asyncio
import json
import warnings
from datetime import UTC, datetime, timedelta

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
from emotional_memory.mood import MoodDecayConfig
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
        # State should have moved (mood updated)
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

    async def test_observe_updates_state_without_storing_memory(self):
        vector = AppraisalVector(
            novelty=1.0,
            goal_relevance=1.0,
            coping_potential=1.0,
            norm_congruence=1.0,
            self_relevance=1.0,
        )
        appraisal = SyncToAsyncAppraisalEngine(StaticAppraisalEngine(vector))
        em = _async_engine(appraisal_engine=appraisal)
        initial = em.get_state()

        tag = await em.observe("assistant reassurance")

        assert tag.appraisal == vector
        assert await em.count() == 0
        assert em.get_state() != initial


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

    async def test_retrieve_with_explanations_exposes_score_breakdown(self):
        class IndexEmbedder:
            def __init__(self) -> None:
                self._map = {
                    "query": [1.0, 0.0, 0.0, 0.0],
                    "relevant": [1.0, 0.0, 0.0, 0.0],
                    "other": [0.0, 1.0, 0.0, 0.0],
                }

            def embed(self, text: str) -> list[float]:
                return self._map.get(text, [0.0, 0.0, 0.0, 0.0])

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [self.embed(text) for text in texts]

        em = _async_engine(embedder=IndexEmbedder())
        await em.encode("relevant")
        await em.encode("other")

        explanations = await em.retrieve_with_explanations("query", top_k=1)

        assert len(explanations) == 1
        explanation = explanations[0]
        assert explanation.memory.content == "relevant"
        assert explanation.memory.tag.retrieval_count == 1
        assert explanation.selected_as_seed is True
        assert explanation.pass1_rank == 1
        assert explanation.pass2_rank == 1
        assert explanation.candidate_count == 2
        assert explanation.breakdown.weights.total() == pytest.approx(1.0)
        assert explanation.breakdown.total_score == pytest.approx(explanation.score)
        assert explanation.breakdown.weighted_signals.total() == pytest.approx(explanation.score)


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
        assert em2.get_state().mood.valence == pytest.approx(em.get_state().mood.valence, abs=1e-9)

    async def test_momentum_preserved_after_restore(self):
        em = _async_engine()
        for v in [0.2, 0.5, 0.8]:
            em.set_affect(CoreAffect(valence=v, arousal=0.5))
        snapshot = em.save_state()

        em2 = _async_engine()
        em2.load_state(snapshot)
        # Pin `now` so both update() calls use the same timestamp and the
        # momentum comparison is deterministic regardless of test ordering.
        now = datetime.now(UTC)
        next1 = em.get_state().update(CoreAffect(valence=0.9, arousal=0.5), now=now)
        next2 = em2.get_state().update(CoreAffect(valence=0.9, arousal=0.5), now=now)
        assert next1.momentum.d_valence == pytest.approx(next2.momentum.d_valence, abs=1e-6)

    async def test_reset_state_restores_initial_baseline(self):
        em = _async_engine()
        em.set_affect(CoreAffect(valence=0.8, arousal=0.7))
        baseline = _async_engine().get_state()
        assert em.get_state().core_affect != baseline.core_affect

        em.reset_state()

        reset = em.get_state()
        assert reset.core_affect == baseline.core_affect
        assert reset.mood.valence == baseline.mood.valence
        assert reset.mood.arousal == baseline.mood.arousal
        assert reset.mood.dominance == baseline.mood.dominance
        assert reset.mood.inertia == baseline.mood.inertia
        assert reset.momentum == baseline.momentum


# ---------------------------------------------------------------------------
# AsyncEmotionalMemory — get_current_mood
# ---------------------------------------------------------------------------


class TestAsyncGetCurrentMood:
    async def test_returns_mood_without_decay(self):
        em = _async_engine()
        s = em.get_current_mood()
        assert s.valence == pytest.approx(0.0)

    async def test_returns_regressed_mood_with_decay(self):
        from datetime import UTC, datetime, timedelta

        cfg = EmotionalMemoryConfig(mood_decay=MoodDecayConfig(base_half_life_seconds=1.0))
        em = _async_engine(config=cfg)
        em.set_affect(CoreAffect(valence=1.0, arousal=0.8))
        future = datetime.now(tz=UTC) + timedelta(hours=1)
        s = em.get_current_mood(now=future)
        assert s.valence < em.get_state().mood.valence


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

    async def test_sync_to_async_store_update(self):
        from conftest import make_test_memory

        store = _sync_store()
        adapter = SyncToAsyncStore(store)
        m = make_test_memory("original")
        await adapter.save(m)
        m_updated = m.model_copy(update={"content": "updated"})
        await adapter.update(m_updated)
        got = await adapter.get(m.id)
        assert got is not None
        assert got.content == "updated"

    async def test_sync_to_async_store_search_by_embedding(self):
        from conftest import make_test_memory

        store = _sync_store()
        adapter = SyncToAsyncStore(store)
        m = make_test_memory("searchable", embedding=[1.0, 0.0])
        await adapter.save(m)
        results = await adapter.search_by_embedding([1.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].id == m.id

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


class TestAsyncFacadeMethods:
    async def test_get_returns_encoded_memory(self):
        em = _async_engine()
        m = await em.encode("findable")
        result = await em.get(m.id)
        assert result is not None
        assert result.content == "findable"

    async def test_get_missing_returns_none(self):
        em = _async_engine()
        assert await em.get("nonexistent-id") is None

    async def test_list_all_empty(self):
        em = _async_engine()
        assert await em.list_all() == []

    async def test_list_all_returns_all(self):
        em = _async_engine()
        m1 = await em.encode("one")
        m2 = await em.encode("two")
        ids = {m.id for m in await em.list_all()}
        assert m1.id in ids
        assert m2.id in ids

    async def test_count_empty(self):
        em = _async_engine()
        assert await em.count() == 0

    async def test_count_after_encode(self):
        em = _async_engine()
        await em.encode("a")
        await em.encode("b")
        assert await em.count() == 2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestAsyncInputValidation:
    async def test_encode_batch_metadata_length_mismatch(self):
        em = _async_engine()
        with pytest.raises(ValueError, match="metadata length"):
            await em.encode_batch(["a", "b"], metadata=[{"x": 1}])

    async def test_retrieve_top_k_zero_raises(self):
        em = _async_engine()
        await em.encode("something")
        with pytest.raises(ValueError, match="top_k"):
            await em.retrieve("query", top_k=0)

    async def test_retrieve_top_k_negative_raises(self):
        em = _async_engine()
        with pytest.raises(ValueError, match="top_k"):
            await em.retrieve("query", top_k=-1)


# ---------------------------------------------------------------------------
# Helpers for close / context-manager tests
# ---------------------------------------------------------------------------


class _AsyncCloseableStore(InMemoryStore):
    """InMemoryStore extended with close() that records the call."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _async_engine_with_closeable() -> tuple[AsyncEmotionalMemory, _AsyncCloseableStore]:
    store = _AsyncCloseableStore()
    em = AsyncEmotionalMemory(
        store=SyncToAsyncStore(store),
        embedder=SyncToAsyncEmbedder(FixedEmbedder([1.0, 0.0])),
    )
    return em, store


def _old_timestamp() -> datetime:
    return datetime.now(tz=UTC) - timedelta(days=365)


# ---------------------------------------------------------------------------
# Async prune()
# ---------------------------------------------------------------------------


class TestAsyncPrune:
    async def test_prune_removes_old_memories(self) -> None:
        store = _sync_store()
        em = AsyncEmotionalMemory(
            store=SyncToAsyncStore(store),
            embedder=SyncToAsyncEmbedder(FixedEmbedder([1.0, 0.0])),
        )
        mem = await em.encode("old")
        aged = mem.model_copy(
            update={"tag": mem.tag.model_copy(update={"timestamp": _old_timestamp()})}
        )
        store.update(aged)
        removed = await em.prune(threshold=0.05)
        assert removed == 1
        assert await em.count() == 0

    async def test_prune_keeps_fresh_memories(self) -> None:
        em = _async_engine()
        em.set_affect(CoreAffect(valence=0.5, arousal=0.8))  # high arousal → non-zero strength
        await em.encode("fresh")
        removed = await em.prune(threshold=0.05)
        assert removed == 0
        assert await em.count() == 1

    async def test_prune_returns_count(self) -> None:
        store = _sync_store()
        em = AsyncEmotionalMemory(
            store=SyncToAsyncStore(store),
            embedder=SyncToAsyncEmbedder(FixedEmbedder([1.0, 0.0])),
        )
        # arousal=0.5: gives non-zero initial strength but stays below the floor
        # threshold (0.7), so backdated memories decay below prune threshold=0.05
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mems = [await em.encode(f"mem{i}") for i in range(3)]
        for m in mems[:2]:
            aged = m.model_copy(
                update={"tag": m.tag.model_copy(update={"timestamp": _old_timestamp()})}
            )
            store.update(aged)
        removed = await em.prune(threshold=0.05)
        assert removed == 2
        assert await em.count() == 1

    async def test_prune_empty_store(self) -> None:
        em = _async_engine()
        assert await em.prune() == 0


# ---------------------------------------------------------------------------
# Async export_memories() / import_memories()
# ---------------------------------------------------------------------------


class TestAsyncExportImport:
    async def test_export_returns_list_of_dicts(self) -> None:
        em = _async_engine()
        await em.encode("first")
        await em.encode("second")
        data = await em.export_memories()
        assert isinstance(data, list)
        assert len(data) == 2
        assert all(isinstance(d, dict) for d in data)

    async def test_export_is_json_serialisable(self) -> None:
        em = _async_engine()
        await em.encode("content")
        json.dumps(await em.export_memories())  # must not raise

    async def test_import_restores_to_new_engine(self) -> None:
        em1 = _async_engine()
        await em1.encode("alpha")
        await em1.encode("beta")
        data = await em1.export_memories()

        em2 = _async_engine()
        written = await em2.import_memories(data)
        assert written == 2
        contents = {m.content for m in await em2.list_all()}
        assert "alpha" in contents
        assert "beta" in contents

    async def test_import_skips_duplicates(self) -> None:
        em = _async_engine()
        await em.encode("unique")
        data = await em.export_memories()
        written = await em.import_memories(data)
        assert written == 0
        assert await em.count() == 1

    async def test_roundtrip_preserves_tags(self) -> None:
        em1 = _async_engine()
        m = await em1.encode("roundtrip")
        data = await em1.export_memories()

        em2 = _async_engine()
        await em2.import_memories(data)
        restored = (await em2.list_all())[0]
        assert restored.content == m.content
        assert restored.tag.core_affect.valence == pytest.approx(m.tag.core_affect.valence)


# ---------------------------------------------------------------------------
# Async close() and context manager
# ---------------------------------------------------------------------------


class TestAsyncCloseAndContextManager:
    async def test_close_calls_store_close(self) -> None:
        em, store = _async_engine_with_closeable()
        await em.close()
        assert store.closed

    async def test_close_safe_without_close_method(self) -> None:
        em = _async_engine()  # SyncToAsyncStore wrapping InMemoryStore (no close)
        await em.close()  # must not raise

    async def test_async_context_manager_calls_close(self) -> None:
        store = _AsyncCloseableStore()
        async with AsyncEmotionalMemory(
            store=SyncToAsyncStore(store),
            embedder=SyncToAsyncEmbedder(FixedEmbedder([1.0, 0.0])),
        ):
            pass
        assert store.closed

    async def test_async_context_manager_returns_engine(self) -> None:
        async with _async_engine() as em:
            assert isinstance(em, AsyncEmotionalMemory)


# ---------------------------------------------------------------------------
# Bidirectional resonance links (async)
# ---------------------------------------------------------------------------


class TestAsyncBidirectionalResonanceLinks:
    def _engine_with_low_threshold(self) -> AsyncEmotionalMemory:
        from emotional_memory.engine import EmotionalMemoryConfig
        from emotional_memory.resonance import ResonanceConfig

        config = EmotionalMemoryConfig(
            resonance=ResonanceConfig(threshold=0.0, temporal_half_life_seconds=1e9)
        )
        return AsyncEmotionalMemory(
            store=SyncToAsyncStore(InMemoryStore()),
            embedder=SyncToAsyncEmbedder(FixedEmbedder([1.0, 0.0])),
            config=config,
        )

    async def test_target_memory_has_backward_link(self) -> None:
        """After A is encoded after B, B must carry a backward link targeting A."""
        em = self._engine_with_low_threshold()
        b = await em.encode("first memory")
        a = await em.encode("second memory")

        b_updated = await em.get(b.id)
        assert b_updated is not None
        backward_ids = [lnk.target_id for lnk in b_updated.tag.resonance_links]
        assert a.id in backward_ids

    async def test_backward_link_strength_mirrors_forward(self) -> None:
        em = self._engine_with_low_threshold()
        b = await em.encode("older memory")
        a = await em.encode("newer memory")

        forward_strength = next(
            lnk.strength for lnk in a.tag.resonance_links if lnk.target_id == b.id
        )
        b_updated = await em.get(b.id)
        assert b_updated is not None
        backward_strength = next(
            lnk.strength for lnk in b_updated.tag.resonance_links if lnk.target_id == a.id
        )
        assert forward_strength == pytest.approx(backward_strength)

    async def test_encode_batch_creates_backward_links(self) -> None:
        em = self._engine_with_low_threshold()
        memories = await em.encode_batch(["first", "second", "third"])
        first = await em.get(memories[0].id)
        assert first is not None
        backward_targets = [lnk.target_id for lnk in first.tag.resonance_links]
        later_ids = {m.id for m in memories[1:]}
        assert any(tid in later_ids for tid in backward_targets)


# ---------------------------------------------------------------------------
# Spreading activation + Hebbian (async)
# ---------------------------------------------------------------------------


class TestAsyncSpreadingAndHebbian:
    def _engine_with_resonance(self) -> AsyncEmotionalMemory:
        from emotional_memory.engine import EmotionalMemoryConfig
        from emotional_memory.resonance import ResonanceConfig

        config = EmotionalMemoryConfig(
            resonance=ResonanceConfig(
                threshold=0.0,
                temporal_half_life_seconds=1e9,
                propagation_hops=2,
                hebbian_increment=0.1,
            )
        )
        return AsyncEmotionalMemory(
            store=SyncToAsyncStore(InMemoryStore()),
            embedder=SyncToAsyncEmbedder(FixedEmbedder([1.0, 0.0])),
            config=config,
        )

    async def test_retrieve_works_with_spreading(self) -> None:
        """Smoke test: async retrieve with spreading activation returns results."""
        em = self._engine_with_resonance()
        await em.encode("alpha memory")
        await em.encode("beta memory")
        await em.encode("gamma memory")
        results = await em.retrieve("memory", top_k=2)
        assert len(results) == 2

    async def test_hebbian_strengthening_applied(self) -> None:
        """After co-retrieval, at least one link between results should be strengthened."""
        em = self._engine_with_resonance()
        a = await em.encode("first memory")
        b = await em.encode("second memory")

        await em.retrieve("memory", top_k=2)

        a_after = await em.get(a.id)
        b_after = await em.get(b.id)
        assert a_after is not None
        assert b_after is not None
        all_links = list(a_after.tag.resonance_links) + list(b_after.tag.resonance_links)
        link_targets = {lnk.target_id for lnk in all_links}
        assert a.id in link_targets or b.id in link_targets


# ---------------------------------------------------------------------------
# Async elaborate() and elaborate_pending()
# ---------------------------------------------------------------------------


def _appraisal_engine(valence: float = 0.7, arousal: float = 0.6) -> StaticAppraisalEngine:
    return StaticAppraisalEngine(
        AppraisalVector(
            novelty=0.5,
            goal_relevance=0.8,
            coping_potential=0.7,
            norm_congruence=0.6,
            self_relevance=0.5,
        )
    )


def _dual_path_engine() -> AsyncEmotionalMemory:
    return AsyncEmotionalMemory(
        store=SyncToAsyncStore(InMemoryStore()),
        embedder=SyncToAsyncEmbedder(DeterministicEmbedder()),
        appraisal_engine=SyncToAsyncAppraisalEngine(_appraisal_engine()),
        config=EmotionalMemoryConfig(dual_path_encoding=True),
    )


class TestAsyncElaborate:
    async def test_elaborate_clears_pending_appraisal(self) -> None:
        em = _dual_path_engine()
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mem = await em.encode("Pending appraisal memory.")
        assert mem.tag.pending_appraisal is True

        updated = await em.elaborate(mem.id)
        assert updated is not None
        assert updated.tag.pending_appraisal is False

    async def test_elaborate_populates_appraisal_vector(self) -> None:
        em = _dual_path_engine()
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mem = await em.encode("Memory needing appraisal.")
        updated = await em.elaborate(mem.id)
        assert updated is not None
        assert updated.tag.appraisal is not None

    async def test_elaborate_blends_core_affect(self) -> None:
        em = _dual_path_engine()
        em.set_affect(CoreAffect(valence=0.0, arousal=0.1))
        mem = await em.encode("Memory at low arousal.")
        raw_affect = mem.tag.core_affect

        updated = await em.elaborate(mem.id)
        assert updated is not None
        assert updated.tag.core_affect != raw_affect

    async def test_elaborate_opens_reconsolidation_window(self) -> None:
        em = _dual_path_engine()
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mem = await em.encode("Memory to elaborate.")
        assert mem.tag.window_opened_at is None

        updated = await em.elaborate(mem.id)
        assert updated is not None
        assert updated.tag.window_opened_at is not None

    async def test_elaborate_persists_to_store(self) -> None:
        em = _dual_path_engine()
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mem = await em.encode("Stored memory.")
        await em.elaborate(mem.id)
        stored = await em.get(mem.id)
        assert stored is not None
        assert stored.tag.pending_appraisal is False

    async def test_elaborate_returns_none_when_not_pending(self) -> None:
        em = AsyncEmotionalMemory(
            store=SyncToAsyncStore(InMemoryStore()),
            embedder=SyncToAsyncEmbedder(DeterministicEmbedder()),
            appraisal_engine=SyncToAsyncAppraisalEngine(_appraisal_engine()),
            config=EmotionalMemoryConfig(dual_path_encoding=False),
        )
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mem = await em.encode("Already elaborated.")
        assert mem.tag.pending_appraisal is False
        result = await em.elaborate(mem.id)
        assert result is None

    async def test_elaborate_returns_none_for_missing_id(self) -> None:
        em = _dual_path_engine()
        result = await em.elaborate("nonexistent-id")
        assert result is None

    async def test_elaborate_returns_none_without_appraisal_engine(self) -> None:
        em = AsyncEmotionalMemory(
            store=SyncToAsyncStore(InMemoryStore()),
            embedder=SyncToAsyncEmbedder(DeterministicEmbedder()),
            appraisal_engine=None,
            config=EmotionalMemoryConfig(dual_path_encoding=True),
        )
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        # No engine → fast path not activated → pending_appraisal is False
        mem = await em.encode("No engine.")
        result = await em.elaborate(mem.id)
        assert result is None


class TestAsyncElaboratePending:
    async def test_elaborate_pending_processes_all(self) -> None:
        em = _dual_path_engine()
        for i in range(3):
            em.set_affect(CoreAffect(valence=0.1 * i, arousal=0.5))
            await em.encode(f"Memory {i} pending appraisal.")

        all_mems = await em.list_all()
        pending_before = sum(1 for m in all_mems if m.tag.pending_appraisal)
        assert pending_before == 3

        elaborated = await em.elaborate_pending()
        assert len(elaborated) == 3

        all_mems_after = await em.list_all()
        pending_after = sum(1 for m in all_mems_after if m.tag.pending_appraisal)
        assert pending_after == 0

    async def test_elaborate_pending_skips_non_pending(self) -> None:
        em = AsyncEmotionalMemory(
            store=SyncToAsyncStore(InMemoryStore()),
            embedder=SyncToAsyncEmbedder(DeterministicEmbedder()),
            appraisal_engine=SyncToAsyncAppraisalEngine(_appraisal_engine()),
            config=EmotionalMemoryConfig(dual_path_encoding=False),
        )
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        await em.encode("Already elaborated.")
        result = await em.elaborate_pending()
        assert result == []

    async def test_elaborate_pending_empty_store(self) -> None:
        em = _dual_path_engine()
        result = await em.elaborate_pending()
        assert result == []


# ---------------------------------------------------------------------------
# Async import_memories(overwrite=True)
# ---------------------------------------------------------------------------


class TestAsyncImportOverwrite:
    async def test_overwrite_replaces_existing(self) -> None:
        em = _async_engine()
        mem = await em.encode("Original content.")
        exported = await em.export_memories()

        # Mutate the exported data
        exported[0]["content"] = "Replaced content."

        count = await em.import_memories(exported, overwrite=True)
        assert count == 1
        updated = await em.get(mem.id)
        assert updated is not None
        assert updated.content == "Replaced content."

    async def test_no_overwrite_skips_duplicates(self) -> None:
        em = _async_engine()
        mem = await em.encode("Original content.")
        exported = await em.export_memories()
        exported[0]["content"] = "Should not replace."

        count = await em.import_memories(exported, overwrite=False)
        assert count == 0
        unchanged = await em.get(mem.id)
        assert unchanged is not None
        assert unchanged.content == "Original content."


# ---------------------------------------------------------------------------
# Async auto_categorize
# ---------------------------------------------------------------------------


class TestAsyncAutoCategorize:
    async def test_auto_categorize_attaches_emotion_label(self) -> None:
        em = AsyncEmotionalMemory(
            store=SyncToAsyncStore(InMemoryStore()),
            embedder=SyncToAsyncEmbedder(FixedEmbedder([1.0, 0.0])),
            config=EmotionalMemoryConfig(auto_categorize=True),
        )
        em.set_affect(CoreAffect(valence=0.8, arousal=0.7))
        mem = await em.encode("A joyful event.")
        assert mem.tag.emotion_label is not None

    async def test_auto_categorize_false_leaves_label_none(self) -> None:
        em = _async_engine()
        em.set_affect(CoreAffect(valence=0.8, arousal=0.7))
        mem = await em.encode("A joyful event.")
        assert mem.tag.emotion_label is None


# ---------------------------------------------------------------------------
# NaN embedding warning (async)
# ---------------------------------------------------------------------------


class _NaNEmbedder:
    """Embedder that always returns NaN values."""

    def embed(self, text: str) -> list[float]:
        return [float("nan"), float("nan")]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float("nan"), float("nan")] for _ in texts]


class TestAsyncNaNEmbeddingWarning:
    async def test_nan_embedding_emits_warning(self) -> None:
        em = AsyncEmotionalMemory(
            store=SyncToAsyncStore(InMemoryStore()),
            embedder=SyncToAsyncEmbedder(_NaNEmbedder()),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await em.encode("This will produce a NaN embedding.")
        assert any("NaN" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Async concurrency: concurrent encodes on separate engines
# ---------------------------------------------------------------------------


class TestAsyncConcurrentEncode:
    async def test_independent_engines_concurrent(self) -> None:
        """Multiple AsyncEmotionalMemory instances running concurrently
        with asyncio.gather() must not interfere with each other."""
        import asyncio

        async def run_engine(n: int) -> int:
            em = AsyncEmotionalMemory(
                store=SyncToAsyncStore(InMemoryStore()),
                embedder=SyncToAsyncEmbedder(FixedEmbedder([1.0, 0.0])),
            )
            for i in range(n):
                await em.encode(f"async memory {i}")
            return await em.count()

        counts = await asyncio.gather(*[run_engine(4) for _ in range(6)])
        assert all(c == 4 for c in counts), f"Unexpected counts: {counts}"

    async def test_shared_engine_concurrent_encode_no_lost_updates(self) -> None:
        """Concurrent encode() on the same engine must not lose state updates (Lock test).

        Without asyncio.Lock, two coroutines could both read the old _state, suspend
        at 'await embed()', then both update from the same base — losing one update.
        """
        import asyncio

        em = _async_engine(embedder=FixedEmbedder([1.0, 0.0]))
        n = 12
        await asyncio.gather(*[em.encode(f"memory {i}") for i in range(n)])
        count = await em.count()
        assert count == n, f"Expected {n} memories after {n} concurrent encodes, got {count}"
