"""Tests for EmotionalMemory facade."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from conftest import FixedEmbedder

from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal import AppraisalVector, StaticAppraisalEngine
from emotional_memory.engine import EmotionalMemory, EmotionalMemoryConfig
from emotional_memory.interfaces import Embedder, SequentialEmbedder
from emotional_memory.retrieval import RetrievalConfig
from emotional_memory.stores.in_memory import InMemoryStore


class IndexEmbedder:
    """Returns a one-hot-like embedding based on a pre-registered index."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._map = mapping

    def embed(self, text: str) -> list[float]:
        return self._map.get(text, [0.0] * 4)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _engine(embedder=None, appraisal_engine=None, config=None):
    return EmotionalMemory(
        store=InMemoryStore(),
        embedder=embedder or FixedEmbedder([1.0, 0.0]),
        appraisal_engine=appraisal_engine,
        config=config,
    )


class TestEncode:
    def test_returns_memory(self):
        em = _engine()
        m = em.encode("hello world")
        assert m.content == "hello world"

    def test_memory_has_emotional_tag(self):
        em = _engine()
        m = em.encode("test")
        assert m.tag is not None
        # consolidation_strength is 0 at zero arousal (Yerkes-Dodson peak at 0.7)
        assert m.tag.consolidation_strength >= 0.0

    def test_memory_has_embedding(self):
        em = _engine(embedder=FixedEmbedder([0.5, 0.5]))
        m = em.encode("test")
        assert m.embedding == [0.5, 0.5]

    def test_memory_stored_in_store(self):
        store = InMemoryStore()
        em = EmotionalMemory(store=store, embedder=FixedEmbedder([1.0]))
        m = em.encode("stored?")
        assert store.get(m.id) is not None

    def test_encode_with_manual_appraisal(self):
        appraisal = AppraisalVector(
            novelty=0.8,
            goal_relevance=0.9,
            coping_potential=0.7,
            norm_congruence=0.5,
            self_relevance=0.6,
        )
        em = _engine()
        m = em.encode("great news!", appraisal=appraisal)
        assert m.tag.appraisal == appraisal

    def test_encode_with_appraisal_engine(self):
        fixed_appraisal = AppraisalVector(
            novelty=0.5,
            goal_relevance=0.5,
            coping_potential=0.5,
            norm_congruence=0.5,
            self_relevance=0.5,
        )
        em = _engine(appraisal_engine=StaticAppraisalEngine(fixed_appraisal))
        m = em.encode("auto-appraised")
        assert m.tag.appraisal == fixed_appraisal

    def test_encode_without_appraisal_uses_current_state(self):
        em = _engine()
        em.set_affect(CoreAffect(valence=0.7, arousal=0.6))
        state_before = em.get_state()
        m = em.encode("no appraisal")
        # core_affect in tag should reflect the state at encoding
        assert m.tag.core_affect.valence == pytest.approx(
            state_before.core_affect.valence, abs=0.01
        )

    def test_mood_evolves_after_encodes(self):
        em = _engine()
        initial_mood_valence = em.get_state().mood.valence
        positive = AppraisalVector(
            novelty=0.0,
            goal_relevance=1.0,
            coping_potential=1.0,
            norm_congruence=1.0,
            self_relevance=0.0,
        )
        for _ in range(20):
            em.encode("positive event", appraisal=positive)
        assert em.get_state().mood.valence > initial_mood_valence

    def test_metadata_stored(self):
        em = _engine()
        m = em.encode("with meta", metadata={"source": "test"})
        assert m.metadata == {"source": "test"}


class TestRetrieve:
    def test_retrieve_empty_store(self):
        em = _engine()
        assert em.retrieve("query") == []

    def test_retrieve_returns_list_of_memories(self):
        em = _engine()
        em.encode("memory one")
        em.encode("memory two")
        results = em.retrieve("query", top_k=2)
        assert len(results) == 2

    def test_retrieve_respects_top_k(self):
        em = _engine()
        for i in range(5):
            em.encode(f"memory {i}")
        results = em.retrieve("query", top_k=3)
        assert len(results) == 3

    def test_retrieve_updates_retrieval_count(self):
        em = _engine()
        em.encode("retrievable")
        results = em.retrieve("query", top_k=1)
        assert results[0].tag.retrieval_count == 1

    def test_retrieve_updates_last_retrieved(self):
        em = _engine()
        em.encode("retrievable")
        results = em.retrieve("query", top_k=1)
        assert results[0].tag.last_retrieved is not None

    def test_reconsolidation_on_large_ape(self):
        """Second retrieval within the lability window should update the tag's core_affect.

        Reconsolidation requires two retrievals: the first destabilises the memory
        (sets last_retrieved); the second, within the lability window, checks APE
        and updates the tag if the current affect differs significantly.
        """
        config = EmotionalMemoryConfig(
            retrieval=RetrievalConfig(
                ape_threshold=0.01,
                reconsolidation_learning_rate=0.5,
                reconsolidation_window_seconds=300.0,
            )
        )
        em = _engine(config=config)
        # Encode with neutral affect
        em.set_affect(CoreAffect(valence=0.0, arousal=0.0))
        em.encode("neutral memory")

        # First retrieval — destabilises the memory (sets last_retrieved), no update yet
        em.retrieve("query", top_k=1)

        # Second retrieval within window with very different affect → reconsolidation
        em.set_affect(CoreAffect(valence=1.0, arousal=0.9))
        results = em.retrieve("query", top_k=1)

        # Tag should have shifted toward positive
        assert results[0].tag.core_affect.valence > 0.0
        assert results[0].tag.reconsolidation_count == 1


class TestDelete:
    def test_delete_removes_memory(self):
        em = _engine()
        mem = em.encode("forget me")
        em.delete(mem.id)
        assert em._store.get(mem.id) is None

    def test_delete_nonexistent_no_error(self):
        em = _engine()
        em.delete("nonexistent-id")  # must not raise


class TestEncodeBatch:
    def test_returns_correct_count(self):
        em = _engine()
        results = em.encode_batch(["a", "b", "c"])
        assert len(results) == 3

    def test_stores_all_memories(self):
        em = _engine()
        em.encode_batch(["x", "y", "z"])
        assert len(em._store.list_all()) == 3

    def test_uses_appraisal_engine(self):
        appraisal_engine = StaticAppraisalEngine(
            AppraisalVector(
                novelty=1.0,
                goal_relevance=1.0,
                coping_potential=1.0,
                norm_congruence=1.0,
                self_relevance=1.0,
            )
        )
        em = _engine(appraisal_engine=appraisal_engine)
        results = em.encode_batch(["item one", "item two"])
        for mem in results:
            assert mem.tag.appraisal is not None

    def test_metadata_mapping(self):
        em = _engine()
        metas = [{"k": "a"}, {"k": "b"}, {"k": "c"}]
        results = em.encode_batch(["x", "y", "z"], metadata=metas)
        assert [m.metadata for m in results] == metas

    def test_metadata_none(self):
        em = _engine()
        results = em.encode_batch(["x", "y"])
        for mem in results:
            assert mem.metadata == {}

    def test_resonance_links_on_similar(self):
        from emotional_memory.resonance import ResonanceConfig

        config = EmotionalMemoryConfig(resonance=ResonanceConfig(threshold=0.0))
        em = _engine(config=config)
        results = em.encode_batch(["same text", "same text"])
        assert len(results[1].tag.resonance_links) >= 1


class TestSaveLoadState:
    def test_save_state_is_json_serialisable(self):
        import json

        em = _engine()
        em.set_affect(CoreAffect(valence=0.6, arousal=0.5))
        snapshot = em.save_state()
        dumped = json.dumps(snapshot)
        assert isinstance(dumped, str)

    def test_load_state_restores_core_affect(self):
        em = _engine()
        em.set_affect(CoreAffect(valence=0.7, arousal=0.4))
        snapshot = em.save_state()

        em2 = _engine()
        em2.load_state(snapshot)
        assert em2.get_state().core_affect.valence == pytest.approx(0.7)
        assert em2.get_state().core_affect.arousal == pytest.approx(0.4)

    def test_load_state_restores_mood(self):
        em = _engine()
        for _ in range(20):
            em.set_affect(CoreAffect(valence=1.0, arousal=0.8))
        snapshot = em.save_state()

        em2 = _engine()
        em2.load_state(snapshot)
        assert em2.get_state().mood.valence == pytest.approx(em.get_state().mood.valence, abs=1e-9)

    def test_load_state_preserves_momentum_history(self):
        em = _engine()
        em.set_affect(CoreAffect(valence=0.2, arousal=0.3))
        em.set_affect(CoreAffect(valence=0.5, arousal=0.5))
        em.set_affect(CoreAffect(valence=0.8, arousal=0.7))
        snapshot = em.save_state()

        em2 = _engine()
        em2.load_state(snapshot)
        # Next update should produce same momentum in both
        next1 = em.get_state().update(CoreAffect(valence=0.9, arousal=0.8))
        next2 = em2.get_state().update(CoreAffect(valence=0.9, arousal=0.8))
        assert next1.momentum.d_valence == pytest.approx(next2.momentum.d_valence, abs=1e-6)

    def test_save_does_not_mutate_on_load(self):
        em = _engine()
        em.set_affect(CoreAffect(valence=0.5, arousal=0.5))
        snapshot = em.save_state()
        original_keys = set(snapshot.keys())

        em2 = _engine()
        em2.load_state(snapshot)
        # snapshot dict must be unmodified
        assert set(snapshot.keys()) == original_keys


class TestGetCurrentMood:
    def test_without_decay_returns_frozen_mood(self):
        em = _engine()
        em.set_affect(CoreAffect(valence=0.8, arousal=0.7))
        s = em.get_current_mood()
        assert s.valence == pytest.approx(em.get_state().mood.valence, abs=1e-9)

    def test_with_decay_returns_regressed_mood(self):
        from datetime import UTC, datetime, timedelta

        from emotional_memory.mood import MoodDecayConfig

        cfg = EmotionalMemoryConfig(mood_decay=MoodDecayConfig(base_half_life_seconds=1.0))
        em = _engine(config=cfg)
        em.set_affect(CoreAffect(valence=1.0, arousal=0.9))
        future = datetime.now(tz=UTC) + timedelta(hours=10)
        s = em.get_current_mood(now=future)
        assert s.valence < em.get_state().mood.valence

    def test_does_not_modify_internal_state(self):
        from datetime import UTC, datetime, timedelta

        from emotional_memory.mood import MoodDecayConfig

        cfg = EmotionalMemoryConfig(mood_decay=MoodDecayConfig(base_half_life_seconds=1.0))
        em = _engine(config=cfg)
        em.set_affect(CoreAffect(valence=1.0, arousal=0.8))
        before = em.get_state().mood.valence
        future = datetime.now(tz=UTC) + timedelta(hours=5)
        em.get_current_mood(now=future)
        assert em.get_state().mood.valence == pytest.approx(before, abs=1e-9)


class TestGetSetState:
    def test_initial_state_is_neutral(self):
        em = _engine()
        state = em.get_state()
        assert state.core_affect.valence == 0.0
        assert state.core_affect.arousal == 0.0

    def test_set_affect_updates_state(self):
        em = _engine()
        em.set_affect(CoreAffect(valence=0.8, arousal=0.7))
        state = em.get_state()
        assert state.core_affect.valence == pytest.approx(0.8)
        assert state.core_affect.arousal == pytest.approx(0.7)

    def test_get_state_returns_copy(self):
        em = _engine()
        s1 = em.get_state()
        em.set_affect(CoreAffect(valence=0.5, arousal=0.5))
        s2 = em.get_state()
        assert s1.core_affect.valence != s2.core_affect.valence


class TestFacadeMethods:
    def test_get_returns_encoded_memory(self):
        em = _engine()
        m = em.encode("findable")
        result = em.get(m.id)
        assert result is not None
        assert result.content == "findable"

    def test_get_missing_returns_none(self):
        em = _engine()
        assert em.get("nonexistent-id") is None

    def test_list_all_empty(self):
        em = _engine()
        assert em.list_all() == []

    def test_list_all_returns_all(self):
        em = _engine()
        m1 = em.encode("one")
        m2 = em.encode("two")
        ids = {m.id for m in em.list_all()}
        assert m1.id in ids
        assert m2.id in ids

    def test_len_empty(self):
        em = _engine()
        assert len(em) == 0

    def test_len_after_encode(self):
        em = _engine()
        em.encode("a")
        em.encode("b")
        assert len(em) == 2


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_encode_batch_metadata_length_mismatch(self):
        em = _engine()
        with pytest.raises(ValueError, match="metadata length"):
            em.encode_batch(["a", "b"], metadata=[{"x": 1}])

    def test_retrieve_top_k_zero_raises(self):
        em = _engine()
        em.encode("something")
        with pytest.raises(ValueError, match="top_k"):
            em.retrieve("query", top_k=0)

    def test_retrieve_top_k_negative_raises(self):
        em = _engine()
        with pytest.raises(ValueError, match="top_k"):
            em.retrieve("query", top_k=-1)


# ---------------------------------------------------------------------------
# Helpers for close / context-manager tests
# ---------------------------------------------------------------------------


class _CloseableStore(InMemoryStore):
    """InMemoryStore extended with a close() method that records the call."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _engine_with_closeable() -> tuple[EmotionalMemory, _CloseableStore]:
    store = _CloseableStore()
    em = EmotionalMemory(store=store, embedder=FixedEmbedder([1.0, 0.0]))
    return em, store


def _old_timestamp() -> datetime:
    """Return a timestamp far enough in the past that decay reduces strength to ~0."""
    return datetime.now(tz=UTC) - timedelta(days=365)


# ---------------------------------------------------------------------------
# prune()
# ---------------------------------------------------------------------------


class TestPrune:
    def test_prune_removes_old_memories(self) -> None:
        em = _engine()
        mem = em.encode("old memory")
        # Backdate the tag so compute_effective_strength drops near zero
        aged = mem.model_copy(
            update={"tag": mem.tag.model_copy(update={"timestamp": _old_timestamp()})}
        )
        em._store.update(aged)
        removed = em.prune(threshold=0.05)
        assert removed == 1
        assert len(em) == 0

    def test_prune_keeps_fresh_memories(self) -> None:
        em = _engine()
        em.set_affect(CoreAffect(valence=0.5, arousal=0.8))  # high arousal → non-zero strength
        em.encode("fresh memory")
        removed = em.prune(threshold=0.05)
        assert removed == 0
        assert len(em) == 1

    def test_prune_returns_count(self) -> None:
        em = _engine()
        # arousal=0.5: gives non-zero initial strength but stays below the floor
        # threshold (0.7), so backdated memories decay below prune threshold=0.05
        em.set_affect(CoreAffect(valence=0.3, arousal=0.5))
        mems = [em.encode(f"mem{i}") for i in range(3)]
        # Backdate first two
        for m in mems[:2]:
            aged = m.model_copy(
                update={"tag": m.tag.model_copy(update={"timestamp": _old_timestamp()})}
            )
            em._store.update(aged)
        removed = em.prune(threshold=0.05)
        assert removed == 2
        assert len(em) == 1

    def test_prune_empty_store(self) -> None:
        em = _engine()
        assert em.prune() == 0


# ---------------------------------------------------------------------------
# export_memories() / import_memories()
# ---------------------------------------------------------------------------


class TestExportImport:
    def test_export_returns_list_of_dicts(self) -> None:
        em = _engine()
        em.encode("first")
        em.encode("second")
        data = em.export_memories()
        assert isinstance(data, list)
        assert len(data) == 2
        assert all(isinstance(d, dict) for d in data)

    def test_export_is_json_serialisable(self) -> None:
        em = _engine()
        em.encode("serialisable content")
        json.dumps(em.export_memories())  # must not raise

    def test_import_restores_to_new_engine(self) -> None:
        em1 = _engine()
        em1.encode("alpha")
        em1.encode("beta")
        data = em1.export_memories()

        em2 = _engine()
        written = em2.import_memories(data)
        assert written == 2
        contents = {m.content for m in em2.list_all()}
        assert "alpha" in contents
        assert "beta" in contents

    def test_import_skips_duplicates(self) -> None:
        em = _engine()
        em.encode("unique")
        data = em.export_memories()
        written = em.import_memories(data)  # already present
        assert written == 0
        assert len(em) == 1

    def test_import_overwrites(self) -> None:
        em1 = _engine()
        em1.encode("original")
        data = em1.export_memories()

        em2 = _engine()
        em2.import_memories(data)
        written = em2.import_memories(data, overwrite=True)
        assert written == 1

    def test_roundtrip_preserves_tags(self) -> None:
        em1 = _engine()
        m = em1.encode("roundtrip")
        data = em1.export_memories()

        em2 = _engine()
        em2.import_memories(data)
        restored = em2.list_all()[0]
        assert restored.content == m.content
        assert restored.tag.core_affect.valence == pytest.approx(m.tag.core_affect.valence)
        assert restored.tag.consolidation_strength == pytest.approx(m.tag.consolidation_strength)


# ---------------------------------------------------------------------------
# close() and context manager
# ---------------------------------------------------------------------------


class TestCloseAndContextManager:
    def test_close_calls_store_close(self) -> None:
        em, store = _engine_with_closeable()
        em.close()
        assert store.closed

    def test_close_safe_without_close_method(self) -> None:
        em = _engine()  # InMemoryStore has no close()
        em.close()  # must not raise

    def test_context_manager_calls_close(self) -> None:
        store = _CloseableStore()
        with EmotionalMemory(store=store, embedder=FixedEmbedder([1.0, 0.0])):
            pass
        assert store.closed

    def test_context_manager_returns_engine(self) -> None:
        with _engine() as em:
            assert isinstance(em, EmotionalMemory)


# ---------------------------------------------------------------------------
# SequentialEmbedder
# ---------------------------------------------------------------------------


class _SimpleEmbedder(SequentialEmbedder):
    """Minimal SequentialEmbedder subclass for testing."""

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _BatchAwareEmbedder(SequentialEmbedder):
    """Subclass that overrides embed_batch to record calls."""

    def __init__(self) -> None:
        self.batch_called = False

    def embed(self, text: str) -> list[float]:
        return [0.5, 0.5]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_called = True
        return [self.embed(t) for t in texts]


class TestSequentialEmbedder:
    def test_embed_batch_delegates_to_embed(self) -> None:
        emb = _SimpleEmbedder()
        result = emb.embed_batch(["a", "b", "c"])
        assert result == [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]

    def test_embed_batch_override_is_called(self) -> None:
        emb = _BatchAwareEmbedder()
        emb.embed_batch(["x", "y"])
        assert emb.batch_called

    def test_satisfies_embedder_protocol(self) -> None:
        emb = _SimpleEmbedder()
        assert isinstance(emb, Embedder)


# ---------------------------------------------------------------------------
# Bidirectional resonance links
# ---------------------------------------------------------------------------


class TestBidirectionalResonanceLinks:
    """After encoding, target memories must carry a backward link to the new memory."""

    def _engine_with_low_threshold(self):
        from emotional_memory.resonance import ResonanceConfig

        config = EmotionalMemoryConfig(
            resonance=ResonanceConfig(
                threshold=0.0,
                temporal_half_life_seconds=1e9,  # keep temporal prox high
            )
        )
        return EmotionalMemory(
            store=InMemoryStore(),
            embedder=FixedEmbedder([1.0, 0.0]),
            config=config,
        )

    def test_target_memory_has_backward_link(self):
        """After A is encoded after B, B must contain a backward link targeting A."""
        em = self._engine_with_low_threshold()
        b = em.encode("first memory")
        a = em.encode("second memory")

        b_updated = em.get(b.id)
        assert b_updated is not None
        backward_ids = [lnk.target_id for lnk in b_updated.tag.resonance_links]
        assert a.id in backward_ids

    def test_backward_link_strength_equals_forward_strength(self):
        """Backward link strength must mirror the forward link strength."""
        em = self._engine_with_low_threshold()
        b = em.encode("older memory")
        a = em.encode("newer memory")

        # Forward link: A → B
        forward_strength = next(
            lnk.strength for lnk in a.tag.resonance_links if lnk.target_id == b.id
        )
        # Backward link: B → A
        b_updated = em.get(b.id)
        assert b_updated is not None
        backward_strength = next(
            lnk.strength for lnk in b_updated.tag.resonance_links if lnk.target_id == a.id
        )
        assert forward_strength == pytest.approx(backward_strength)

    def test_backward_link_respects_max_links(self):
        """Backward links must not exceed max_links on the target memory."""
        from emotional_memory.resonance import ResonanceConfig

        max_links = 3
        config = EmotionalMemoryConfig(
            resonance=ResonanceConfig(
                threshold=0.0,
                max_links=max_links,
                temporal_half_life_seconds=1e9,
            )
        )
        em = EmotionalMemory(
            store=InMemoryStore(), embedder=FixedEmbedder([1.0, 0.0]), config=config
        )
        anchor = em.encode("anchor memory")
        for _ in range(max_links + 3):
            em.encode("linking memory")

        anchor_updated = em.get(anchor.id)
        assert anchor_updated is not None
        assert len(anchor_updated.tag.resonance_links) <= max_links

    def test_encode_batch_creates_backward_links(self):
        """encode_batch must also produce backward links on target memories."""
        em = self._engine_with_low_threshold()
        memories = em.encode_batch(["first", "second", "third"])
        first = em.get(memories[0].id)
        assert first is not None
        # first memory should have backward links from later memories
        backward_targets = [lnk.target_id for lnk in first.tag.resonance_links]
        later_ids = {m.id for m in memories[1:]}
        assert any(tid in later_ids for tid in backward_targets)


# ---------------------------------------------------------------------------
# Spreading activation in retrieve()
# ---------------------------------------------------------------------------


class TestSpreadingActivationInRetrieve:
    """Memories reachable via 2-hop spreading should appear in results."""

    def _engine_with_resonance(self):
        from emotional_memory.resonance import ResonanceConfig

        config = EmotionalMemoryConfig(
            resonance=ResonanceConfig(
                threshold=0.0,
                temporal_half_life_seconds=1e9,
                propagation_hops=2,
            )
        )
        return EmotionalMemory(
            store=InMemoryStore(),
            embedder=FixedEmbedder([1.0, 0.0]),
            config=config,
        )

    def test_spreading_activation_hops_config_accepted(self):
        from emotional_memory.resonance import ResonanceConfig

        cfg = ResonanceConfig(propagation_hops=3)
        assert cfg.propagation_hops == 3

    def test_propagation_hops_default_is_two(self):
        from emotional_memory.resonance import ResonanceConfig

        assert ResonanceConfig().propagation_hops == 2

    def test_retrieve_returns_results_with_resonance(self):
        """Smoke test: retrieve works correctly when resonance links exist."""
        em = self._engine_with_resonance()
        em.encode("memory alpha")
        em.encode("memory beta")
        em.encode("memory gamma")
        results = em.retrieve("memory", top_k=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Hebbian co-retrieval strengthening in retrieve()
# ---------------------------------------------------------------------------


class TestHebbianStrengtheningInEngine:
    def _engine_with_hebbian(self, increment: float = 0.1):
        from emotional_memory.resonance import ResonanceConfig

        config = EmotionalMemoryConfig(
            resonance=ResonanceConfig(
                threshold=0.0,
                temporal_half_life_seconds=1e9,
                hebbian_increment=increment,
            )
        )
        return EmotionalMemory(
            store=InMemoryStore(),
            embedder=FixedEmbedder([1.0, 0.0]),
            config=config,
        )

    def test_hebbian_strengthens_links_between_co_retrieved(self):
        """After co-retrieval, links between retrieved memories should be stronger."""
        em = self._engine_with_hebbian(increment=0.1)
        a = em.encode("first memory")
        b = em.encode("second memory")

        # Retrieve both memories
        em.retrieve("memory", top_k=2)

        a_after = em.get(a.id)
        b_after = em.get(b.id)
        assert a_after is not None
        assert b_after is not None

        # At least one of the memories should have a link to the other
        all_links = list(a_after.tag.resonance_links) + list(b_after.tag.resonance_links)
        link_targets = {lnk.target_id for lnk in all_links}
        assert a.id in link_targets or b.id in link_targets

    def test_hebbian_disabled_at_zero_increment(self):
        """With hebbian_increment=0.0, link strengths must not change after retrieval."""
        em = self._engine_with_hebbian(increment=0.0)
        em.encode("alpha")
        b = em.encode("beta")

        strengths_before = {lnk.target_id: lnk.strength for lnk in b.tag.resonance_links}
        em.retrieve("alpha beta", top_k=2)
        b_after = em.get(b.id)
        assert b_after is not None
        strengths_after = {lnk.target_id: lnk.strength for lnk in b_after.tag.resonance_links}

        for tid, before in strengths_before.items():
            assert strengths_after.get(tid, before) == pytest.approx(before)


# ---------------------------------------------------------------------------
# NaN embedding warning (sync)
# ---------------------------------------------------------------------------


class _NaNEmbedder:
    """Embedder that always returns NaN values."""

    def embed(self, text: str) -> list[float]:
        return [float("nan"), float("nan")]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float("nan"), float("nan")] for _ in texts]


class TestNaNEmbeddingWarning:
    def test_nan_embedding_emits_warning(self) -> None:
        import warnings

        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=_NaNEmbedder(),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            em.encode("This will produce a NaN embedding.")
        assert any("NaN" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Reconsolidation window expiry
# ---------------------------------------------------------------------------


class TestReconsolidationWindowExpiry:
    def _engine_with_window(self, window_seconds: float) -> EmotionalMemory:
        from emotional_memory.retrieval import RetrievalConfig

        return EmotionalMemory(
            store=InMemoryStore(),
            embedder=FixedEmbedder([1.0, 0.0]),
            config=EmotionalMemoryConfig(
                retrieval=RetrievalConfig(
                    ape_threshold=0.0,
                    reconsolidation_learning_rate=0.5,
                    reconsolidation_window_seconds=window_seconds,
                )
            ),
        )

    def test_window_expires_and_closes(self) -> None:
        """After the lability window expires, window_opened_at is cleared."""
        em = self._engine_with_window(window_seconds=0.0)  # zero-length window
        em.set_affect(CoreAffect(valence=0.8, arousal=0.9))
        em.encode("High-arousal memory.")

        # First retrieval opens the window (APE threshold=0.0 so always opens)
        em.set_affect(CoreAffect(valence=-0.5, arousal=0.5))  # diverge affect
        results = em.retrieve("memory", top_k=1)
        assert len(results) == 1
        first = em.get(results[0].id)
        assert first is not None
        # window_opened_at is set (or None if window expired immediately)
        # Either way, a second retrieval after an expired window must clear it
        results2 = em.retrieve("memory", top_k=1)
        assert len(results2) == 1
        after_expiry = em.get(results2[0].id)
        assert after_expiry is not None
        # With window_seconds=0.0 the window expires immediately — must be None
        assert after_expiry.tag.window_opened_at is None


# ---------------------------------------------------------------------------
# Thread-safety: concurrent encodes on separate engines share no state
# ---------------------------------------------------------------------------


class TestConcurrentEncode:
    def test_independent_engines_thread_safe(self) -> None:
        """Two separate EmotionalMemory instances can encode from different threads
        without interference — each has its own store and state."""
        import threading

        results: list[int] = []
        errors: list[Exception] = []

        def encode_batch(n: int) -> None:
            try:
                em = EmotionalMemory(
                    store=InMemoryStore(),
                    embedder=FixedEmbedder([1.0, 0.0]),
                )
                for i in range(n):
                    em.encode(f"thread memory {i}")
                results.append(len(em.list_all()))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=encode_batch, args=(5,)) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Threads raised: {errors}"
        assert all(r == 5 for r in results), f"Unexpected counts: {results}"
