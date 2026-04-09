"""Tests for EmotionalMemory facade."""

import pytest
from conftest import FixedEmbedder

from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal import AppraisalVector, StaticAppraisalEngine
from emotional_memory.engine import EmotionalMemory, EmotionalMemoryConfig
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

    def test_stimmung_evolves_after_encodes(self):
        em = _engine()
        initial_stimmung_valence = em.get_state().stimmung.valence
        positive = AppraisalVector(
            novelty=0.0,
            goal_relevance=1.0,
            coping_potential=1.0,
            norm_congruence=1.0,
            self_relevance=0.0,
        )
        for _ in range(20):
            em.encode("positive event", appraisal=positive)
        assert em.get_state().stimmung.valence > initial_stimmung_valence

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

    def test_load_state_restores_stimmung(self):
        em = _engine()
        for _ in range(20):
            em.set_affect(CoreAffect(valence=1.0, arousal=0.8))
        snapshot = em.save_state()

        em2 = _engine()
        em2.load_state(snapshot)
        assert em2.get_state().stimmung.valence == pytest.approx(
            em.get_state().stimmung.valence, abs=1e-9
        )

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


class TestGetCurrentStimmung:
    def test_without_decay_returns_frozen_stimmung(self):
        em = _engine()
        em.set_affect(CoreAffect(valence=0.8, arousal=0.7))
        s = em.get_current_stimmung()
        assert s.valence == pytest.approx(em.get_state().stimmung.valence, abs=1e-9)

    def test_with_decay_returns_regressed_stimmung(self):
        from datetime import UTC, datetime, timedelta

        from emotional_memory.stimmung import StimmungDecayConfig

        cfg = EmotionalMemoryConfig(stimmung_decay=StimmungDecayConfig(base_half_life_seconds=1.0))
        em = _engine(config=cfg)
        em.set_affect(CoreAffect(valence=1.0, arousal=0.9))
        future = datetime.now(tz=UTC) + timedelta(hours=10)
        s = em.get_current_stimmung(now=future)
        assert s.valence < em.get_state().stimmung.valence

    def test_does_not_modify_internal_state(self):
        from datetime import UTC, datetime, timedelta

        from emotional_memory.stimmung import StimmungDecayConfig

        cfg = EmotionalMemoryConfig(stimmung_decay=StimmungDecayConfig(base_half_life_seconds=1.0))
        em = _engine(config=cfg)
        em.set_affect(CoreAffect(valence=1.0, arousal=0.8))
        before = em.get_state().stimmung.valence
        future = datetime.now(tz=UTC) + timedelta(hours=5)
        em.get_current_stimmung(now=future)
        assert em.get_state().stimmung.valence == pytest.approx(before, abs=1e-9)


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
