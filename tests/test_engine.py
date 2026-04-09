"""Tests for EmotionalMemory facade."""

import pytest

from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal import AppraisalVector, StaticAppraisalEngine
from emotional_memory.engine import EmotionalMemory, EmotionalMemoryConfig
from emotional_memory.retrieval import RetrievalConfig
from emotional_memory.stores.in_memory import InMemoryStore

from conftest import FixedEmbedder


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
        """Retrieval with very different affect should update the tag's core_affect."""
        config = EmotionalMemoryConfig(
            retrieval=RetrievalConfig(ape_threshold=0.01, reconsolidation_learning_rate=0.5)
        )
        em = _engine(config=config)
        # Encode with neutral affect
        em.set_affect(CoreAffect(valence=0.0, arousal=0.0))
        em.encode("neutral memory")

        # Now set very positive affect before retrieval
        em.set_affect(CoreAffect(valence=1.0, arousal=0.9))
        results = em.retrieve("query", top_k=1)

        # Tag should have shifted toward positive
        assert results[0].tag.core_affect.valence > 0.0
        assert results[0].tag.reconsolidation_count == 1


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
