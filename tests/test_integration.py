"""End-to-end integration scenarios for emotional_memory."""

from datetime import UTC, datetime, timedelta

from conftest import DeterministicEmbedder, PolarEmbedder

from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal import AppraisalVector
from emotional_memory.decay import DecayConfig
from emotional_memory.engine import EmotionalMemory, EmotionalMemoryConfig
from emotional_memory.resonance import ResonanceConfig
from emotional_memory.retrieval import RetrievalConfig
from emotional_memory.stores.in_memory import InMemoryStore


def _positive_appraisal() -> AppraisalVector:
    return AppraisalVector(
        novelty=0.2,
        goal_relevance=0.9,
        coping_potential=0.8,
        norm_congruence=0.7,
        self_relevance=0.6,
    )


def _negative_appraisal() -> AppraisalVector:
    return AppraisalVector(
        novelty=0.5,
        goal_relevance=-0.8,
        coping_potential=0.1,
        norm_congruence=-0.5,
        self_relevance=0.7,
    )


class TestMoodCongruentRetrieval:
    """Under negative Stimmung, negative memories should rank higher."""

    def test_negative_stimmung_biases_toward_negative_memories(self):
        config = EmotionalMemoryConfig(
            retrieval=RetrievalConfig(
                base_weights=[0.15, 0.40, 0.25, 0.10, 0.05, 0.05],  # strong emotional bias
                ape_threshold=0.5,
            ),
            stimmung_alpha=0.3,
        )
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=PolarEmbedder(),
            config=config,
        )

        # Encode 5 positive and 5 negative memories
        for _ in range(5):
            em.encode("joy and success today", appraisal=_positive_appraisal())
        for _ in range(5):
            em.encode("sadness and failure", appraisal=_negative_appraisal())

        # Drive Stimmung deeply negative
        em.set_affect(CoreAffect(valence=-0.95, arousal=0.6))
        for _ in range(10):
            em.set_affect(CoreAffect(valence=-0.95, arousal=0.6))

        results = em.retrieve("anything", top_k=3)

        negative_count = sum(1 for m in results if m.tag.core_affect.valence < 0.0)
        assert negative_count >= 2, (
            f"Expected ≥2 negative memories in top-3, got {negative_count}. "
            f"Valences: {[m.tag.core_affect.valence for m in results]}"
        )


class TestDecayRanking:
    """Memories encoded long ago should rank lower (lower effective strength)."""

    def test_recent_memories_rank_higher_than_old(self):
        store = InMemoryStore()
        config = EmotionalMemoryConfig(
            decay=DecayConfig(base_decay=0.8, arousal_modulation=0.0),
            retrieval=RetrievalConfig(
                base_weights=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # only decay/recency
                ape_threshold=1.0,  # disable reconsolidation
            ),
        )
        em = EmotionalMemory(store=store, embedder=DeterministicEmbedder(), config=config)

        # Use an appraisal that generates non-zero arousal so consolidation_strength > 0
        appraisal = AppraisalVector(
            novelty=0.8,
            goal_relevance=0.5,
            coping_potential=0.5,
            norm_congruence=0.0,
            self_relevance=0.8,
        )
        em.encode("old memory", appraisal=appraisal)
        em.encode("recent memory", appraisal=appraisal)

        # Back-date the old memory's timestamp AND preserve its consolidation_strength
        all_mems = store.list_all()
        old = next(m for m in all_mems if m.content == "old memory")
        old_tag = old.tag.model_copy(
            update={"timestamp": datetime.now(tz=UTC) - timedelta(days=30)}
        )
        store.update(old.model_copy(update={"tag": old_tag}))

        results = em.retrieve("query", top_k=2)
        assert results[0].content == "recent memory"


class TestResonanceLinks:
    """Encoding similar content should create resonance links."""

    def test_similar_content_creates_resonance_links(self):
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=DeterministicEmbedder(),
            config=EmotionalMemoryConfig(resonance=ResonanceConfig(threshold=0.05, max_links=5)),
        )
        # Encode same content twice — embeddings identical → high semantic resonance
        em.encode("the project went well")
        m2 = em.encode("the project went well")

        # m2 should have at least one resonance link to m1
        assert len(m2.tag.resonance_links) >= 1


class TestReconsolidation:
    """Retrieval with divergent affect should update the tag."""

    def test_emotional_context_shift_updates_tag(self):
        config = EmotionalMemoryConfig(
            retrieval=RetrievalConfig(
                ape_threshold=0.01,
                reconsolidation_learning_rate=0.5,
            )
        )
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=DeterministicEmbedder(),
            config=config,
        )

        # Encode under neutral affect
        em.set_affect(CoreAffect(valence=0.0, arousal=0.0))
        em.encode("the event")
        original_valence = em._store.list_all()[0].tag.core_affect.valence

        # Retrieve under very positive affect → reconsolidation
        em.set_affect(CoreAffect(valence=1.0, arousal=0.9))
        results = em.retrieve("the event", top_k=1)

        assert results[0].tag.reconsolidation_count == 1
        assert results[0].tag.core_affect.valence > original_valence
