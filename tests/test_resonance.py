import math
from datetime import UTC, datetime, timedelta

import pytest
from conftest import make_test_memory

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.models import Memory, ResonanceLink, make_emotional_tag
from emotional_memory.mood import MoodField
from emotional_memory.resonance import (
    ResonanceConfig,
    build_resonance_links,
    hebbian_strengthen,
    spreading_activation,
    temporal_proximity,
)


def _now():
    return datetime.now(tz=UTC)


class TestTemporalProximity:
    def test_simultaneous_is_one(self):
        t = _now()
        assert math.isclose(temporal_proximity(t, t), 1.0)

    def test_decreases_with_distance(self):
        t = _now()
        t2 = t + timedelta(hours=1)
        t3 = t + timedelta(hours=5)
        assert temporal_proximity(t, t2) > temporal_proximity(t, t3)

    def test_half_life(self):
        t = _now()
        t2 = t + timedelta(seconds=3600)
        result = temporal_proximity(t, t2, half_life_seconds=3600.0)
        assert math.isclose(result, 0.5, rel_tol=1e-6)

    def test_symmetric(self):
        t1 = _now()
        t2 = t1 + timedelta(hours=2)
        assert math.isclose(temporal_proximity(t1, t2), temporal_proximity(t2, t1))


class TestBuildResonanceLinks:
    def test_no_links_for_empty_candidates(self):
        m = make_test_memory()
        links = build_resonance_links(m, [], ResonanceConfig())
        assert links == []

    def test_no_self_links(self):
        m = make_test_memory()
        links = build_resonance_links(m, [m], ResonanceConfig())
        assert links == []

    def test_semantically_similar_gets_semantic_link(self):
        config = ResonanceConfig(threshold=0.1, temporal_half_life_seconds=1.0)
        new_m = make_test_memory(embedding=[1.0, 0.0, 0.0])
        similar = make_test_memory(embedding=[1.0, 0.0, 0.0], offset_seconds=100_000)
        links = build_resonance_links(new_m, [similar], config)
        assert len(links) == 1
        assert links[0].link_type == "semantic"

    def test_emotionally_similar_gets_emotional_link(self):
        # No embeddings (sem_sim=0), same affect, far apart in time
        config = ResonanceConfig(
            threshold=0.1,
            semantic_weight=0.0,
            emotional_weight=1.0,
            temporal_weight=0.0,
        )
        new_m = make_test_memory(valence=0.9, arousal=0.9, offset_seconds=0)
        similar = make_test_memory(valence=0.9, arousal=0.9, offset_seconds=100_000)
        links = build_resonance_links(new_m, [similar], config)
        assert len(links) == 1
        assert links[0].link_type == "emotional"

    def test_temporally_close_gets_temporal_link(self):
        config = ResonanceConfig(threshold=0.1, temporal_half_life_seconds=3600.0)
        new_m = make_test_memory(valence=0.0, arousal=0.0, offset_seconds=0)
        close = make_test_memory(valence=0.0, arousal=0.0, offset_seconds=1)
        links = build_resonance_links(new_m, [close], config)
        assert len(links) == 1
        assert links[0].link_type in ("temporal", "emotional")

    def test_below_threshold_no_link(self):
        config = ResonanceConfig(threshold=0.99)  # almost impossible to reach
        new_m = make_test_memory(embedding=[1.0, 0.0])
        far = make_test_memory(embedding=[0.0, 1.0], offset_seconds=100_000)
        links = build_resonance_links(new_m, [far], config)
        assert links == []

    def test_max_links_respected(self):
        config = ResonanceConfig(threshold=0.0, max_links=2)
        new_m = make_test_memory(embedding=[1.0, 0.0])
        candidates = [make_test_memory(embedding=[1.0, 0.0]) for _ in range(10)]
        links = build_resonance_links(new_m, candidates, config)
        assert len(links) <= 2

    def test_links_sorted_by_strength_desc(self):
        config = ResonanceConfig(threshold=0.0, temporal_half_life_seconds=1.0)
        new_m = make_test_memory(embedding=[1.0, 0.0, 0.0])
        strong = make_test_memory(embedding=[1.0, 0.0, 0.0])  # perfect semantic match
        weak = make_test_memory(embedding=[0.0, 1.0, 0.0])  # orthogonal
        links = build_resonance_links(new_m, [weak, strong], config)
        if len(links) >= 2:
            assert links[0].strength >= links[1].strength


class TestResonanceProperties:
    """Mathematical invariants for the resonance module."""

    @pytest.mark.parametrize("n_candidates", [1, 5, 10, 25])
    def test_max_links_always_respected(self, n_candidates):
        max_links = 3
        config = ResonanceConfig(threshold=0.0, max_links=max_links)
        new_m = make_test_memory(embedding=[1.0, 0.0])
        candidates = [make_test_memory(embedding=[1.0, 0.0]) for _ in range(n_candidates)]
        links = build_resonance_links(new_m, candidates, config)
        assert len(links) <= max_links

    def test_no_self_links_always(self):
        config = ResonanceConfig(threshold=0.0)
        m = make_test_memory()
        links = build_resonance_links(m, [m, m, m], config)
        assert links == []

    def test_link_strength_in_unit_range(self):
        """All resonance link strengths are in [0, 1]."""
        config = ResonanceConfig(threshold=0.0)
        new_m = make_test_memory(embedding=[1.0, 0.0])
        candidates = [
            make_test_memory(embedding=[1.0, 0.0], valence=0.8),
            make_test_memory(embedding=[0.5, 0.5], valence=-0.5),
            make_test_memory(embedding=[0.0, 1.0], valence=0.0),
        ]
        links = build_resonance_links(new_m, candidates, config)
        for link in links:
            assert 0.0 <= link.strength <= 1.0

    def test_temporal_proximity_symmetric(self):
        """temporal_proximity(t1, t2) == temporal_proximity(t2, t1)."""
        t1 = datetime.now(tz=UTC)
        t2 = t1 + timedelta(hours=3)
        assert math.isclose(temporal_proximity(t1, t2), temporal_proximity(t2, t1))


# ---------------------------------------------------------------------------
# Causal link fix: target_precedes_source
# ---------------------------------------------------------------------------


class TestCausalLinkFix:
    def test_causal_link_is_reachable(self):
        """Encoding a new memory against an older one can produce a causal link.

        Before the fix, source_before_target was inverted so causal links were
        almost never assigned. Now target_precedes_source is correct: existing
        memories (older) precede the new memory being encoded.
        """
        config = ResonanceConfig(
            threshold=0.0,
            semantic_weight=1.0,
            emotional_weight=0.0,
            temporal_weight=0.0,
            causal_temporal_threshold=0.0,  # disable temporal gate
            causal_semantic_threshold=0.5,
            temporal_half_life_seconds=3600.0,
        )
        # existing memory encoded 1 hour ago — it precedes the new memory
        old = make_test_memory(embedding=[1.0, 0.0, 0.0], offset_seconds=3600)
        # new memory has current timestamp (default offset=0)
        new_m = make_test_memory(embedding=[1.0, 0.0, 0.0], offset_seconds=0)
        links = build_resonance_links(new_m, [old], config)
        assert len(links) == 1
        assert links[0].link_type == "causal"

    def test_configurable_thresholds_affect_classification(self):
        """Custom contrastive threshold changes link type assignment."""
        # Very high contrastive threshold: valence diff > 1.9 is impossible for ±1 range.
        # Causal disabled by requiring semantic_sim > 1.0 (impossible).
        config_strict = ResonanceConfig(
            threshold=0.0,
            contrastive_valence_threshold=1.9,
            contrastive_temporal_threshold=0.0,
            causal_temporal_threshold=0.0,
            causal_semantic_threshold=1.0,  # semantic_sim > 1.0 never true → no causal
            semantic_weight=0.0,
            emotional_weight=1.0,
            temporal_weight=0.0,
        )
        # Use large offset so temporal_prox ≈ 0 (half-life=3600s, offset=100_000s → prox≈0.001)
        # ensuring the emotional_sim fallback dominates the argmax, not temporal.
        pos = make_test_memory(valence=0.9, arousal=0.5, offset_seconds=100_000)
        neg = make_test_memory(valence=-0.9, arousal=0.5, offset_seconds=0)
        links = build_resonance_links(neg, [pos], config_strict)
        assert len(links) == 1
        # With high contrastive threshold, should fall through to emotional
        assert links[0].link_type == "emotional"

        # Default threshold: valence diff of 1.8 > 1.0 → contrastive
        config_default = ResonanceConfig(
            threshold=0.0,
            contrastive_temporal_threshold=0.0,
            causal_temporal_threshold=0.0,
            causal_semantic_threshold=1.0,  # semantic_sim > 1.0 never true → no causal
            semantic_weight=0.0,
            emotional_weight=1.0,
            temporal_weight=0.0,
        )
        links2 = build_resonance_links(neg, [pos], config_default)
        assert len(links2) == 1
        assert links2[0].link_type == "contrastive"


# ---------------------------------------------------------------------------
# Spreading activation
# ---------------------------------------------------------------------------


def _mem_with_link(source_id: str, target_id: str, strength: float) -> Memory:
    """Create a test memory pre-loaded with one resonance link."""
    tag = make_emotional_tag(
        core_affect=CoreAffect.neutral(),
        momentum=AffectiveMomentum.zero(),
        mood=MoodField.neutral(),
        consolidation_strength=0.5,
    )
    link = ResonanceLink(
        source_id=source_id, target_id=target_id, strength=strength, link_type="semantic"
    )
    tag = tag.model_copy(update={"resonance_links": [link]})
    from emotional_memory.models import Memory as _Mem

    return _Mem(id=source_id, content="test", tag=tag)


class TestSpreadingActivation:
    def test_empty_seeds_returns_empty(self):
        candidates = [make_test_memory()]
        result = spreading_activation(set(), candidates, hops=2)
        assert result == {}

    def test_empty_candidates_returns_empty(self):
        result = spreading_activation({"a"}, [], hops=2)
        assert result == {}

    def test_no_links_returns_empty(self):
        """Memories without links produce no spreading."""
        m1 = make_test_memory()
        m2 = make_test_memory()
        result = spreading_activation({m1.id}, [m1, m2], hops=2)
        assert result == {}

    def test_one_hop_direct_neighbour(self):
        """Seed → A: A gets activation = seed_activation * link_strength."""
        m_a = make_test_memory()
        m_seed = _mem_with_link("seed", m_a.id, 0.8)
        # Override id so it matches
        m_seed = m_seed.model_copy(update={"id": "seed"})
        candidates = [m_seed, m_a]
        result = spreading_activation({"seed"}, candidates, hops=1)
        assert m_a.id in result
        assert math.isclose(result[m_a.id], 0.8, rel_tol=1e-6)

    def test_two_hop_indirect_neighbour(self):
        """Seed → A → B: B receives activation decayed through two hops."""
        m_b = make_test_memory()
        m_a = _mem_with_link("node_a", m_b.id, 0.9)
        m_seed = _mem_with_link("seed", "node_a", 0.8)
        m_seed = m_seed.model_copy(update={"id": "seed"})
        m_a = m_a.model_copy(update={"id": "node_a"})
        candidates = [m_seed, m_a, m_b]
        result = spreading_activation({"seed"}, candidates, hops=2)
        assert "node_a" in result
        assert m_b.id in result
        # B should have activation ≈ 0.8 * 0.9 = 0.72
        assert math.isclose(result[m_b.id], 0.72, rel_tol=1e-6)

    def test_two_hop_not_reached_with_one_hop(self):
        """B is 2 hops from seed — hops=1 should not reach B."""
        m_b = make_test_memory()
        m_a = _mem_with_link("node_a", m_b.id, 0.9)
        m_seed = _mem_with_link("seed", "node_a", 0.8)
        m_seed = m_seed.model_copy(update={"id": "seed"})
        m_a = m_a.model_copy(update={"id": "node_a"})
        candidates = [m_seed, m_a, m_b]
        result = spreading_activation({"seed"}, candidates, hops=1)
        assert m_b.id not in result  # only 1 hop: B not reached

    def test_seeds_excluded_from_result(self):
        """Seeds themselves must not appear in the activation map."""
        m_a = make_test_memory()
        m_seed = _mem_with_link("seed", m_a.id, 0.7)
        m_seed = m_seed.model_copy(update={"id": "seed"})
        result = spreading_activation({"seed"}, [m_seed, m_a], hops=2)
        assert "seed" not in result

    def test_activation_decays_with_hops(self):
        """Activation level at hop 2 is lower than at hop 1 (strength < 1)."""
        m_b = make_test_memory()
        m_a = _mem_with_link("node_a", m_b.id, 0.6)
        m_seed = _mem_with_link("seed", "node_a", 0.6)
        m_seed = m_seed.model_copy(update={"id": "seed"})
        m_a = m_a.model_copy(update={"id": "node_a"})
        candidates = [m_seed, m_a, m_b]
        result = spreading_activation({"seed"}, candidates, hops=2)
        assert result["node_a"] > result[m_b.id]

    def test_activation_capped_at_one(self):
        """Activation can never exceed 1.0."""
        m_target = make_test_memory()
        # Two separate seeds both pointing to the same target at high strength
        seed1 = _mem_with_link("s1", m_target.id, 1.0)
        seed2 = _mem_with_link("s2", m_target.id, 1.0)
        seed1 = seed1.model_copy(update={"id": "s1"})
        seed2 = seed2.model_copy(update={"id": "s2"})
        candidates = [seed1, seed2, m_target]
        result = spreading_activation({"s1", "s2"}, candidates, hops=1)
        assert result.get(m_target.id, 0.0) <= 1.0


# ---------------------------------------------------------------------------
# Hebbian co-retrieval strengthening
# ---------------------------------------------------------------------------


class TestHebbianStrengthen:
    def _mem_with_links(self, own_id: str, target_ids: list[str]) -> Memory:
        tag = make_emotional_tag(
            core_affect=CoreAffect.neutral(),
            momentum=AffectiveMomentum.zero(),
            mood=MoodField.neutral(),
            consolidation_strength=0.5,
        )
        links = [
            ResonanceLink(source_id=own_id, target_id=tid, strength=0.5, link_type="semantic")
            for tid in target_ids
        ]
        tag = tag.model_copy(update={"resonance_links": links})
        from emotional_memory.models import Memory as _Mem

        return _Mem(id=own_id, content="test", tag=tag)

    def test_co_retrieved_link_strengthened(self):
        mem = self._mem_with_links("A", ["B", "C"])
        result = hebbian_strengthen(mem, {"B"}, increment=0.1)
        b_link = next(lnk for lnk in result if lnk.target_id == "B")
        c_link = next(lnk for lnk in result if lnk.target_id == "C")
        assert math.isclose(b_link.strength, 0.6, rel_tol=1e-6)
        assert math.isclose(c_link.strength, 0.5, rel_tol=1e-6)  # unchanged

    def test_strength_capped_at_one(self):
        tag = make_emotional_tag(
            core_affect=CoreAffect.neutral(),
            momentum=AffectiveMomentum.zero(),
            mood=MoodField.neutral(),
            consolidation_strength=0.5,
        )
        link = ResonanceLink(source_id="A", target_id="B", strength=0.95, link_type="semantic")
        tag = tag.model_copy(update={"resonance_links": [link]})
        from emotional_memory.models import Memory as _Mem

        mem = _Mem(id="A", content="test", tag=tag)
        result = hebbian_strengthen(mem, {"B"}, increment=0.5)
        assert result[0].strength == pytest.approx(1.0)

    def test_no_co_retrieved_ids_returns_unchanged(self):
        mem = self._mem_with_links("A", ["B"])
        result = hebbian_strengthen(mem, set(), increment=0.1)
        assert result == list(mem.tag.resonance_links)

    def test_zero_increment_returns_unchanged(self):
        mem = self._mem_with_links("A", ["B"])
        result = hebbian_strengthen(mem, {"B"}, increment=0.0)
        assert result == list(mem.tag.resonance_links)

    def test_no_matching_links_returns_unchanged(self):
        """When co_retrieved_ids don't match any link targets, links are unchanged."""
        mem = self._mem_with_links("A", ["B", "C"])
        result = hebbian_strengthen(mem, {"D", "E"}, increment=0.2)
        assert result == list(mem.tag.resonance_links)

    def test_empty_links_returns_original(self):
        tag = make_emotional_tag(
            core_affect=CoreAffect.neutral(),
            momentum=AffectiveMomentum.zero(),
            mood=MoodField.neutral(),
            consolidation_strength=0.5,
        )
        from emotional_memory.models import Memory as _Mem

        mem = _Mem(id="A", content="test", tag=tag)
        result = hebbian_strengthen(mem, {"B"}, increment=0.1)
        assert result == []
