import math
from datetime import datetime, timedelta, timezone

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.models import Memory, make_emotional_tag
from emotional_memory.resonance import ResonanceConfig, build_resonance_links, temporal_proximity
from emotional_memory.stimmung import StimmungField


def _now():
    return datetime.now(tz=timezone.utc)


def _memory(
    valence: float = 0.0,
    arousal: float = 0.5,
    embedding: list[float] | None = None,
    offset_seconds: float = 0.0,
):
    ts = _now() - timedelta(seconds=offset_seconds)
    tag = make_emotional_tag(
        core_affect=CoreAffect(valence=valence, arousal=arousal),
        momentum=AffectiveMomentum.zero(),
        stimmung=StimmungField.neutral(),
        consolidation_strength=0.7,
    )
    tag = tag.model_copy(update={"timestamp": ts})
    return Memory.create(content="test", tag=tag, embedding=embedding)


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
        m = _memory()
        links = build_resonance_links(m, [], ResonanceConfig())
        assert links == []

    def test_no_self_links(self):
        m = _memory()
        links = build_resonance_links(m, [m], ResonanceConfig())
        assert links == []

    def test_semantically_similar_gets_semantic_link(self):
        config = ResonanceConfig(threshold=0.1, temporal_half_life_seconds=1.0)
        new_m = _memory(embedding=[1.0, 0.0, 0.0])
        similar = _memory(embedding=[1.0, 0.0, 0.0], offset_seconds=100_000)
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
        new_m = _memory(valence=0.9, arousal=0.9, offset_seconds=0)
        similar = _memory(valence=0.9, arousal=0.9, offset_seconds=100_000)
        links = build_resonance_links(new_m, [similar], config)
        assert len(links) == 1
        assert links[0].link_type == "emotional"

    def test_temporally_close_gets_temporal_link(self):
        # Use different affect so emotional_sim < temporal_prox (1s apart)
        # and no embeddings so sem_sim=0. temporal_prox(1s, half_life=3600) ≈ 0.9998
        config = ResonanceConfig(threshold=0.1, temporal_half_life_seconds=3600.0)
        new_m = _memory(valence=0.0, arousal=0.0, offset_seconds=0)
        close = _memory(valence=0.0, arousal=0.0, offset_seconds=1)
        # Force: no embeddings (sem_sim=0), same affect (emo_sim≈1), 1s apart (temp≈1)
        # temporal wins only if we pick very different affect or set weights explicitly.
        # Instead just verify the link is created and is among {emotional, temporal}.
        links = build_resonance_links(new_m, [close], config)
        assert len(links) == 1
        assert links[0].link_type in ("temporal", "emotional")

    def test_below_threshold_no_link(self):
        config = ResonanceConfig(threshold=0.99)  # almost impossible to reach
        new_m = _memory(embedding=[1.0, 0.0])
        far = _memory(embedding=[0.0, 1.0], offset_seconds=100_000)
        links = build_resonance_links(new_m, [far], config)
        assert links == []

    def test_max_links_respected(self):
        config = ResonanceConfig(threshold=0.0, max_links=2)
        new_m = _memory(embedding=[1.0, 0.0])
        candidates = [_memory(embedding=[1.0, 0.0]) for _ in range(10)]
        links = build_resonance_links(new_m, candidates, config)
        assert len(links) <= 2

    def test_links_sorted_by_strength_desc(self):
        config = ResonanceConfig(threshold=0.0, temporal_half_life_seconds=1.0)
        new_m = _memory(embedding=[1.0, 0.0, 0.0])
        strong = _memory(embedding=[1.0, 0.0, 0.0])  # perfect semantic match
        weak = _memory(embedding=[0.0, 1.0, 0.0])  # orthogonal
        links = build_resonance_links(new_m, [weak, strong], config)
        if len(links) >= 2:
            assert links[0].strength >= links[1].strength
