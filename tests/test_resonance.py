import math
from datetime import datetime, timedelta, timezone

from emotional_memory.resonance import ResonanceConfig, build_resonance_links, temporal_proximity

from conftest import make_test_memory


def _now():
    return datetime.now(tz=timezone.utc)


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
