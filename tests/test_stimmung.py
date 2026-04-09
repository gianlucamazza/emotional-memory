import math

import pytest
from pydantic import ValidationError

from emotional_memory.affect import CoreAffect
from emotional_memory.stimmung import StimmungField


class TestStimmungField:
    def test_neutral_values(self):
        s = StimmungField.neutral()
        assert s.valence == 0.0
        assert s.arousal == 0.0
        assert s.dominance == 0.5
        assert s.inertia == 0.5

    def test_frozen(self):
        s = StimmungField.neutral()
        with pytest.raises(ValidationError):
            s.valence = 0.5  # type: ignore[misc]

    def test_clamp_valence(self):
        s = StimmungField(
            valence=2.0,
            arousal=0.5,
            dominance=0.5,
            inertia=0.5,
            timestamp=StimmungField.neutral().timestamp,
        )
        assert s.valence == 1.0

    def test_clamp_arousal(self):
        s = StimmungField(
            valence=0.0,
            arousal=-1.0,
            dominance=0.5,
            inertia=0.5,
            timestamp=StimmungField.neutral().timestamp,
        )
        assert s.arousal == 0.0

    def test_update_moves_toward_affect(self):
        s = StimmungField.neutral()
        positive = CoreAffect(valence=1.0, arousal=1.0)
        s2 = s.update(positive, alpha=0.1)
        # With inertia=0.5, effective_alpha = 0.1 * 0.5 = 0.05
        assert s2.valence > s.valence
        assert s2.arousal > s.arousal

    def test_update_returns_new_instance(self):
        s = StimmungField.neutral()
        s2 = s.update(CoreAffect(valence=1.0, arousal=1.0))
        assert s is not s2

    def test_high_inertia_resists_change(self):
        high = StimmungField(
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
            inertia=0.9,
            timestamp=StimmungField.neutral().timestamp,
        )
        low = StimmungField(
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
            inertia=0.1,
            timestamp=StimmungField.neutral().timestamp,
        )
        affect = CoreAffect(valence=1.0, arousal=1.0)
        assert high.update(affect).valence < low.update(affect).valence

    def test_repeated_positive_updates_drift_positive(self):
        s = StimmungField.neutral()
        affect = CoreAffect(valence=1.0, arousal=0.8)
        for _ in range(50):
            s = s.update(affect, alpha=0.2)
        assert s.valence > 0.5

    def test_distance_zero_for_equal(self):
        s = StimmungField.neutral()
        assert s.distance(s) == 0.0

    def test_distance_symmetric(self):
        a = StimmungField.neutral()
        b = StimmungField(
            valence=0.5, arousal=0.5, dominance=0.8, inertia=0.3, timestamp=a.timestamp
        )
        assert math.isclose(a.distance(b), b.distance(a))

    def test_zero_inertia_follows_affect_fully(self):
        # inertia=0 → effective_alpha = alpha * 1.0 = alpha
        s = StimmungField(
            valence=0.0,
            arousal=0.0,
            dominance=0.5,
            inertia=0.0,
            timestamp=StimmungField.neutral().timestamp,
        )
        affect = CoreAffect(valence=1.0, arousal=1.0)
        s2 = s.update(affect, alpha=0.5)
        assert math.isclose(s2.valence, 0.5)  # 0.5 * 0 + 0.5 * 1 = 0.5
