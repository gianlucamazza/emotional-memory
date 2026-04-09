import math

import pytest
from pydantic import ValidationError

from emotional_memory.affect import AffectiveMomentum, CoreAffect


class TestCoreAffect:
    def test_neutral(self):
        ca = CoreAffect.neutral()
        assert ca.valence == 0.0
        assert ca.arousal == 0.0

    def test_clamp_valence_high(self):
        ca = CoreAffect(valence=1.5, arousal=0.5)
        assert ca.valence == 1.0

    def test_clamp_valence_low(self):
        ca = CoreAffect(valence=-2.0, arousal=0.5)
        assert ca.valence == -1.0

    def test_clamp_arousal_high(self):
        ca = CoreAffect(valence=0.0, arousal=1.5)
        assert ca.arousal == 1.0

    def test_clamp_arousal_low(self):
        ca = CoreAffect(valence=0.0, arousal=-0.5)
        assert ca.arousal == 0.0

    def test_valid_range_unchanged(self):
        ca = CoreAffect(valence=0.5, arousal=0.7)
        assert ca.valence == 0.5
        assert ca.arousal == 0.7

    def test_distance_zero_for_equal(self):
        ca = CoreAffect(valence=0.3, arousal=0.6)
        assert ca.distance(ca) == 0.0

    def test_distance_symmetric(self):
        a = CoreAffect(valence=0.0, arousal=0.0)
        b = CoreAffect(valence=1.0, arousal=1.0)
        assert math.isclose(a.distance(b), b.distance(a))

    def test_distance_known_value(self):
        a = CoreAffect(valence=0.0, arousal=0.0)
        b = CoreAffect(valence=1.0, arousal=0.0)
        assert math.isclose(a.distance(b), 1.0)

    def test_lerp_alpha_zero_returns_self(self):
        a = CoreAffect(valence=-0.5, arousal=0.2)
        b = CoreAffect(valence=0.8, arousal=0.9)
        result = a.lerp(b, 0.0)
        assert math.isclose(result.valence, a.valence)
        assert math.isclose(result.arousal, a.arousal)

    def test_lerp_alpha_one_returns_other(self):
        a = CoreAffect(valence=-0.5, arousal=0.2)
        b = CoreAffect(valence=0.8, arousal=0.9)
        result = a.lerp(b, 1.0)
        assert math.isclose(result.valence, b.valence)
        assert math.isclose(result.arousal, b.arousal)

    def test_lerp_midpoint(self):
        a = CoreAffect(valence=0.0, arousal=0.0)
        b = CoreAffect(valence=1.0, arousal=1.0)
        result = a.lerp(b, 0.5)
        assert math.isclose(result.valence, 0.5)
        assert math.isclose(result.arousal, 0.5)

    def test_lerp_clamps_alpha(self):
        a = CoreAffect(valence=0.0, arousal=0.0)
        b = CoreAffect(valence=1.0, arousal=1.0)
        # alpha > 1 clamped to 1
        result = a.lerp(b, 2.0)
        assert math.isclose(result.valence, 1.0)

    def test_frozen(self):
        ca = CoreAffect(valence=0.5, arousal=0.5)
        with pytest.raises(ValidationError):
            ca.valence = 0.9  # type: ignore[misc]


class TestAffectiveMomentum:
    def test_zero(self):
        m = AffectiveMomentum.zero()
        assert m.d_valence == 0.0
        assert m.d_arousal == 0.0
        assert m.dd_valence == 0.0
        assert m.dd_arousal == 0.0

    def test_magnitude_zero(self):
        assert AffectiveMomentum.zero().magnitude() == 0.0

    def test_magnitude_known(self):
        m = AffectiveMomentum(d_valence=3.0, d_arousal=4.0)
        assert math.isclose(m.magnitude(), 5.0)

    def test_defaults(self):
        m = AffectiveMomentum(d_valence=0.1, d_arousal=0.2)
        assert m.dd_valence == 0.0
        assert m.dd_arousal == 0.0
