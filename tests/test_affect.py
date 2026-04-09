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


class TestCoreAffectProperties:
    """Mathematical invariants for CoreAffect."""

    @pytest.mark.parametrize(
        "valence,arousal",
        [(-5.0, 3.0), (2.0, -1.0), (0.5, 0.5), (-1.0, 0.0), (1.0, 1.0)],
    )
    def test_always_clamped(self, valence, arousal):
        ca = CoreAffect(valence=valence, arousal=arousal)
        assert -1.0 <= ca.valence <= 1.0
        assert 0.0 <= ca.arousal <= 1.0

    @pytest.mark.parametrize(
        "v1,v2",
        [(0.0, 0.5), (-0.5, 0.5), (0.3, 0.8), (-0.9, -0.4)],
    )
    def test_positive_velocity_when_valence_increases(self, v1, v2):
        """After two updates where valence rises, d_valence must be positive."""
        from emotional_memory.state import AffectiveState

        state = AffectiveState.initial()
        state = state.update(CoreAffect(valence=v1, arousal=0.5))
        state = state.update(CoreAffect(valence=v2, arousal=0.5))
        assert state.momentum.d_valence > 0.0, (
            f"v1={v1}→v2={v2}: expected positive d_valence, got {state.momentum.d_valence:.3f}"
        )

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_lerp_output_in_range(self, alpha):
        a = CoreAffect(valence=-1.0, arousal=0.0)
        b = CoreAffect(valence=1.0, arousal=1.0)
        c = a.lerp(b, alpha)
        assert -1.0 <= c.valence <= 1.0
        assert 0.0 <= c.arousal <= 1.0
