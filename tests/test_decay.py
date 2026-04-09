from datetime import UTC, datetime, timedelta

import pytest

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.decay import DecayConfig, compute_effective_strength
from emotional_memory.models import make_emotional_tag
from emotional_memory.stimmung import StimmungField


def _tag(arousal: float = 0.5, strength: float = 0.8, retrieval_count: int = 0):
    tag = make_emotional_tag(
        core_affect=CoreAffect(valence=0.0, arousal=arousal),
        momentum=AffectiveMomentum.zero(),
        stimmung=StimmungField.neutral(),
        consolidation_strength=strength,
    )
    return tag.model_copy(update={"retrieval_count": retrieval_count})


def _now():
    return datetime.now(tz=UTC)


def _later(seconds: float):
    return _now() + timedelta(seconds=seconds)


class TestComputeEffectiveStrength:
    def test_strength_decreases_over_time(self):
        tag = _tag(arousal=0.5, strength=1.0)
        config = DecayConfig()
        s1 = compute_effective_strength(tag, _later(10), config)
        s2 = compute_effective_strength(tag, _later(1000), config)
        assert s1 > s2

    def test_high_arousal_decays_slower(self):
        config = DecayConfig()
        later = _later(500)
        high = compute_effective_strength(_tag(arousal=0.9), later, config)
        low = compute_effective_strength(_tag(arousal=0.1), later, config)
        assert high > low

    def test_more_retrievals_decay_slower(self):
        config = DecayConfig()
        later = _later(500)
        many = compute_effective_strength(_tag(retrieval_count=10), later, config)
        zero = compute_effective_strength(_tag(retrieval_count=0), later, config)
        assert many > zero

    def test_floor_for_high_arousal(self):
        config = DecayConfig(floor_arousal_threshold=0.7, floor_value=0.1)
        tag = _tag(arousal=0.9, strength=0.9)
        # Very far in the future
        very_late = _later(10_000_000)
        s = compute_effective_strength(tag, very_late, config)
        assert s >= config.floor_value

    def test_no_floor_for_low_arousal(self):
        config = DecayConfig(floor_arousal_threshold=0.7, floor_value=0.1, base_decay=2.0)
        tag = _tag(arousal=0.1, strength=0.5)
        very_late = _later(10_000_000)
        s = compute_effective_strength(tag, very_late, config)
        assert s < config.floor_value

    def test_strength_at_min_seconds_near_initial(self):
        config = DecayConfig(min_seconds=1.0, base_decay=0.1)
        tag = _tag(strength=0.8)
        # elapsed = min_seconds = 1.0 → strength ≈ 0.8 * 1^(-0.x) ≈ 0.8
        s = compute_effective_strength(tag, _later(0), config)
        assert abs(s - 0.8) < 0.1

    def test_strength_clamped_to_unit(self):
        config = DecayConfig(base_decay=0.0)
        tag = _tag(strength=1.0)
        s = compute_effective_strength(tag, _later(1), config)
        assert 0.0 <= s <= 1.0


class TestDecayProperties:
    """Mathematical invariants that must hold for all parameter combinations."""

    @pytest.mark.parametrize("arousal", [0.0, 0.2, 0.5, 0.7, 0.9, 1.0])
    def test_decay_monotonic(self, arousal):
        """Strength strictly decreases (or stays constant) over increasing time."""
        config = DecayConfig()
        tag = _tag(arousal=arousal, strength=1.0)
        times = [10, 100, 1_000, 10_000]
        strengths = [compute_effective_strength(tag, _later(t), config) for t in times]
        for i in range(len(strengths) - 1):
            assert strengths[i] >= strengths[i + 1] - 1e-9, (
                f"arousal={arousal}: non-monotone at t={times[i]}→{times[i + 1]}: "
                f"{strengths[i]:.4f} > {strengths[i + 1]:.4f}"
            )

    @pytest.mark.parametrize("arousal", [0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_effective_strength_bounded(self, arousal):
        """Effective strength is always in [0.0, 1.0]."""
        config = DecayConfig()
        tag = _tag(arousal=arousal, strength=0.8)
        for t in [1, 100, 10_000, 1_000_000]:
            s = compute_effective_strength(tag, _later(t), config)
            assert 0.0 <= s <= 1.0, f"arousal={arousal}, t={t}: s={s:.4f} out of range"

    def test_power_exponent_accelerates_decay(self):
        """power=2.0 decays faster than power=1.0 at the same elapsed time."""
        later = _later(1_000)
        tag = _tag(arousal=0.5, strength=1.0)
        s_normal = compute_effective_strength(tag, later, DecayConfig(power=1.0))
        s_fast = compute_effective_strength(tag, later, DecayConfig(power=2.0))
        assert s_fast < s_normal, (
            f"power=2.0 ({s_fast:.4f}) should decay faster than power=1.0 ({s_normal:.4f})"
        )

    @pytest.mark.parametrize("retrieval_count", [0, 1, 5, 10])
    def test_more_retrievals_always_slower_decay(self, retrieval_count):
        """One extra retrieval never makes decay faster."""
        config = DecayConfig(retrieval_boost=0.1)
        later = _later(1_000)
        tag_n = _tag(retrieval_count=retrieval_count)
        tag_n1 = _tag(retrieval_count=retrieval_count + 1)
        s_n = compute_effective_strength(tag_n, later, config)
        s_n1 = compute_effective_strength(tag_n1, later, config)
        assert s_n1 >= s_n - 1e-9, (
            f"count={retrieval_count + 1} ({s_n1:.4f}) should be "
            f">= count={retrieval_count} ({s_n:.4f})"
        )
