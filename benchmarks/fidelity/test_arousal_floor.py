"""Fidelity benchmark: Arousal floor — high-arousal memories resist full decay.

Hypothesis: memories encoded under high arousal never fully fade; they
maintain a minimum strength floor (Merleau-Ponty's body memory — habitual
emotional patterns persist).

Reference: McGaugh, J.L. (2004). Modulation of memory storage processes.
"""

from datetime import UTC, datetime, timedelta

import pytest

from emotional_memory import CoreAffect
from emotional_memory.affect import AffectiveMomentum
from emotional_memory.decay import DecayConfig, compute_effective_strength
from emotional_memory.models import make_emotional_tag
from emotional_memory.stimmung import StimmungField

pytestmark = pytest.mark.fidelity


def _ancient_time():
    return datetime.now(tz=UTC) - timedelta(days=3650)  # 10 years ago


def _tag(arousal: float, strength: float = 1.0):
    tag = make_emotional_tag(
        core_affect=CoreAffect(valence=0.0, arousal=arousal),
        momentum=AffectiveMomentum.zero(),
        stimmung=StimmungField.neutral(),
        consolidation_strength=strength,
    )
    return tag.model_copy(update={"timestamp": _ancient_time()})


def test_high_arousal_floor_respected():
    """High-arousal memory retains at least floor_value after extreme time decay."""
    config = DecayConfig(floor_arousal_threshold=0.7, floor_value=0.1, base_decay=1.0)
    tag = _tag(arousal=0.9)
    s = compute_effective_strength(tag, datetime.now(tz=UTC), config)
    assert s >= config.floor_value, (
        f"High-arousal memory should not decay below floor: strength={s:.4f}, "
        f"floor={config.floor_value}"
    )


def test_low_arousal_can_decay_below_floor():
    """Low-arousal memories are NOT protected by the floor."""
    config = DecayConfig(floor_arousal_threshold=0.7, floor_value=0.1, base_decay=2.0)
    tag = _tag(arousal=0.1, strength=0.5)
    s = compute_effective_strength(tag, datetime.now(tz=UTC), config)
    assert s < config.floor_value, (
        f"Low-arousal memory should decay below floor: strength={s:.4f}, "
        f"floor={config.floor_value}"
    )


@pytest.mark.parametrize("arousal", [0.7, 0.8, 0.9, 1.0])
def test_floor_applies_to_all_high_arousal(arousal):
    """Floor applies for any arousal >= floor_arousal_threshold."""
    config = DecayConfig(floor_arousal_threshold=0.7, floor_value=0.15, base_decay=1.5)
    tag = _tag(arousal=arousal)
    s = compute_effective_strength(tag, datetime.now(tz=UTC), config)
    assert s >= config.floor_value, (
        f"arousal={arousal}: expected s >= {config.floor_value}, got {s:.4f}"
    )


def test_floor_asymmetry():
    """The gap between high-arousal and low-arousal strength after long decay is measurable."""
    config = DecayConfig(floor_arousal_threshold=0.7, floor_value=0.1, base_decay=1.0)
    now = datetime.now(tz=UTC)
    s_high = compute_effective_strength(_tag(arousal=0.9), now, config)
    s_low = compute_effective_strength(_tag(arousal=0.1), now, config)

    assert s_high > s_low, (
        f"High-arousal ({s_high:.4f}) should remain above low-arousal ({s_low:.4f}) "
        "after extreme time decay"
    )
