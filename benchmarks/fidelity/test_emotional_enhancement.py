"""Fidelity benchmark: Emotional enhancement of memory consolidation.

Hypothesis: high-arousal events are consolidated more strongly than
low-arousal events, producing more durable memories.

References:
  - Cahill, L. & McGaugh, J.L. (1995). A novel demonstration of enhanced
    memory associated with emotional arousal. Consciousness and Cognition.
  - McGaugh, J.L. (2004). The amygdala modulates the consolidation of
    memories of emotionally arousing experiences. Annual Review of Neuroscience.
"""

import pytest

from benchmarks.conftest import make_fidelity_engine
from emotional_memory import CoreAffect
from emotional_memory.appraisal import consolidation_strength

pytestmark = pytest.mark.fidelity


def test_high_arousal_yields_stronger_consolidation():
    """High-arousal memories are consolidated more strongly than low-arousal."""
    engine = make_fidelity_engine()

    # Encode high-arousal memory
    engine.set_affect(CoreAffect(valence=0.3, arousal=0.9))
    high = engine.encode("Critical moment: everything was at stake, heart pounding.")

    # Encode low-arousal memory
    engine.set_affect(CoreAffect(valence=0.3, arousal=0.1))
    low = engine.encode("Routine afternoon: calm and uneventful administrative work.")

    assert high.tag.consolidation_strength > low.tag.consolidation_strength, (
        f"High-arousal consolidation ({high.tag.consolidation_strength:.3f}) "
        f"should exceed low-arousal ({low.tag.consolidation_strength:.3f})"
    )


def test_emotional_enhancement_peaks_at_mid_arousal():
    """Consolidation follows an inverted-U: peak near arousal=0.7, not at max."""
    # With mood_arousal = 0, effective_arousal = 0.7 * encoding_arousal
    # Peak at effective_arousal = 0.7 → encoding_arousal = 1.0
    # (Because 0.7 * 1.0 = 0.7 exactly)
    # Mid-high arousal (0.7) vs max arousal (1.0) vs low (0.0)
    mood_arousal = 0.7  # fix mood to make effective_arousal predictable

    s_low = consolidation_strength(0.0, mood_arousal)
    s_mid = consolidation_strength(0.7, mood_arousal)
    s_max = consolidation_strength(1.0, mood_arousal)

    # The inverted-U: neither extreme should beat the mid-range
    assert s_mid >= s_low, f"Mid arousal should beat low: {s_mid:.3f} vs {s_low:.3f}"
    assert s_mid >= s_max, f"Mid arousal should beat max: {s_mid:.3f} vs {s_max:.3f}"


def test_arousal_modulates_decay():
    """High-arousal memories remain stronger over time than low-arousal ones."""
    from datetime import UTC, datetime, timedelta

    from emotional_memory.affect import AffectiveMomentum
    from emotional_memory.decay import DecayConfig, compute_effective_strength
    from emotional_memory.models import make_emotional_tag
    from emotional_memory.mood import MoodField

    config = DecayConfig()
    now = datetime.now(tz=UTC)
    later = now + timedelta(days=7)

    tag_high = make_emotional_tag(
        core_affect=CoreAffect(valence=0.0, arousal=0.9),
        momentum=AffectiveMomentum.zero(),
        mood=MoodField.neutral(),
        consolidation_strength=1.0,
    )
    tag_high = tag_high.model_copy(update={"timestamp": now})

    tag_low = make_emotional_tag(
        core_affect=CoreAffect(valence=0.0, arousal=0.1),
        momentum=AffectiveMomentum.zero(),
        mood=MoodField.neutral(),
        consolidation_strength=1.0,
    )
    tag_low = tag_low.model_copy(update={"timestamp": now})

    s_high = compute_effective_strength(tag_high, later, config)
    s_low = compute_effective_strength(tag_low, later, config)

    assert s_high > s_low, (
        f"High-arousal memory should be stronger after 7 days: high={s_high:.3f}, low={s_low:.3f}"
    )
