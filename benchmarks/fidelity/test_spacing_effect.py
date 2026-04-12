"""Fidelity benchmark: Spacing effect on memory consolidation.

Hypothesis: memories that are retrieved multiple times decay more slowly
than memories retrieved rarely, due to the spacing effect (Ebbinghaus 1885).
Each retrieval reduces the effective decay exponent via retrieval_boost.

Reference: Ebbinghaus, H. (1885). Über das Gedächtnis.
"""

import pytest

from benchmarks.conftest import make_fidelity_engine
from emotional_memory import CoreAffect
from emotional_memory.affect import AffectiveMomentum
from emotional_memory.decay import DecayConfig, compute_effective_strength
from emotional_memory.models import make_emotional_tag
from emotional_memory.mood import MoodField

pytestmark = pytest.mark.fidelity


def test_multiple_retrievals_slow_decay():
    """A memory retrieved N times decays more slowly than one retrieved 0 times."""
    from datetime import UTC, datetime, timedelta

    config = DecayConfig(base_decay=0.5, retrieval_boost=0.1)
    now = datetime.now(tz=UTC)
    later = now + timedelta(days=30)

    tag_base = make_emotional_tag(
        core_affect=CoreAffect(valence=0.0, arousal=0.5),
        momentum=AffectiveMomentum.zero(),
        mood=MoodField.neutral(),
        consolidation_strength=1.0,
    )
    tag_base = tag_base.model_copy(update={"timestamp": now})

    # Never retrieved
    tag_zero = tag_base.model_copy(update={"retrieval_count": 0})
    # Retrieved 10 times (spaced practice)
    tag_many = tag_base.model_copy(update={"retrieval_count": 10})

    s_zero = compute_effective_strength(tag_zero, later, config)
    s_many = compute_effective_strength(tag_many, later, config)

    assert s_many > s_zero, (
        f"Spaced-practice memory should be stronger: "
        f"10x retrieved={s_many:.3f}, never retrieved={s_zero:.3f}"
    )


@pytest.mark.parametrize("retrieval_count", [0, 1, 3, 5, 10])
def test_more_retrievals_monotonically_slower_decay(retrieval_count):
    """Effective strength is monotonically non-decreasing with retrieval_count."""
    from datetime import UTC, datetime, timedelta

    config = DecayConfig(base_decay=0.5, retrieval_boost=0.1)
    now = datetime.now(tz=UTC)
    later = now + timedelta(days=14)

    tag_base = make_emotional_tag(
        core_affect=CoreAffect(valence=0.0, arousal=0.5),
        momentum=AffectiveMomentum.zero(),
        mood=MoodField.neutral(),
        consolidation_strength=1.0,
    )
    tag_base = tag_base.model_copy(update={"timestamp": now})

    tag_n = tag_base.model_copy(update={"retrieval_count": retrieval_count})
    tag_n1 = tag_base.model_copy(update={"retrieval_count": retrieval_count + 1})

    s_n = compute_effective_strength(tag_n, later, config)
    s_n1 = compute_effective_strength(tag_n1, later, config)

    assert s_n1 >= s_n - 1e-9, (
        f"count={retrieval_count + 1} ({s_n1:.4f}) should be "
        f">= count={retrieval_count} ({s_n:.4f})"
    )


def test_spacing_effect_via_engine_retrieve():
    """retrieve() increments retrieval_count, slowing future decay."""
    engine = make_fidelity_engine(ape_threshold=10.0)  # disable reconsolidation

    engine.set_affect(CoreAffect(valence=0.0, arousal=0.5))
    engine.encode("Repeatedly practiced skill that improves with each session")

    # Retrieve 5 times to simulate spaced practice
    for _ in range(5):
        results = engine.retrieve("practiced skill", top_k=1)

    mem = results[0]
    assert mem.tag.retrieval_count >= 5, (
        f"Expected retrieval_count >= 5, got {mem.tag.retrieval_count}"
    )
