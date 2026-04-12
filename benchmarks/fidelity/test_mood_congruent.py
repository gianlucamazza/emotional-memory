"""Fidelity benchmark: Mood-congruent memory recall (Bower 1981).

Hypothesis: retrieval under a given mood preferentially surfaces memories
encoded under a congruent mood. The effect must be detectable even after
controlling for semantic similarity (ConstantEmbedder neutralises s1).

Reference: Bower, G.H. (1981). Mood and memory. American Psychologist, 36(2).
"""

import pytest

from benchmarks.conftest import make_fidelity_engine
from emotional_memory import CoreAffect

pytestmark = pytest.mark.fidelity


def _encode_emotional_pairs(engine):
    """Encode 4 positive and 4 negative memories with matching content."""
    positive_affect = CoreAffect(valence=0.9, arousal=0.7)
    negative_affect = CoreAffect(valence=-0.9, arousal=0.6)

    engine.set_affect(positive_affect)
    for i in range(4):
        engine.encode(f"Positive experience {i}: success and joy in work")

    engine.set_affect(negative_affect)
    for i in range(4):
        engine.encode(f"Negative experience {i}: difficulty and frustration")


def test_positive_mood_surfaces_positive_memories():
    """Under positive Mood, positive memories should dominate the top-3."""
    engine = make_fidelity_engine(mood_alpha=0.4)
    _encode_emotional_pairs(engine)

    # Drive Mood strongly positive
    for _ in range(15):
        engine.set_affect(CoreAffect(valence=0.95, arousal=0.7))

    results = engine.retrieve("experience", top_k=3)
    positive_count = sum(1 for m in results if m.tag.core_affect.valence > 0)

    assert positive_count >= 2, (
        f"Expected >=2 positive memories in top-3 under positive mood, "
        f"got {positive_count}. Valences: {[m.tag.core_affect.valence for m in results]}"
    )


def test_negative_mood_surfaces_negative_memories():
    """Under negative Mood, negative memories should dominate the top-3."""
    engine = make_fidelity_engine(mood_alpha=0.4)
    _encode_emotional_pairs(engine)

    # Drive Mood strongly negative
    for _ in range(15):
        engine.set_affect(CoreAffect(valence=-0.95, arousal=0.6))

    results = engine.retrieve("experience", top_k=3)
    negative_count = sum(1 for m in results if m.tag.core_affect.valence < 0)

    assert negative_count >= 2, (
        f"Expected >=2 negative memories in top-3 under negative mood, "
        f"got {negative_count}. Valences: {[m.tag.core_affect.valence for m in results]}"
    )


def test_mood_switch_reverses_retrieval_priority():
    """Switching from positive to negative mood should reverse what surfaces first."""
    engine = make_fidelity_engine(mood_alpha=0.4)
    _encode_emotional_pairs(engine)

    # Retrieve under positive mood
    for _ in range(15):
        engine.set_affect(CoreAffect(valence=0.95, arousal=0.7))
    pos_results = engine.retrieve("experience", top_k=4)
    pos_avg_valence = sum(m.tag.core_affect.valence for m in pos_results) / len(pos_results)

    # Switch to strongly negative mood
    for _ in range(20):
        engine.set_affect(CoreAffect(valence=-0.95, arousal=0.6))
    neg_results = engine.retrieve("experience", top_k=4)
    neg_avg_valence = sum(m.tag.core_affect.valence for m in neg_results) / len(neg_results)

    assert pos_avg_valence > neg_avg_valence, (
        f"Mood switch should reverse retrieval polarity: "
        f"pos_avg={pos_avg_valence:.3f}, neg_avg={neg_avg_valence:.3f}"
    )
