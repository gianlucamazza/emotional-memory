"""Fidelity benchmark: Dominance retrieval gap (G7 design specification).

This test documents the discriminating scenario that a 3D (valence x arousal x
dominance) CoreAffect implementation must satisfy.  It is currently marked
``xfail(strict=True)`` because ``CoreAffect`` is 2-dimensional; dominance lives
only in ``MoodField`` and is not a primary retrieval signal.

When v0.8.0 promotes dominance to a ``CoreAffect`` dimension, this test MUST
pass — it serves as the acceptance criterion for that change.

Theory
------
Mehrabian, A., & Russell, J. A. (1974). An approach to environmental
psychology. MIT Press.
PAD predicts that dominance (perceived control) is orthogonal to the
valence-arousal plane and provides independent discriminative power when
two memories share mood but differ in situational control.
"""

from __future__ import annotations

import pytest

from benchmarks.conftest import make_fidelity_engine
from emotional_memory import CoreAffect

pytestmark = pytest.mark.fidelity

# Shared affect: moderate negative valence, high arousal — same for both memories
# so they are indistinguishable in the current 2D space.
_SHARED_VALENCE = -0.4
_SHARED_AROUSAL = 0.7


def _setup_dominance_scenario(engine):
    """Encode two memories with identical CoreAffect but opposite dominance context."""
    shared = CoreAffect(valence=_SHARED_VALENCE, arousal=_SHARED_AROUSAL)
    engine.set_affect(shared)
    engine.encode(
        "I seized control of the heated argument and redirected the discussion.",
        metadata={"dominance": "high"},
    )
    engine.set_affect(shared)
    engine.encode(
        "I felt completely helpless as the situation spiralled beyond my reach.",
        metadata={"dominance": "low"},
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CoreAffect is 2D (valence x arousal); dominance promotion to CoreAffect "
        "required for dominance-aware retrieval — targeted for v0.8.0. "
        "See docs/research/11_dominance_design.md section 4."
    ),
)
def test_high_dominance_query_prefers_high_dominance_memory():
    """A high-dominance query state should rank the assertive memory above the helpless one.

    Currently FAILS because both memories receive identical CoreAffect fingerprints
    (same valence, same arousal).  Once dominance is a primary CoreAffect dimension
    this test must pass.
    """
    engine = make_fidelity_engine(mood_alpha=0.4)
    _setup_dominance_scenario(engine)

    # Drive the engine toward a high-dominance state.
    # Until CoreAffect gains a dominance field, there is no dedicated signal for this.
    # We approximate via positive valence / high arousal (→ high MoodField dominance)
    # but this is NOT discriminating because both memories were encoded at the same
    # valence/arousal.
    for _ in range(10):
        engine.set_affect(CoreAffect(valence=0.8, arousal=0.9))  # high dominance signal

    results = engine.retrieve("I want to remember when I was in control", top_k=2)
    assert len(results) == 2

    top_memory = results[0]
    assert "seized control" in top_memory.content, (
        f"Expected high-dominance memory at rank 1, got: {top_memory.content!r}\n"
        "This failure is EXPECTED until CoreAffect gains a dominance dimension (v0.8.0)."
    )


def test_dominance_gap_is_current_limitation():
    """Confirms that the current 2D system cannot distinguish memories by dominance.

    This test PASSES (documents the known gap): both memories share identical
    CoreAffect and therefore have the same affect-proximity retrieval score.
    """
    engine = make_fidelity_engine()
    _setup_dominance_scenario(engine)

    # Retrieve both memories; their affect scores are identical
    results = engine.retrieve("control and helplessness", top_k=2)
    assert len(results) == 2

    # Both memories encode at the same CoreAffect — scores should be very close
    # (differences arise only from content-based semantic similarity, not from affect)
    contents = {r.content for r in results}
    assert any("seized control" in c for c in contents), "High-dominance memory not retrieved"
    assert any("helpless" in c for c in contents), "Low-dominance memory not retrieved"
