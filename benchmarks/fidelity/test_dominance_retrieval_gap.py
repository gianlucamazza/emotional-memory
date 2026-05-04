"""Fidelity benchmark: Dominance retrieval gap (G7).

CoreAffect is now 3D (valence x arousal x dominance — PAD model).  A high-dominance
query state must rank the assertive memory above the helpless one even when both
memories share identical valence and arousal at encoding.

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
# so they are indistinguishable on the valence-arousal plane alone.
_SHARED_VALENCE = -0.4
_SHARED_AROUSAL = 0.7


def _setup_dominance_scenario(engine):
    """Encode two memories with identical valence/arousal but opposite dominance."""
    engine.set_affect(CoreAffect(valence=_SHARED_VALENCE, arousal=_SHARED_AROUSAL, dominance=0.9))
    engine.encode(
        "I seized control of the heated argument and redirected the discussion.",
        metadata={"dominance": "high"},
    )
    engine.set_affect(CoreAffect(valence=_SHARED_VALENCE, arousal=_SHARED_AROUSAL, dominance=0.1))
    engine.encode(
        "I felt completely helpless as the situation spiralled beyond my reach.",
        metadata={"dominance": "low"},
    )


def test_high_dominance_query_prefers_high_dominance_memory():
    """A high-dominance query state ranks the assertive memory above the helpless one.

    Both memories share the same valence and arousal — only dominance differs.
    The retrieval signal s3 (affect_proximity) now operates in 3D PAD space,
    so the high-dominance query state correctly prefers the high-dominance memory.
    """
    engine = make_fidelity_engine(mood_alpha=0.4)
    _setup_dominance_scenario(engine)

    for _ in range(10):
        engine.set_affect(CoreAffect(valence=0.8, arousal=0.9, dominance=0.9))

    results = engine.retrieve("I want to remember when I was in control", top_k=2)
    assert len(results) == 2

    top_memory = results[0]
    assert "seized control" in top_memory.content, (
        f"Expected high-dominance memory at rank 1, got: {top_memory.content!r}"
    )
