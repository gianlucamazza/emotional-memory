"""Fidelity benchmark: APE-gated reconsolidation window (Nader & Schiller, 2000).

Validates that the reconsolidation lability window is opened only by a
high-APE retrieval, not by any retrieval.  This corrects the previously
documented discrepancy between the design docs and the implementation.

Reference:
  Nader, K., Schafe, G. E., & Le Doux, J. E. (2000). Fear memories require
    protein synthesis in the amygdala for reconsolidation after retrieval.
    Nature, 406(6797), 722-726.
  Lee, J. L. C. (2009). Reconsolidation: maintaining memory relevance.
    Trends in Neurosciences, 32(8), 413-420.
"""

import pytest

from benchmarks.conftest import make_fidelity_engine
from emotional_memory import CoreAffect

pytestmark = pytest.mark.fidelity


def test_low_ape_retrieval_does_not_open_window():
    """A retrieval with low APE should not open the reconsolidation window.

    The key fix: previously ANY retrieval set last_retrieved and the window
    was effectively always open.  Now window_opened_at is only set when
    APE > ape_threshold.
    """
    engine = make_fidelity_engine(ape_threshold=0.5)
    engine.set_affect(CoreAffect(valence=0.5, arousal=0.6))
    engine.encode("Memory encoded under moderate positive affect.")

    # Retrieve under identical affect → APE ≈ 0 << 0.5
    engine.set_affect(CoreAffect(valence=0.5, arousal=0.6))
    results = engine.retrieve("moderate memory", top_k=1)

    assert results[0].tag.window_opened_at is None, (
        "Low-APE retrieval must NOT open the reconsolidation window"
    )
    assert results[0].tag.reconsolidation_count == 0, (
        "Low-APE retrieval must NOT trigger reconsolidation"
    )


def test_high_ape_retrieval_opens_window_and_reconsolidates():
    """A retrieval with APE > threshold opens the window and updates core_affect."""
    engine = make_fidelity_engine(ape_threshold=0.3)
    engine.set_affect(CoreAffect(valence=0.0, arousal=0.3))
    engine.encode("Memory encoded under neutral affect.")
    original_valence = engine.list_all()[0].tag.core_affect.valence

    # Retrieve under very different affect → high APE
    engine.set_affect(CoreAffect(valence=0.9, arousal=0.9))
    results = engine.retrieve("neutral memory", top_k=1)

    assert results[0].tag.window_opened_at is not None, (
        "High-APE retrieval must open the reconsolidation window"
    )
    assert results[0].tag.reconsolidation_count >= 1, (
        "High-APE retrieval must trigger immediate reconsolidation"
    )
    assert results[0].tag.core_affect.valence > original_valence, (
        "core_affect.valence should shift positive after positive-context reconsolidation"
    )


def test_open_window_allows_reconsolidation_on_next_retrieval():
    """Within an open lability window, even a low-APE retrieval reconsolidates.

    Sequence: high-APE retrieval opens window → low-APE retrieval within
    window also reconsolidates.
    """
    engine = make_fidelity_engine(ape_threshold=0.3, mood_alpha=0.05)
    engine.set_affect(CoreAffect(valence=0.0, arousal=0.3))
    engine.encode("Memory to reconsolidate twice.")

    # Step 1: high-APE retrieval opens window
    engine.set_affect(CoreAffect(valence=0.9, arousal=0.9))
    r1 = engine.retrieve("neutral memory", top_k=1)
    assert r1[0].tag.window_opened_at is not None
    count_after_r1 = r1[0].tag.reconsolidation_count

    # Step 2: low-APE retrieval within window should also reconsolidate
    # Keep affect close to the now-updated core_affect but still different
    engine.set_affect(CoreAffect(valence=0.85, arousal=0.85))
    r2 = engine.retrieve("neutral memory", top_k=1)

    # reconsolidation_count should have incremented again
    assert r2[0].tag.reconsolidation_count > count_after_r1, (
        "Retrieval within open lability window should trigger additional reconsolidation"
    )
