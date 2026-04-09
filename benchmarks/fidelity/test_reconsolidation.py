"""Fidelity benchmark: Memory reconsolidation via Affective Prediction Error.

Hypothesis: when a memory is retrieved under an emotional context that
significantly differs from its encoding context (high APE), the memory's
affective tag is updated toward the current context (reconsolidation).
The update is proportional to the APE but capped at 50% per retrieval.

Reference: Nader, K. & Schiller, D. (2000). Memory reconsolidation.
"""

import math

import pytest

from benchmarks.conftest import make_fidelity_engine
from emotional_memory import CoreAffect
from emotional_memory.affect import AffectiveMomentum
from emotional_memory.models import make_emotional_tag
from emotional_memory.retrieval import reconsolidate
from emotional_memory.stimmung import StimmungField

pytestmark = pytest.mark.fidelity


def _neutral_tag():
    tag = make_emotional_tag(
        core_affect=CoreAffect(valence=0.0, arousal=0.0),
        momentum=AffectiveMomentum.zero(),
        stimmung=StimmungField.neutral(),
        consolidation_strength=0.5,
    )
    return tag


def test_reconsolidation_updates_core_affect():
    """High-APE retrieval within lability window shifts the tag's valence.

    Reconsolidation requires two retrievals: the first destabilises the memory
    (sets last_retrieved); the second, within reconsolidation_window_seconds,
    updates the tag if APE exceeds the threshold.
    """
    engine = make_fidelity_engine(ape_threshold=0.01, stimmung_alpha=0.3)

    engine.set_affect(CoreAffect(valence=0.0, arousal=0.3))
    engine.encode("A neutral workplace discussion that ended inconclusively.")
    original_valence = engine._store.list_all()[0].tag.core_affect.valence

    # First retrieval — destabilises the memory (sets last_retrieved)
    engine.retrieve("workplace discussion", top_k=1)

    # Second retrieval within window under strongly positive affect → reconsolidation
    for _ in range(5):
        engine.set_affect(CoreAffect(valence=0.9, arousal=0.8))
    results = engine.retrieve("workplace discussion", top_k=1)

    updated = results[0]
    assert updated.tag.reconsolidation_count >= 1, "Expected at least one reconsolidation"
    assert updated.tag.core_affect.valence > original_valence, (
        f"Valence should shift positive after positive-context retrieval: "
        f"original={original_valence:.3f}, updated={updated.tag.core_affect.valence:.3f}"
    )


def test_reconsolidation_proportional_to_ape():
    """Larger APE produces larger shift in core_affect."""
    tag = _neutral_tag()
    target = CoreAffect(valence=1.0, arousal=1.0)
    lr = 0.2

    updated_small = reconsolidate(tag, target, ape=0.2, learning_rate=lr)
    updated_large = reconsolidate(tag, target, ape=1.0, learning_rate=lr)

    assert updated_large.core_affect.valence > updated_small.core_affect.valence, (
        "Larger APE should produce larger valence shift"
    )


def test_reconsolidation_capped_at_50_percent():
    """Even extreme APE cannot shift valence by more than 50% per retrieval."""
    tag = _neutral_tag()
    target = CoreAffect(valence=1.0, arousal=1.0)

    updated = reconsolidate(tag, target, ape=1000.0, learning_rate=100.0)
    # alpha capped at 0.5 → valence = 0 + 0.5*(1-0) = 0.5
    assert math.isclose(updated.core_affect.valence, 0.5), (
        f"Valence shift capped at 50%: got {updated.core_affect.valence:.3f}"
    )


def test_reconsolidation_count_incremented():
    """Each reconsolidation call increments the reconsolidation_count."""
    tag = _neutral_tag()
    target = CoreAffect(valence=0.8, arousal=0.7)

    updated = reconsolidate(tag, target, ape=0.5, learning_rate=0.3)
    assert updated.reconsolidation_count == tag.reconsolidation_count + 1


def test_low_ape_does_not_trigger_reconsolidation():
    """Below-threshold APE leaves the tag unchanged (engine-level test)."""
    engine = make_fidelity_engine(ape_threshold=10.0)  # very high threshold

    engine.set_affect(CoreAffect(valence=0.5, arousal=0.5))
    engine.encode("Memory encoded at moderate positive affect.")

    # Retrieve under similar affect → APE << threshold → no reconsolidation
    engine.set_affect(CoreAffect(valence=0.6, arousal=0.5))
    results = engine.retrieve("moderate memory", top_k=1)

    assert results[0].tag.reconsolidation_count == 0, (
        "Low-APE retrieval should not trigger reconsolidation"
    )
