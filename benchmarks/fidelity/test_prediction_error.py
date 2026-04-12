"""Fidelity benchmark: affective prediction error and adaptive learning.

Validates three phenomena:
1. expected_affect converges toward the repeatedly observed affect (EMA learning)
2. prediction_learning_rate adapts based on error magnitude (Pearce-Hall, 1980)
3. Reconsolidation shift is proportional to APE magnitude (Schultz, 1997)

References:
  Schultz, W. (1997). A neural substrate of prediction and reward.
    Science, 275(5306), 1593-1599.
  Pearce, J. M., & Hall, G. (1980). A model for Pavlovian conditioning.
    Psychological Review, 87(6), 532-552.
"""

import pytest

from benchmarks.conftest import make_fidelity_engine
from emotional_memory import CoreAffect
from emotional_memory.affect import AffectiveMomentum
from emotional_memory.models import make_emotional_tag
from emotional_memory.mood import MoodField
from emotional_memory.retrieval import compute_ape, reconsolidate, update_prediction

pytestmark = pytest.mark.fidelity


def _tag(valence: float = 0.0, arousal: float = 0.5):
    return make_emotional_tag(
        core_affect=CoreAffect(valence=valence, arousal=arousal),
        momentum=AffectiveMomentum.zero(),
        mood=MoodField.neutral(),
        consolidation_strength=0.5,
    )


def test_expected_affect_converges_toward_observed():
    """After repeated retrievals, expected_affect approaches the observed mean.

    Simulates a memory retrieved 30 times under consistent affect (0.7, 0.8).
    The prediction model (EMA) should converge to that value.
    """
    tag = _tag(valence=0.0, arousal=0.5)
    target = CoreAffect(valence=0.7, arousal=0.8)

    for _ in range(30):
        ape = compute_ape(tag, target)
        tag = update_prediction(tag, target, ape=ape)

    assert tag.expected_affect is not None
    assert tag.expected_affect.valence == pytest.approx(0.7, abs=0.15), (
        f"expected_affect.valence should approach 0.7, got {tag.expected_affect.valence:.3f}"
    )
    assert tag.expected_affect.arousal == pytest.approx(0.8, abs=0.15), (
        f"expected_affect.arousal should approach 0.8, got {tag.expected_affect.arousal:.3f}"
    )


def test_learning_rate_adapts_to_prediction_error():
    """Large prediction error increases lr; small error decreases it (Pearce-Hall).

    Two parallel runs start from the same tag:
    - Run A: high APE (surprising) → lr should increase
    - Run B: zero APE (predictable) → lr should decrease
    """
    base_lr = 0.35
    tag = _tag().model_copy(update={"prediction_learning_rate": base_lr})
    observed_close = tag.core_affect  # APE ≈ 0
    observed_far = CoreAffect(valence=1.0, arousal=1.0)  # large APE

    tag_a = update_prediction(tag, observed_far, ape=compute_ape(tag, observed_far))
    tag_b = update_prediction(tag, observed_close, ape=compute_ape(tag, observed_close))

    assert tag_a.prediction_learning_rate > base_lr, (
        "Surprising observation should increase learning rate"
    )
    assert tag_b.prediction_learning_rate < base_lr, (
        "Predictable observation should decrease learning rate"
    )


def test_reconsolidation_shift_proportional_to_ape():
    """Larger APE produces a larger shift in core_affect (Schultz RPE analogue).

    Three APE values (small/medium/large) should produce monotonically
    increasing shifts in core_affect valence.
    """
    tag = _tag(valence=0.0, arousal=0.5)
    target = CoreAffect(valence=1.0, arousal=1.0)
    lr = 0.3

    shifts = []
    for ape in (0.1, 0.5, 1.0):
        updated = reconsolidate(tag, target, ape=ape, learning_rate=lr, adapt_rate=True)
        shifts.append(updated.core_affect.valence)

    assert shifts[0] < shifts[1] < shifts[2], (
        f"Shifts should be monotonically increasing: {shifts}"
    )


def test_prediction_model_reduces_ape_over_time():
    """As expected_affect converges, APE for a consistent observed affect decreases.

    Simulates 20 retrievals under the same affect; APE in round 20 should be
    lower than APE in round 1 because the prediction model has adapted.
    """
    tag = _tag(valence=0.0, arousal=0.5)
    target = CoreAffect(valence=0.7, arousal=0.8)

    ape_first = compute_ape(tag, target)
    for _ in range(20):
        ape = compute_ape(tag, target)
        tag = update_prediction(tag, target, ape=ape)
    ape_last = compute_ape(tag, target)

    assert ape_last < ape_first, (
        f"APE should decrease after adaptation: first={ape_first:.3f} last={ape_last:.3f}"
    )


def test_engine_updates_prediction_on_every_retrieval():
    """The engine calls update_prediction() on each retrieved memory.

    After two retrievals under consistent affect, expected_affect should be
    initialised (not None) on the stored memory.
    """
    engine = make_fidelity_engine(ape_threshold=10.0)  # disable reconsolidation
    engine.set_affect(CoreAffect(valence=0.5, arousal=0.6))
    engine.encode("A memory to track prediction updates.")

    stored_before = engine.list_all()[0]
    assert stored_before.tag.expected_affect is None

    engine.retrieve("prediction update memory", top_k=1)
    stored_after = engine.list_all()[0]
    assert stored_after.tag.expected_affect is not None, (
        "expected_affect should be initialised after first retrieval"
    )
