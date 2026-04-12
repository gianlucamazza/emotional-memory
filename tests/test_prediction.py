"""Unit tests for compute_ape, update_prediction, and adaptive reconsolidate."""

import pytest

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.models import make_emotional_tag
from emotional_memory.mood import MoodField
from emotional_memory.retrieval import compute_ape, reconsolidate, update_prediction


def _tag(valence: float = 0.0, arousal: float = 0.5):
    return make_emotional_tag(
        core_affect=CoreAffect(valence=valence, arousal=arousal),
        momentum=AffectiveMomentum.zero(),
        mood=MoodField.neutral(),
        consolidation_strength=0.5,
    )


# ---------------------------------------------------------------------------
# compute_ape
# ---------------------------------------------------------------------------


class TestComputeApe:
    def test_falls_back_to_core_affect_when_no_expected(self):
        tag = _tag(valence=0.0, arousal=0.5)
        assert tag.expected_affect is None
        observed = CoreAffect(valence=1.0, arousal=1.0)
        ape = compute_ape(tag, observed)
        expected_dist = tag.core_affect.distance(observed)
        assert ape == pytest.approx(expected_dist)

    def test_uses_expected_affect_when_set(self):
        tag = _tag(valence=0.0, arousal=0.5)
        # Manually set expected_affect to a different point
        expected = CoreAffect(valence=0.5, arousal=0.7)
        tag_with_pred = tag.model_copy(update={"expected_affect": expected})
        observed = CoreAffect(valence=1.0, arousal=1.0)

        ape_with_pred = compute_ape(tag_with_pred, observed)
        ape_fallback = compute_ape(tag, observed)

        # expected_affect (0.5,0.7) is closer to observed (1.0,1.0) than core_affect (0.0,0.5)
        assert ape_with_pred < ape_fallback

    def test_zero_ape_when_observed_matches_core_affect(self):
        tag = _tag(valence=0.3, arousal=0.6)
        ape = compute_ape(tag, tag.core_affect)
        assert ape == pytest.approx(0.0, abs=1e-9)

    def test_zero_ape_when_observed_matches_expected(self):
        expected = CoreAffect(valence=0.5, arousal=0.7)
        tag = _tag().model_copy(update={"expected_affect": expected})
        ape = compute_ape(tag, expected)
        assert ape == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# update_prediction
# ---------------------------------------------------------------------------


class TestUpdatePrediction:
    def test_initialises_expected_affect(self):
        tag = _tag(valence=0.0, arousal=0.5)
        assert tag.expected_affect is None
        observed = CoreAffect(valence=0.8, arousal=0.8)
        updated = update_prediction(tag, observed, ape=0.5)
        assert updated.expected_affect is not None

    def test_expected_affect_moves_toward_observed(self):
        tag = _tag(valence=0.0, arousal=0.5)
        observed = CoreAffect(valence=1.0, arousal=1.0)
        updated = update_prediction(tag, observed, ape=0.5)
        assert updated.expected_affect is not None
        # expected_affect should be between core_affect and observed
        assert 0.0 < updated.expected_affect.valence < 1.0
        assert 0.5 < updated.expected_affect.arousal < 1.0

    def test_lr_increases_on_large_ape(self):
        """Large prediction error → learning rate increases (Pearce-Hall)."""
        tag = _tag()
        observed = CoreAffect(valence=1.0, arousal=1.0)
        original_lr = tag.prediction_learning_rate
        updated = update_prediction(tag, observed, ape=1.0)  # max ape
        assert updated.prediction_learning_rate > original_lr

    def test_lr_decreases_on_small_ape(self):
        """Small prediction error → learning rate decreases."""
        tag = _tag(valence=0.5, arousal=0.7)
        # Set lr above the APE so it should decrease
        tag = tag.model_copy(update={"prediction_learning_rate": 0.5})
        observed = CoreAffect(valence=0.5, arousal=0.7)  # matches → APE ≈ 0
        updated = update_prediction(tag, observed, ape=0.0)
        assert updated.prediction_learning_rate < 0.5

    def test_lr_stays_within_bounds(self):
        tag = _tag()
        # Extreme APE: lr should not exceed 0.80
        tag_high_lr = tag.model_copy(update={"prediction_learning_rate": 0.79})
        updated = update_prediction(tag_high_lr, CoreAffect(valence=1.0, arousal=1.0), ape=2.0)
        assert updated.prediction_learning_rate <= 0.80

        # Zero APE: lr should not fall below 0.05
        tag_low_lr = tag.model_copy(update={"prediction_learning_rate": 0.06})
        updated = update_prediction(tag_low_lr, tag.core_affect, ape=0.0)
        assert updated.prediction_learning_rate >= 0.05

    def test_all_other_fields_unchanged(self):
        tag = _tag(valence=0.3, arousal=0.6)
        observed = CoreAffect(valence=0.8, arousal=0.9)
        updated = update_prediction(tag, observed, ape=0.5)
        assert updated.core_affect == tag.core_affect
        assert updated.consolidation_strength == tag.consolidation_strength
        assert updated.reconsolidation_count == tag.reconsolidation_count
        assert updated.pending_appraisal == tag.pending_appraisal

    def test_convergence_over_many_updates(self):
        """expected_affect converges toward the repeated observed value."""
        tag = _tag(valence=0.0, arousal=0.5)
        target = CoreAffect(valence=0.8, arousal=0.8)
        for _ in range(50):
            ape = compute_ape(tag, target)
            tag = update_prediction(tag, target, ape=ape)
        assert tag.expected_affect is not None
        assert tag.expected_affect.valence == pytest.approx(0.8, abs=0.1)
        assert tag.expected_affect.arousal == pytest.approx(0.8, abs=0.1)


# ---------------------------------------------------------------------------
# reconsolidate with adaptive rate
# ---------------------------------------------------------------------------


class TestReconsolidateAdaptive:
    def test_adapt_rate_true_larger_ape_gives_larger_shift(self):
        tag = _tag(valence=0.0, arousal=0.5)
        target = CoreAffect(valence=1.0, arousal=1.0)
        small = reconsolidate(tag, target, ape=0.1, learning_rate=0.3, adapt_rate=True)
        large = reconsolidate(tag, target, ape=1.0, learning_rate=0.3, adapt_rate=True)
        assert large.core_affect.valence > small.core_affect.valence

    def test_adapt_rate_false_matches_original_formula(self):
        """adapt_rate=False should behave like the original formula."""
        tag = _tag(valence=0.0, arousal=0.5)
        target = CoreAffect(valence=1.0, arousal=1.0)
        ape, lr = 0.5, 0.2
        updated = reconsolidate(tag, target, ape=ape, learning_rate=lr, adapt_rate=False)
        alpha = min(ape * lr, 0.5)  # original formula
        expected_valence = tag.core_affect.lerp(target, alpha).valence
        assert updated.core_affect.valence == pytest.approx(expected_valence)

    def test_cap_still_holds_with_adapt_rate(self):
        tag = _tag(valence=0.0, arousal=0.5)
        target = CoreAffect(valence=1.0, arousal=1.0)
        updated = reconsolidate(tag, target, ape=100.0, learning_rate=100.0, adapt_rate=True)
        assert updated.core_affect.valence <= 0.5 + 1e-9
