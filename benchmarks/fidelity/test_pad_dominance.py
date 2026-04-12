"""Fidelity benchmark: PAD dominance dimension (Mehrabian & Russell 1974).

Hypothesis
----------
The Pleasure-Arousal-Dominance model predicts that dominance (perceived
control) is modulated by the combination of valence and arousal:

- Positive valence + high arousal  → high dominance (excited control)
- Negative valence + high arousal  → low dominance  (threatened, loss of control)
- Low arousal (any valence)        → moderate dominance (calm neutrality)

The MoodField dominance update uses the heuristic:
  dominance_signal = 0.5 + 0.25 * valence * arousal

This benchmark verifies the directional predictions of the PAD model.

Theory
------
Mehrabian, A., & Russell, J. A. (1974). An approach to environmental
  psychology. MIT Press.
  Three independent dimensions: Pleasure (valence), Arousal, Dominance.
  High arousal + positive affect → approach + control.
  High arousal + negative affect → avoidance + loss of control.
"""

from __future__ import annotations

import pytest

from emotional_memory import CoreAffect
from emotional_memory.mood import MoodField

pytestmark = pytest.mark.fidelity


def _dominance_after_update(valence: float, arousal: float, steps: int = 10) -> float:
    """Drive a neutral MoodField with repeated updates and return steady-state dominance."""
    mood = MoodField.neutral()
    affect = CoreAffect(valence=valence, arousal=arousal)
    for _ in range(steps):
        mood = mood.update(affect, alpha=0.3)
    return mood.dominance


class TestPADDominanceFidelity:
    def test_positive_high_arousal_raises_dominance(self):
        """Positive valence + high arousal should push dominance above neutral (0.5)."""
        dominance = _dominance_after_update(valence=0.9, arousal=0.9)
        assert dominance > 0.5, (
            f"Positive/high-arousal affect must raise dominance above 0.5, got {dominance:.4f}"
        )

    def test_negative_high_arousal_lowers_dominance(self):
        """Negative valence + high arousal should push dominance below neutral (0.5)."""
        dominance = _dominance_after_update(valence=-0.9, arousal=0.9)
        assert dominance < 0.5, (
            f"Negative/high-arousal affect must lower dominance below 0.5, got {dominance:.4f}"
        )

    def test_positive_high_arousal_more_dominant_than_negative(self):
        """Positive/high-arousal dominance must exceed negative/high-arousal dominance."""
        dom_positive = _dominance_after_update(valence=0.8, arousal=0.8)
        dom_negative = _dominance_after_update(valence=-0.8, arousal=0.8)
        assert dom_positive > dom_negative, (
            f"PAD: positive affect must yield higher dominance than negative: "
            f"positive={dom_positive:.4f}, negative={dom_negative:.4f}"
        )

    def test_low_arousal_dominance_near_neutral(self):
        """Low arousal (any valence) keeps dominance close to the neutral baseline."""
        dom_pos = _dominance_after_update(valence=0.9, arousal=0.05)
        dom_neg = _dominance_after_update(valence=-0.9, arousal=0.05)

        # Both should be within 0.15 of neutral (0.5)
        assert abs(dom_pos - 0.5) < 0.15, (
            f"Low-arousal positive dominance too far from neutral: {dom_pos:.4f}"
        )
        assert abs(dom_neg - 0.5) < 0.15, (
            f"Low-arousal negative dominance too far from neutral: {dom_neg:.4f}"
        )

    def test_dominance_clamped_to_unit_interval(self):
        """Dominance must always remain in [0, 1] regardless of extreme affect."""
        extremes = [
            CoreAffect(valence=1.0, arousal=1.0),
            CoreAffect(valence=-1.0, arousal=1.0),
            CoreAffect(valence=1.0, arousal=0.0),
            CoreAffect(valence=-1.0, arousal=0.0),
        ]
        for affect in extremes:
            mood = MoodField.neutral()
            for _ in range(20):
                mood = mood.update(affect, alpha=0.5)
            assert 0.0 <= mood.dominance <= 1.0, (
                f"Dominance out of [0,1] for {affect}: {mood.dominance}"
            )

    @pytest.mark.parametrize(
        "valence,arousal",
        [
            (0.5, 0.8),
            (0.8, 0.5),
            (0.3, 0.9),
        ],
    )
    def test_dominance_signal_formula(self, valence: float, arousal: float):
        """The dominance signal in a single update step matches the documented formula."""
        mood = MoodField.neutral()
        alpha = 0.3
        updated = mood.update(CoreAffect(valence=valence, arousal=arousal), alpha=alpha)

        expected_signal = 0.5 + 0.25 * valence * arousal
        expected_dominance = (1.0 - alpha * (1.0 - mood.inertia)) * mood.dominance + alpha * (
            1.0 - mood.inertia
        ) * expected_signal
        # Clamp to [0, 1]
        expected_dominance = max(0.0, min(1.0, expected_dominance))

        assert updated.dominance == pytest.approx(expected_dominance, abs=1e-6), (
            f"Dominance formula mismatch for v={valence}, a={arousal}: "
            f"expected={expected_dominance:.6f}, got={updated.dominance:.6f}"
        )
