"""Fidelity benchmark: PAD dominance dimension (Mehrabian & Russell 1974).

Hypothesis
----------
The Pleasure-Arousal-Dominance model predicts that dominance (perceived
control) is now a first-class dimension of CoreAffect.  MoodField.update()
tracks core_affect.dominance directly via EMA.  The PAD predictions hold
through the appraisal layer: coping_potential maps linearly to dominance.

- High CoreAffect.dominance  → MoodField.dominance trends above neutral (0.5)
- Low CoreAffect.dominance   → MoodField.dominance trends below neutral (0.5)

Theory
------
Mehrabian, A., & Russell, J. A. (1974). An approach to environmental
  psychology. MIT Press.
  Three independent dimensions: Pleasure (valence), Arousal, Dominance.
"""

from __future__ import annotations

import pytest

from emotional_memory import CoreAffect
from emotional_memory.mood import MoodField

pytestmark = pytest.mark.fidelity


def _dominance_after_update(
    valence: float, arousal: float, dominance: float = 0.5, steps: int = 10
) -> float:
    """Drive a neutral MoodField with repeated updates and return steady-state dominance."""
    mood = MoodField.neutral()
    affect = CoreAffect(valence=valence, arousal=arousal, dominance=dominance)
    for _ in range(steps):
        mood = mood.update(affect, alpha=0.3)
    return mood.dominance


class TestPADDominanceFidelity:
    def test_high_dominance_affect_raises_mood_dominance(self):
        """High CoreAffect.dominance should push MoodField dominance above neutral (0.5)."""
        dominance = _dominance_after_update(valence=0.5, arousal=0.5, dominance=0.9)
        assert dominance > 0.5, (
            f"High-dominance affect must raise MoodField dominance above 0.5, got {dominance:.4f}"
        )

    def test_low_dominance_affect_lowers_mood_dominance(self):
        """Low CoreAffect.dominance should push MoodField dominance below neutral (0.5)."""
        dominance = _dominance_after_update(valence=0.5, arousal=0.5, dominance=0.1)
        assert dominance < 0.5, (
            f"Low-dominance affect must lower MoodField dominance below 0.5, got {dominance:.4f}"
        )

    def test_high_dominance_exceeds_low_dominance(self):
        """High-dominance CoreAffect must yield higher MoodField dominance than low-dominance."""
        dom_high = _dominance_after_update(valence=0.5, arousal=0.5, dominance=0.9)
        dom_low = _dominance_after_update(valence=0.5, arousal=0.5, dominance=0.1)
        assert dom_high > dom_low, (
            f"PAD: high-dominance affect must yield higher MoodField dominance than low: "
            f"high={dom_high:.4f}, low={dom_low:.4f}"
        )

    def test_neutral_dominance_stays_near_neutral(self):
        """Neutral CoreAffect.dominance (0.5) keeps MoodField dominance close to baseline."""
        dom = _dominance_after_update(valence=0.9, arousal=0.9, dominance=0.5)
        assert abs(dom - 0.5) < 0.05, (
            f"Neutral dominance affect must keep MoodField dominance near 0.5, got {dom:.4f}"
        )

    def test_dominance_clamped_to_unit_interval(self):
        """Dominance must always remain in [0, 1] regardless of extreme affect."""
        extremes = [
            CoreAffect(valence=1.0, arousal=1.0, dominance=1.0),
            CoreAffect(valence=-1.0, arousal=1.0, dominance=0.0),
            CoreAffect(valence=1.0, arousal=0.0, dominance=1.0),
            CoreAffect(valence=-1.0, arousal=0.0, dominance=0.0),
        ]
        for affect in extremes:
            mood = MoodField.neutral()
            for _ in range(20):
                mood = mood.update(affect, alpha=0.5)
            assert 0.0 <= mood.dominance <= 1.0, (
                f"Dominance out of [0,1] for {affect}: {mood.dominance}"
            )

    @pytest.mark.parametrize(
        "dominance,alpha",
        [
            (0.9, 0.3),
            (0.1, 0.3),
            (0.7, 0.5),
        ],
    )
    def test_dominance_ema_formula(self, dominance: float, alpha: float):
        """MoodField dominance update is a direct EMA of core_affect.dominance."""
        mood = MoodField.neutral()
        affect = CoreAffect(valence=0.0, arousal=0.0, dominance=dominance)
        updated = mood.update(affect, alpha=alpha)

        eff_alpha = alpha * (1.0 - mood.inertia)
        expected_dominance = (1.0 - eff_alpha) * mood.dominance + eff_alpha * dominance
        expected_dominance = max(0.0, min(1.0, expected_dominance))

        assert updated.dominance == pytest.approx(expected_dominance, abs=1e-6), (
            f"Dominance EMA mismatch for dominance={dominance}, alpha={alpha}: "
            f"expected={expected_dominance:.6f}, got={updated.dominance:.6f}"
        )
