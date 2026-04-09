"""Fidelity benchmark: Stimmung-adaptive retrieval weights (Heidegger).

Hypothesis: the current Stimmung modulates which retrieval signal dominates.
  - Negative Stimmung → emotional signals weighted more heavily
  - High arousal → momentum signal weighted more heavily
  - Neutral calm → semantic signal weighted more heavily

Reference: Heidegger, M. (1927). Being and Time §29 — Stimmung as the
ontological ground of disclosure.
"""

import math
from datetime import UTC, datetime

import pytest

from emotional_memory.retrieval import adaptive_weights
from emotional_memory.stimmung import StimmungField

pytestmark = pytest.mark.fidelity


def _stimmung(valence: float, arousal: float) -> StimmungField:
    return StimmungField(
        valence=valence,
        arousal=arousal,
        dominance=0.5,
        inertia=0.5,
        timestamp=datetime.now(tz=UTC),
    )


BASE = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
# Indices: 0=semantic, 1=stimmung_congruence, 2=affect_proximity,
#           3=momentum, 4=recency, 5=resonance


def test_negative_stimmung_raises_emotional_weight():
    """w[1] (stimmung_congruence) increases under negative Stimmung."""
    w_neg = adaptive_weights(_stimmung(-0.8, 0.3), BASE)
    w_neu = adaptive_weights(_stimmung(0.0, 0.0), BASE)
    assert w_neg[1] > w_neu[1], (
        f"Negative Stimmung should raise stimmung_congruence weight: "
        f"{w_neg[1]:.3f} vs {w_neu[1]:.3f}"
    )


def test_negative_stimmung_lowers_semantic_weight():
    """w[0] (semantic) decreases under negative Stimmung."""
    w_neg = adaptive_weights(_stimmung(-0.8, 0.3), BASE)
    w_neu = adaptive_weights(_stimmung(0.0, 0.0), BASE)
    assert w_neg[0] < w_neu[0], (
        f"Negative Stimmung should lower semantic weight: {w_neg[0]:.3f} vs {w_neu[0]:.3f}"
    )


def test_high_arousal_raises_momentum_weight():
    """w[3] (momentum_alignment) increases under high-arousal Stimmung."""
    w_high = adaptive_weights(_stimmung(0.0, 0.9), BASE)
    w_neu = adaptive_weights(_stimmung(0.0, 0.0), BASE)
    assert w_high[3] > w_neu[3], (
        f"High arousal should raise momentum weight: {w_high[3]:.3f} vs {w_neu[3]:.3f}"
    )


def test_calm_stimmung_raises_semantic_weight():
    """w[0] (semantic) increases under calm neutral Stimmung."""
    w_calm = adaptive_weights(_stimmung(0.1, 0.1), BASE)
    w_active = adaptive_weights(_stimmung(0.5, 0.6), BASE)
    assert w_calm[0] > w_active[0], (
        f"Calm Stimmung should raise semantic weight: {w_calm[0]:.3f} vs {w_active[0]:.3f}"
    )


@pytest.mark.parametrize(
    "valence,arousal",
    [(-1.0, 0.0), (-0.8, 0.3), (0.0, 0.9), (0.1, 0.1), (0.5, 0.5)],
)
def test_weights_always_sum_to_one(valence, arousal):
    """Adaptive weights always sum to 1.0 regardless of Stimmung."""
    w = adaptive_weights(_stimmung(valence, arousal), BASE)
    assert math.isclose(w.sum(), 1.0, rel_tol=1e-9), (
        f"Weights sum to {w.sum():.6f} for valence={valence}, arousal={arousal}"
    )


@pytest.mark.parametrize(
    "valence,arousal",
    [(-1.0, 0.0), (-0.8, 0.3), (0.0, 0.9), (0.1, 0.1), (0.5, 0.5)],
)
def test_weights_non_negative(valence, arousal):
    """No weight is ever negative."""
    w = adaptive_weights(_stimmung(valence, arousal), BASE)
    assert all(wi >= 0.0 for wi in w), (
        f"Negative weight found for valence={valence}, arousal={arousal}: {w}"
    )
