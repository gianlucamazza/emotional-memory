"""Fidelity benchmark: discrete emotion categorization via Plutchik's wheel.

Validates that the angular mapping from continuous (valence, arousal) coordinates
to Plutchik's 8 primary emotions matches Russell's (1980) empirical circumplex
placements for canonical emotion terms.

Reference:
  Plutchik, R. (1980). Emotion: A psychoevolutionary synthesis. Harper & Row.
  Russell, J. A. (1980). A circumplex model of affect.
    Journal of Personality and Social Psychology, 39(6), 1161-1178.
"""

import pytest

from emotional_memory.affect import CoreAffect
from emotional_memory.categorize import categorize_affect

pytestmark = pytest.mark.fidelity


# Canonical emotion → (valence, arousal) placements derived from Russell (1980)
# and validated against Posner et al. (2005).
RUSSELL_PLACEMENTS = [
    # (valence, arousal, expected_primary)
    (0.80, 0.65, "joy"),  # happy, pleased
    (0.50, 0.80, "anticipation"),  # excited, eager
    (0.00, 0.90, "surprise"),  # astonished, aroused
    (-0.55, 0.85, "fear"),  # afraid, tense  (low dominance assumed)
    (-0.80, 0.50, "disgust"),  # disgusted, frustrated
    (-0.70, 0.20, "sadness"),  # sad, depressed
    (0.70, 0.20, "trust"),  # content, serene
]


@pytest.mark.parametrize("valence,arousal,expected", RUSSELL_PLACEMENTS)
def test_russell_placements_match_expected_primary(valence, arousal, expected):
    """Canonical Russell placements should map to the correct Plutchik primary."""
    label = categorize_affect(CoreAffect(valence=valence, arousal=arousal))
    assert label.primary == expected, (
        f"({valence}, {arousal}) → expected {expected!r}, got {label.primary!r}"
    )


def test_dominance_disambiguates_fear_vs_anger():
    """High arousal + negative valence maps to fear or anger based on dominance."""
    ca = CoreAffect(valence=-0.55, arousal=0.85)
    assert categorize_affect(ca, dominance=0.1).primary == "fear"
    assert categorize_affect(ca, dominance=0.9).primary == "anger"


def test_intensity_increases_with_radial_distance():
    """Emotions further from the origin should have higher intensity."""
    calm = categorize_affect(CoreAffect(valence=0.15, arousal=0.55))  # near centre
    joyful = categorize_affect(CoreAffect(valence=0.60, arousal=0.65))  # mid range
    ecstatic = categorize_affect(CoreAffect(valence=1.00, arousal=1.00))  # extreme

    intensity_order = {"low": 0, "moderate": 1, "high": 2}
    assert intensity_order[calm.intensity] <= intensity_order[joyful.intensity]
    assert intensity_order[joyful.intensity] <= intensity_order[ecstatic.intensity]


def test_all_8_primaries_are_reachable():
    """Every Plutchik primary can be returned by categorize_affect."""
    points = [
        (CoreAffect(valence=0.8, arousal=0.6), None),  # joy
        (CoreAffect(valence=0.5, arousal=0.9), None),  # anticipation
        (CoreAffect(valence=0.0, arousal=1.0), None),  # surprise
        (CoreAffect(valence=-0.5, arousal=0.85), 0.2),  # fear
        (CoreAffect(valence=-0.5, arousal=0.85), 0.8),  # anger
        (CoreAffect(valence=-0.9, arousal=0.5), None),  # disgust
        (CoreAffect(valence=-0.7, arousal=0.2), None),  # sadness
        (CoreAffect(valence=0.7, arousal=0.20), None),  # trust
    ]
    primaries = {categorize_affect(ca, dominance=dom).primary for ca, dom in points}
    expected = {"joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"}
    assert primaries == expected
