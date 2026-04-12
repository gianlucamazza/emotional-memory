"""Fidelity benchmark: Yerkes-Dodson inverted-U consolidation curve.

Hypothesis: memory consolidation strength follows an inverted-U function
of arousal, peaking at moderate-to-high arousal (effective ≈ 0.7) and
declining at both extremes.

Reference: Yerkes, R.M. & Dodson, J.D. (1908). The relation of strength of
stimulus to rapidity of habit-formation. Journal of Comparative Neurology.
"""

import pytest

from emotional_memory.appraisal import consolidation_strength

pytestmark = pytest.mark.fidelity

AROUSAL_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_consolidation_peak_is_not_at_extremes():
    """The peak consolidation is not at arousal=0 or arousal=1.

    With mood_arousal=1.0, effective = 0.7*enc + 0.3, so the peak
    (effective=0.7) is reached at enc_arousal ≈ 0.571 — safely in the interior.
    """
    values = [consolidation_strength(a, 1.0) for a in AROUSAL_LEVELS]
    peak_idx = values.index(max(values))
    assert peak_idx not in (0, len(AROUSAL_LEVELS) - 1), (
        f"Peak at extreme index {peak_idx} — expected mid-range. "
        f"Values: {[f'{v:.3f}' for v in values]}"
    )


def test_consolidation_inverted_u_shape():
    """Consolidation rises then falls — classic inverted-U."""
    # Use mood_arousal=0.7 so effective_arousal = 0.7*enc + 0.21
    # Peak at effective=0.7 → enc_arousal ≈ 0.7 (effective = 0.7*0.7+0.21=0.7)
    mood_a = 0.7
    values = [consolidation_strength(a, mood_a) for a in AROUSAL_LEVELS]
    peak_idx = values.index(max(values))

    # Monotone increase up to peak, monotone decrease after
    for i in range(1, peak_idx):
        assert values[i] >= values[i - 1] - 1e-9, (
            f"Non-monotone increase before peak at idx {i}: {values[i - 1]:.3f} → {values[i]:.3f}"
        )
    for i in range(peak_idx + 1, len(values)):
        assert values[i] <= values[i - 1] + 1e-9, (
            f"Non-monotone decrease after peak at idx {i}: {values[i - 1]:.3f} → {values[i]:.3f}"
        )


def test_consolidation_peak_near_effective_07():
    """Peak consolidation_strength == 1.0 when effective_arousal == 0.7."""
    # effective = 0.7*enc + 0.3*stim = 0.7 when enc=stim=0.7
    peak = consolidation_strength(0.7, 0.7)
    assert abs(peak - 1.0) < 1e-9, f"Peak should be 1.0, got {peak}"


@pytest.mark.parametrize(
    "arousal,mood_arousal",
    [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (0.7, 0.7),
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, -1.0),  # clamped inputs
        (2.0, 2.0),  # clamped inputs
    ],
)
def test_consolidation_always_in_unit_range(arousal, mood_arousal):
    """consolidation_strength is always in [0.0, 1.0] regardless of inputs."""
    v = consolidation_strength(arousal, mood_arousal)
    assert 0.0 <= v <= 1.0, (
        f"Out of range: consolidation_strength({arousal}, {mood_arousal}) = {v}"
    )


def test_low_arousal_consolidation_below_peak():
    """Very low arousal yields consolidation well below the peak of 1.0."""
    v = consolidation_strength(0.0, 0.0)
    assert v < 0.8, f"Low-arousal consolidation ({v:.3f}) should be well below 1.0"
