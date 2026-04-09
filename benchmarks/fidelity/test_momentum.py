"""Fidelity benchmark: Affective momentum and inertia (Spinoza).

Hypothesis: the system's affective trajectory (velocity and acceleration)
captures the direction of change rather than just the current state.
After a sustained positive trajectory, a single negative event produces
negative velocity (d_valence < 0) but positive acceleration change signals
the deceleration.

Reference: Spinoza, B. (1677). Ethics III, Def. of Emotions.
"""

import pytest

from emotional_memory import CoreAffect
from emotional_memory.state import AffectiveState

pytestmark = pytest.mark.fidelity


def _build_state(*valence_sequence: float) -> AffectiveState:
    state = AffectiveState.initial()
    for v in valence_sequence:
        state = state.update(CoreAffect(valence=v, arousal=0.5))
    return state


def test_positive_trajectory_yields_positive_velocity():
    """Sustained positive affect → positive d_valence."""
    state = _build_state(0.1, 0.3, 0.5, 0.7)
    assert state.momentum.d_valence > 0, (
        f"Expected positive velocity after rising valence, got {state.momentum.d_valence:.3f}"
    )


def test_negative_trajectory_yields_negative_velocity():
    """Sustained negative affect → negative d_valence."""
    state = _build_state(0.7, 0.5, 0.3, 0.1)
    assert state.momentum.d_valence < 0, (
        f"Expected negative velocity after falling valence, got {state.momentum.d_valence:.3f}"
    )


def test_deceleration_after_peak():
    """After an acceleration phase, slowing down produces negative dd_valence."""
    # Accelerate: +0.2, +0.4 steps → positive dd_valence
    # Then decelerate: +0.1 step → dd_valence turns negative
    state = _build_state(0.0, 0.2, 0.6, 0.7)
    # v0→v1: +0.2, v1→v2: +0.4, v2→v3: +0.1
    # d_valence[-1] = 0.7 - 0.6 = 0.1
    # d_valence[-2] = 0.6 - 0.2 = 0.4
    # dd_valence = 0.1 - 0.4 = -0.3 (deceleration)
    assert state.momentum.dd_valence < 0, (
        f"Expected negative acceleration (deceleration), got {state.momentum.dd_valence:.3f}"
    )


def test_constant_velocity_zero_acceleration():
    """Constant-step increases produce near-zero acceleration."""
    state = _build_state(0.0, 0.2, 0.4, 0.6)
    # d_valence[-1] = 0.2, d_valence[-2] = 0.2 → dd = 0
    import math

    assert math.isclose(state.momentum.dd_valence, 0.0, abs_tol=1e-9), (
        f"Constant velocity should yield ~0 acceleration, got {state.momentum.dd_valence}"
    )


def test_zero_momentum_with_no_history():
    """Single-update state has no velocity (not enough history)."""
    state = AffectiveState.initial()
    state = state.update(CoreAffect(valence=0.9, arousal=0.8))
    assert state.momentum.d_valence == 0.0
    assert state.momentum.d_arousal == 0.0


def test_momentum_magnitude_increases_with_speed():
    """Larger valence steps produce higher momentum magnitude."""
    slow = _build_state(0.0, 0.1)
    fast = _build_state(0.0, 0.9)
    assert fast.momentum.magnitude() > slow.momentum.magnitude()


@pytest.mark.parametrize(
    "sequence",
    [
        (0.1, 0.3, 0.5),
        (-0.3, -0.1, 0.1),
        (0.8, 0.4, 0.0),
    ],
)
def test_momentum_velocity_sign_matches_last_step(sequence):
    """d_valence sign should match the sign of the last step's delta."""
    state = _build_state(*sequence)
    last_delta = sequence[-1] - sequence[-2]
    if last_delta > 0:
        assert state.momentum.d_valence > 0
    elif last_delta < 0:
        assert state.momentum.d_valence < 0
