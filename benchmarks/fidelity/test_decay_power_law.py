"""Fidelity benchmark: ACT-R power-law memory decay (Anderson 1983).

Hypothesis
----------
Memory strength decays as a power function of elapsed time:
  strength(t) = initial * t^(-d)
where d is the effective decay exponent (modulated by arousal and retrieval
count).  This means:

1. Strength is a strictly decreasing function of elapsed time.
2. Decay is faster early on and slows as time passes (concave on a
   log-log plot — the hallmark of a power law vs. exponential decay).
3. High-arousal memories decay more slowly than low-arousal ones.
4. The log-log slope equals -effective_decay (linearity on log-log scale).

Theory
------
Anderson, J. R. (1983). The Architecture of Cognition. Harvard University Press.
  ACT-R memory activation: B_i = ln(Σ t_j^(-d)) ≈ ln(n) - d * ln(T)

McGaugh, J. L. (2004). The amygdala modulates the consolidation of memories
  of emotionally arousing experiences. Annual Review of Neuroscience.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from emotional_memory import CoreAffect
from emotional_memory.affect import AffectiveMomentum
from emotional_memory.decay import DecayConfig, compute_effective_strength
from emotional_memory.models import make_emotional_tag
from emotional_memory.mood import MoodField

pytestmark = pytest.mark.fidelity


def _tag(arousal: float, retrieval_count: int = 0, now: datetime | None = None) -> object:
    if now is None:
        now = datetime.now(tz=UTC)
    tag = make_emotional_tag(
        core_affect=CoreAffect(valence=0.0, arousal=arousal),
        momentum=AffectiveMomentum.zero(),
        mood=MoodField.neutral(),
        consolidation_strength=1.0,
    )
    return tag.model_copy(update={"timestamp": now, "retrieval_count": retrieval_count})


class TestACTRPowerLawFidelity:
    def test_strength_decreases_over_time(self):
        """Strength must be a strictly decreasing function of elapsed time."""
        config = DecayConfig(base_decay=0.5)
        now = datetime.now(tz=UTC)
        tag = _tag(arousal=0.5, now=now)

        checkpoints = [1, 10, 100, 1000, 10_000]  # seconds
        strengths = [
            compute_effective_strength(tag, now + timedelta(seconds=t), config)
            for t in checkpoints
        ]

        for i in range(1, len(strengths)):
            assert strengths[i] < strengths[i - 1], (
                f"Strength must decrease: t={checkpoints[i - 1]}s ({strengths[i - 1]:.4f}) "
                f"→ t={checkpoints[i]}s ({strengths[i]:.4f})"
            )

    def test_log_log_linearity(self):
        """On a log-log scale, strength vs. time must be approximately linear.

        For a pure power law s(t) = C * t^(-d):
          log(s) = log(C) - d * log(t)
        The R² of the linear fit on log-log data should be > 0.99.
        """
        config = DecayConfig(base_decay=0.5, power=1.0)
        now = datetime.now(tz=UTC)
        tag = _tag(arousal=0.5, retrieval_count=0, now=now)

        # Sample 10 log-spaced time points from 1s to 1e6s
        ts = [10 ** (i * 0.6) for i in range(10)]
        log_t = [math.log(t) for t in ts]
        log_s = [
            math.log(
                max(
                    compute_effective_strength(tag, now + timedelta(seconds=t), config),
                    1e-12,  # floor to avoid log(0)
                )
            )
            for t in ts
        ]

        # Simple linear regression on (log_t, log_s)
        n = len(ts)
        mean_x = sum(log_t) / n
        mean_y = sum(log_s) / n
        ss_xx = sum((x - mean_x) ** 2 for x in log_t)
        ss_yy = sum((y - mean_y) ** 2 for y in log_s)
        ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_t, log_s, strict=True))
        r_squared = (ss_xy**2) / (ss_xx * ss_yy) if ss_xx * ss_yy > 0 else 0.0

        assert r_squared > 0.99, (
            f"Log-log linearity R²={r_squared:.4f} below 0.99; decay curve is not a power law"
        )

    def test_high_arousal_decays_slower(self):
        """High-arousal memories must retain more strength than low-arousal ones."""
        config = DecayConfig(base_decay=0.5, arousal_modulation=0.5)
        now = datetime.now(tz=UTC)
        later = now + timedelta(days=30)

        tag_low = _tag(arousal=0.1, now=now)
        tag_high = _tag(arousal=0.9, now=now)

        s_low = compute_effective_strength(tag_low, later, config)
        s_high = compute_effective_strength(tag_high, later, config)

        assert s_high > s_low, (
            f"High-arousal memory should decay slower: high={s_high:.4f}, low={s_low:.4f}"
        )

    def test_high_arousal_floor_is_honoured(self):
        """Memories with arousal above floor_arousal_threshold must not decay below floor_value."""
        config = DecayConfig(
            base_decay=0.9,
            floor_arousal_threshold=0.7,
            floor_value=0.05,
        )
        now = datetime.now(tz=UTC)
        far_future = now + timedelta(days=3650)  # 10 years
        tag = _tag(arousal=0.9, now=now)

        s = compute_effective_strength(tag, far_future, config)
        assert s >= config.floor_value - 1e-9, (
            f"High-arousal memory strength {s:.6f} dropped below floor {config.floor_value}"
        )

    def test_low_arousal_can_fall_below_floor(self):
        """Memories with arousal below the floor threshold may decay arbitrarily low."""
        config = DecayConfig(
            base_decay=0.9,
            floor_arousal_threshold=0.7,
            floor_value=0.05,
        )
        now = datetime.now(tz=UTC)
        far_future = now + timedelta(days=3650)
        tag = _tag(arousal=0.1, now=now)

        s = compute_effective_strength(tag, far_future, config)
        # Low-arousal memory CAN fall below floor_value — that's expected
        # We just verify it's non-negative
        assert s >= 0.0, f"Strength must be non-negative, got {s}"
