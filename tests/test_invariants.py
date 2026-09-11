"""Algebraic invariants over generated inputs (Hypothesis).

These tests certify properties the production docstrings already claim, for
arbitrary valid inputs rather than a hand-picked grid. They are not
psychological fidelity tests — see ``benchmarks/fidelity/``.
"""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from strategies import (
    NOW,
    core_affects,
    decay_configs,
    elapsed_seconds,
    emotional_tags,
    finite_components,
    mood_decay_configs,
    mood_fields,
    vectors,
    weight_bases,
)

from emotional_memory._math import cosine_similarity
from emotional_memory.affect import MAX_PAD_DISTANCE
from emotional_memory.decay import compute_effective_strength, compute_effective_strength_batch
from emotional_memory.retrieval import adaptive_weights


def _in_pad(valence: float, arousal: float, dominance: float) -> bool:
    return (
        -1.0 - 1e-9 <= valence <= 1.0 + 1e-9
        and -1e-9 <= arousal <= 1.0 + 1e-9
        and -1e-9 <= dominance <= 1.0 + 1e-9
    )


class TestDecayInvariants:
    @given(emotional_tags(), decay_configs(), elapsed_seconds)
    def test_strength_bounded_by_initial(self, tag, config, elapsed):
        now = NOW + timedelta(seconds=elapsed)
        strength = compute_effective_strength(tag, now, config)
        assert 0.0 <= strength <= tag.consolidation_strength + 1e-9

    @given(emotional_tags(), decay_configs(), elapsed_seconds, elapsed_seconds)
    def test_strength_monotone_in_time(self, tag, config, t_a, t_b):
        t1, t2 = (t_a, t_b) if t_a <= t_b else (t_b, t_a)
        s1 = compute_effective_strength(tag, NOW + timedelta(seconds=t1), config)
        s2 = compute_effective_strength(tag, NOW + timedelta(seconds=t2), config)
        assert s2 <= s1 + 1e-9

    @given(
        st.lists(emotional_tags(), min_size=1, max_size=16),
        decay_configs(),
        elapsed_seconds,
    )
    def test_batch_matches_scalar(self, tags, config, elapsed):
        now = NOW + timedelta(seconds=elapsed)
        scalar = [compute_effective_strength(tag, now, config) for tag in tags]
        batch = compute_effective_strength_batch(tags, now, config)
        np.testing.assert_allclose(batch, scalar, atol=1e-9)


class TestAdaptiveWeightInvariants:
    @given(mood_fields(), weight_bases)
    def test_weights_are_a_simplex(self, mood, base):
        weights = adaptive_weights(mood, base)
        assert weights.shape == (6,)
        assert np.all(weights >= -1e-15)
        assert math.isclose(float(weights.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-9)


class TestCoreAffectInvariants:
    @given(core_affects())
    def test_distance_to_self_is_zero(self, affect):
        assert affect.distance(affect) == 0.0

    @given(core_affects(), core_affects())
    def test_distance_is_symmetric_and_bounded(self, left, right):
        d_lr = left.distance(right)
        d_rl = right.distance(left)
        assert math.isclose(d_lr, d_rl, rel_tol=1e-12, abs_tol=1e-12)
        assert 0.0 <= d_lr <= MAX_PAD_DISTANCE + 1e-12

    @given(
        core_affects(),
        core_affects(),
        st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
    )
    def test_lerp_stays_in_pad_box(self, start, end, alpha):
        mixed = start.lerp(end, alpha)
        assert _in_pad(mixed.valence, mixed.arousal, mixed.dominance)


class TestMoodFieldInvariants:
    @given(
        mood_fields(),
        core_affects(),
        st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        elapsed_seconds,
        mood_decay_configs(),
    )
    def test_update_stays_in_pad_box(self, mood, affect, alpha, elapsed, decay):
        updated = mood.update(
            affect,
            alpha=alpha,
            now=NOW + timedelta(seconds=elapsed),
            decay_config=decay,
        )
        assert _in_pad(updated.valence, updated.arousal, updated.dominance)
        assert 0.0 - 1e-9 <= updated.inertia <= 1.0 + 1e-9

    @given(mood_fields(), elapsed_seconds, mood_decay_configs())
    def test_regress_stays_in_pad_box(self, mood, elapsed, decay):
        regressed = mood.regress(NOW + timedelta(seconds=elapsed), decay)
        assert _in_pad(regressed.valence, regressed.arousal, regressed.dominance)
        assert math.isclose(regressed.inertia, mood.inertia, abs_tol=1e-12)


class TestCosineInvariants:
    @given(st.integers(1, 16), st.data())
    def test_finite_result_is_bounded(self, dim, data):
        same_len = st.lists(finite_components, min_size=dim, max_size=dim)
        left = data.draw(same_len)
        right = data.draw(same_len)
        score = cosine_similarity(left, right)
        assert -1.0 - 1e-9 <= score <= 1.0 + 1e-9

    @given(vectors)
    def test_equal_nonzero_vectors_are_unit(self, vec):
        assume(any(abs(x) >= 1e-6 for x in vec))
        score = cosine_similarity(vec, vec)
        assert math.isclose(score, 1.0, rel_tol=1e-9, abs_tol=1e-9)

    @given(vectors)
    def test_zero_vector_returns_zero(self, vec):
        zeros = [0.0] * len(vec)
        assert cosine_similarity(zeros, vec) == 0.0
        assert cosine_similarity(vec, zeros) == 0.0

    @given(vectors)
    def test_nan_returns_zero(self, vec):
        poisoned = list(vec)
        poisoned[0] = float("nan")
        assert cosine_similarity(poisoned, vec) == 0.0
        assert cosine_similarity(vec, poisoned) == 0.0
