"""Tests for _math.cosine_similarity edge cases."""

import math

import pytest

from emotional_memory._math import cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors_returns_one(self):
        assert math.isclose(cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)

    def test_orthogonal_returns_zero(self):
        assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)

    def test_negative_correlation(self):
        assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0

    def test_both_zero_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_mismatched_dims_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_single_element(self):
        assert math.isclose(cosine_similarity([3.0], [5.0]), 1.0)

    def test_result_bounded(self):
        a = [0.1, 0.9, 0.5]
        b = [0.8, 0.2, 0.7]
        s = cosine_similarity(a, b)
        assert -1.0 <= s <= 1.0
