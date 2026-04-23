"""Tests for benchmark/common/statistics.py statistical utilities."""

from __future__ import annotations

import itertools
import math

import pytest

from benchmarks.common.statistics import cohens_d_paired, holm_bonferroni


class TestHolmBonferroni:
    def test_single_p_value_unchanged(self) -> None:
        assert holm_bonferroni([0.03]) == pytest.approx([0.03])

    def test_empty_returns_empty(self) -> None:
        assert holm_bonferroni([]) == []

    def test_all_significant_first_pass(self) -> None:
        # m=3: rank 0 → p*3, rank 1 → p*2, rank 2 → p*1
        result = holm_bonferroni([0.01, 0.02, 0.04])
        assert result[0] == pytest.approx(0.03)  # 0.01 * 3
        assert result[1] == pytest.approx(0.04)  # 0.02 * 2
        assert result[2] == pytest.approx(0.04)  # 0.04 * 1

    def test_preserves_order_not_rank(self) -> None:
        # Input order is [0.04, 0.01, 0.02] — output must be in same positional order
        result = holm_bonferroni([0.04, 0.01, 0.02])
        assert result[1] == pytest.approx(0.03)  # 0.01 * 3 (smallest)
        assert result[2] == pytest.approx(0.04)  # 0.02 * 2
        assert result[0] == pytest.approx(0.04)  # 0.04 * 1

    def test_adjusted_never_exceeds_one(self) -> None:
        result = holm_bonferroni([0.5, 0.6, 0.7])
        assert all(p <= 1.0 for p in result)

    def test_monotonic_non_decreasing(self) -> None:
        # Sorted by rank, adjusted p must be non-decreasing
        p_values = [0.001, 0.05, 0.2, 0.4]
        result = holm_bonferroni(p_values)
        # Sort by original p to check ranks
        pairs = sorted(zip(p_values, result, strict=True))
        adjusted_in_rank_order = [adj for _, adj in pairs]
        for a, b in itertools.pairwise(adjusted_in_rank_order):
            assert a <= b + 1e-9

    def test_all_p_one(self) -> None:
        result = holm_bonferroni([1.0, 1.0, 1.0])
        assert all(p == pytest.approx(1.0) for p in result)


class TestCohensDPaired:
    def test_positive_effect(self) -> None:
        # Differences have variance → valid d > 0
        a = [1.0, 0.8, 1.2, 0.9, 1.1]
        b = [0.0, 0.2, 0.0, 0.3, 0.1]
        d = cohens_d_paired(a, b)
        assert d > 0

    def test_negative_effect(self) -> None:
        a = [0.0, 0.2, 0.0, 0.3, 0.1]
        b = [1.0, 0.8, 1.2, 0.9, 1.1]
        d = cohens_d_paired(a, b)
        assert d < 0

    def test_zero_effect(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        d = cohens_d_paired(a, b)
        assert math.isnan(d)  # std of differences is 0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            cohens_d_paired([1.0, 2.0], [1.0])

    def test_insufficient_data_returns_nan(self) -> None:
        assert math.isnan(cohens_d_paired([1.0], [0.0]))
        assert math.isnan(cohens_d_paired([], []))

    def test_hedges_correction_reduces_magnitude(self) -> None:
        a = [1.0, 0.0, 1.0, 0.0]
        b = [0.0, 1.0, 0.0, 1.0]
        d_raw = cohens_d_paired(a, b, hedges_correction=False)
        d_corrected = cohens_d_paired(a, b, hedges_correction=True)
        assert abs(d_corrected) <= abs(d_raw) + 1e-9

    def test_known_value(self) -> None:
        # diffs = [1, 1, 1, 1], mean=1, std=0 → nan (all equal diffs)
        # diffs = [1, -1, 1, -1], mean=0, std≈1.15 → d=0/1.15=0
        a = [1.0, 0.0, 1.0, 0.0]
        b = [0.0, 1.0, 0.0, 1.0]
        d = cohens_d_paired(a, b, hedges_correction=False)
        assert d == pytest.approx(0.0, abs=1e-9)
