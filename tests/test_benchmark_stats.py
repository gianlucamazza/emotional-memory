"""Unit tests for benchmarks/common/statistics.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.common.statistics import (
    bootstrap_ci,
    ci_payload,
    format_point_ci,
    mcnemar_exact,
    paired_bootstrap_diff,
)


class TestBootstrapCI:
    def test_reproducible(self) -> None:
        values = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        r1 = bootstrap_ci(values, n_bootstrap=500, seed=42)
        r2 = bootstrap_ci(values, n_bootstrap=500, seed=42)
        assert r1 == r2

    def test_different_seeds_differ(self) -> None:
        values = [float(i % 2) for i in range(20)]
        _, lo1, hi1 = bootstrap_ci(values, n_bootstrap=200, seed=1)
        _, lo2, hi2 = bootstrap_ci(values, n_bootstrap=200, seed=2)
        assert (lo1, hi1) != (lo2, hi2)

    def test_point_exact(self) -> None:
        values = [0.0, 0.0, 1.0, 1.0, 1.0]
        point, _, _ = bootstrap_ci(values, n_bootstrap=500, seed=0)
        assert point == pytest.approx(0.6)

    def test_degenerate_empty(self) -> None:
        point, lo, hi = bootstrap_ci([], n_bootstrap=100, seed=0)
        assert math.isnan(point)
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_degenerate_singleton(self) -> None:
        point, lo, hi = bootstrap_ci([0.5], n_bootstrap=100, seed=0)
        assert point == pytest.approx(0.5)
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(0.5)

    def test_ci_bounds_ordered(self) -> None:
        values = [float(i % 2) for i in range(50)]
        point, lo, hi = bootstrap_ci(values, n_bootstrap=500, seed=0)
        assert lo <= point <= hi

    def test_coverage_bernoulli(self) -> None:
        """95% CI should cover true p=0.6 in at least 88% of 500 samples."""
        rng = np.random.default_rng(0)
        covered = 0
        n_trials = 500
        for i in range(n_trials):
            sample = rng.binomial(1, 0.6, size=100).tolist()
            _, lo, hi = bootstrap_ci([float(x) for x in sample], n_bootstrap=500, seed=i)
            if lo <= 0.6 <= hi:
                covered += 1
        coverage = covered / n_trials
        assert coverage >= 0.88, f"Coverage too low: {coverage:.3f}"


class TestPairedBootstrapDiff:
    def test_sign_and_significance(self) -> None:
        a = [1.0] * 80 + [0.0] * 20
        b = [0.0] * 80 + [1.0] * 20
        diff, lo, _hi, p = paired_bootstrap_diff(a, b, n_bootstrap=2000, seed=0)
        assert diff == pytest.approx(0.6, abs=1e-6)
        assert p < 0.01
        assert lo > 0.0  # CI excludes 0

    def test_null_hypothesis(self) -> None:
        values = [float(i % 2) for i in range(40)]
        diff, lo, hi, p = paired_bootstrap_diff(values, values, n_bootstrap=1000, seed=0)
        assert diff == pytest.approx(0.0, abs=1e-10)
        assert p == pytest.approx(1.0)
        assert lo <= 0.0 <= hi

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            paired_bootstrap_diff([1.0, 2.0], [1.0], n_bootstrap=10, seed=0)

    def test_p_in_range(self) -> None:
        a = [1.0, 0.0, 1.0, 0.0, 1.0]
        b = [0.0, 1.0, 0.0, 1.0, 0.0]
        _, _, _, p = paired_bootstrap_diff(a, b, n_bootstrap=500, seed=0)
        assert 0.0 <= p <= 1.0


class TestMcNemarExact:
    def test_agresti_example(self) -> None:
        # Agresti 2002: only_a=9, only_b=21 → p ≈ 0.0433 two-sided
        p = mcnemar_exact(9, 21)
        assert p == pytest.approx(0.0433, abs=0.005)

    def test_no_discordant(self) -> None:
        assert mcnemar_exact(0, 0) == pytest.approx(1.0)

    def test_all_a(self) -> None:
        # only_a=10, only_b=0 → k=0, p = 2*(C(10,0)*0.5^10) = 2/1024
        p = mcnemar_exact(10, 0)
        assert p == pytest.approx(2 * 0.5**10, rel=1e-6)

    def test_symmetric(self) -> None:
        assert mcnemar_exact(9, 21) == pytest.approx(mcnemar_exact(21, 9))

    def test_balanced(self) -> None:
        # equal discordants → p should be 1.0
        assert mcnemar_exact(10, 10) == pytest.approx(1.0)


class TestFormatPointCI:
    def test_standard(self) -> None:
        assert format_point_ci(0.6, 0.45, 0.75) == "0.60 [0.45, 0.75]"

    def test_dp_1(self) -> None:
        assert format_point_ci(0.6, 0.4, 0.8, dp=1) == "0.6 [0.4, 0.8]"

    def test_dp_3(self) -> None:
        result = format_point_ci(0.123, 0.050, 0.200, dp=3)
        assert result == "0.123 [0.050, 0.200]"


class TestCIPayload:
    def test_keys(self) -> None:
        payload = ci_payload(0.6, 0.45, 0.75, n_bootstrap=2000)
        assert set(payload.keys()) == {"point", "ci_lower", "ci_upper", "ci_method", "n_bootstrap"}

    def test_values(self) -> None:
        payload = ci_payload(0.6, 0.45, 0.75, n_bootstrap=500, method="test")
        assert payload["point"] == 0.6
        assert payload["ci_lower"] == 0.45
        assert payload["ci_upper"] == 0.75
        assert payload["ci_method"] == "test"
        assert payload["n_bootstrap"] == 500
