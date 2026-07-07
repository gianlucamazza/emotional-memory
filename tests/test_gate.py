"""Tests for the Addendum Y query-affect gate helper (benchmarks/common/gate.py)."""

from __future__ import annotations

import math

import pytest

from benchmarks.common.gate import (
    gate_analysis,
    gated_scores,
    recovery_fraction,
    routing_rate,
)

# Aligned per-query fixtures: cosine, aft, valences (same order).
COSINE = [1.0, 0.0, 1.0, 0.0]
AFT = [0.0, 1.0, 0.0, 1.0]
VALENCES = [0.05, 0.5, -0.1, -0.9]  # |v|<0.2 for idx 0 and 2


def test_gated_selection_exact() -> None:
    # tau=0.2 → idx 0,2 neutral (take cosine), idx 1,3 affect-carrying (take aft).
    g = gated_scores(COSINE, AFT, VALENCES, tau=0.2)
    assert g == [1.0, 1.0, 1.0, 1.0]  # cosine[0], aft[1], cosine[2], aft[3]


def test_tau_zero_is_pure_aft() -> None:
    # abs(v) < 0 is never true → all queries take aft.
    assert gated_scores(COSINE, AFT, VALENCES, tau=0.0) == AFT


def test_tau_above_range_is_pure_cosine() -> None:
    # valence in [-1,1]; tau>1 → abs(v)<tau always true → all cosine.
    assert gated_scores(COSINE, AFT, VALENCES, tau=1.5) == COSINE


def test_boundary_is_strict_less_than() -> None:
    # abs(v) == tau is NOT neutral (strict <): takes aft.
    assert gated_scores([1.0], [0.0], [0.2], tau=0.2) == [0.0]
    assert gated_scores([1.0], [0.0], [0.19], tau=0.2) == [1.0]


def test_routing_rate() -> None:
    assert routing_rate(VALENCES, tau=0.2) == pytest.approx(0.5)  # idx 0,2
    assert routing_rate(VALENCES, tau=0.0) == 0.0
    assert routing_rate(VALENCES, tau=1.5) == 1.0
    assert routing_rate([], tau=0.2) == 0.0


def test_recovery_fraction() -> None:
    # cosine ahead of aft (penalty=0.5); gated halfway → 0.5 recovery.
    assert recovery_fraction(0.75, 0.5, 1.0) == pytest.approx(0.5)
    # gated == cosine → full recovery.
    assert recovery_fraction(1.0, 0.5, 1.0) == pytest.approx(1.0)
    # gated == aft → no recovery.
    assert recovery_fraction(0.5, 0.5, 1.0) == pytest.approx(0.0)
    # no penalty to recover (cosine == aft) → undefined.
    assert math.isnan(recovery_fraction(0.5, 0.5, 0.5))


def test_gate_analysis_summary() -> None:
    r = gate_analysis(COSINE, AFT, VALENCES, tau=0.2)
    assert r.tau == 0.2
    assert r.gated_scores == (1.0, 1.0, 1.0, 1.0)
    assert r.routing_rate == pytest.approx(0.5)
    assert r.gated_mean == pytest.approx(1.0)
    assert r.cosine_mean == pytest.approx(0.5)
    assert r.aft_mean == pytest.approx(0.5)
    # gated_mean(1.0) with cosine==aft==0.5 → no penalty → nan recovery.
    assert math.isnan(r.recovery_fraction)


def test_misaligned_lengths_raise() -> None:
    with pytest.raises(ValueError, match="misaligned"):
        gated_scores([1.0, 0.0], [0.0], [0.1, 0.2], tau=0.2)


def test_neutral_subset_equals_cosine_affect_subset_equals_aft() -> None:
    # The core construction invariant used by the closure sanity check.
    cos = [0.3, 0.7, 0.1, 0.9, 0.2]
    aft = [0.8, 0.2, 0.6, 0.4, 0.5]
    val = [0.0, 0.05, 0.5, -0.15, -0.8]  # neutral: 0,1,3 ; affect: 2,4
    g = gated_scores(cos, aft, val, tau=0.2)
    for i, v in enumerate(val):
        expected = cos[i] if abs(v) < 0.2 else aft[i]
        assert g[i] == expected
