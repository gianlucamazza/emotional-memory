"""Unit tests for the Addendum Z learning-to-rank core (benchmarks/common/ltr.py)."""

import numpy as np
import pytest

from benchmarks.common.ltr import (
    AFT_FIXED_WEIGHTS,
    COSINE_WEIGHTS,
    N_SIGNALS,
    QueryFeatures,
    cross_fit,
    fit_pairwise,
    score_fixed,
)


def _top1_metric(gold: np.ndarray):
    def metric(ranking):
        return 1.0 if gold[ranking[0]] > 0.5 else 0.0

    return metric


def _make_query(features: np.ndarray, gold: np.ndarray) -> QueryFeatures:
    return QueryFeatures(features=features, gold=gold, metric=_top1_metric(gold))


def _query_where_signal_separates(sig: int, n: int, rng, *, sign: float = 1.0) -> QueryFeatures:
    """One query where relevance is driven by feature ``sig`` (sign controls the
    direction: +1 = higher is relevant, -1 = lower is relevant)."""
    feats = rng.normal(size=(n, N_SIGNALS))
    gold = np.zeros(n)
    driver = sign * feats[:, sig]
    gold[np.argmax(driver)] = 1.0  # the candidate maximising sign*feature is relevant
    return _make_query(feats, gold)


class TestFitPairwise:
    def test_recovers_dominant_positive_signal(self):
        rng = np.random.default_rng(0)
        qs = [_query_where_signal_separates(0, 8, rng) for _ in range(60)]
        w = fit_pairwise([q.features for q in qs], [q.gold for q in qs])
        assert w.shape == (N_SIGNALS,)
        assert w[0] == max(w)  # signal 0 gets the largest weight
        assert w[0] > 0

    def test_recovers_counter_congruent_negative_s2(self):
        # Relevance is driven by LOW mood-congruence (s2, index 1): the
        # counter-congruent / support-mode regime named in Addendum X. The learned
        # weight on s2 must be negative — the interpretability readout Hz declares.
        rng = np.random.default_rng(1)
        qs = [_query_where_signal_separates(1, 8, rng, sign=-1.0) for _ in range(80)]
        w = fit_pairwise([q.features for q in qs], [q.gold for q in qs])
        assert w[1] < 0
        assert w[1] == min(w)  # s2 is the most negative component

    def test_no_contrast_returns_zeros(self):
        # All-relevant (or all-irrelevant) queries produce no pairs.
        feats = np.random.default_rng(2).normal(size=(5, N_SIGNALS))
        q = _make_query(feats, np.ones(5))
        w = fit_pairwise([q.features], [q.gold])
        assert np.allclose(w, 0.0)

    def test_deterministic(self):
        rng = np.random.default_rng(3)
        qs = [_query_where_signal_separates(0, 6, rng) for _ in range(20)]
        args = ([q.features for q in qs], [q.gold for q in qs])
        assert np.array_equal(fit_pairwise(*args), fit_pairwise(*args))


class TestScoreFixed:
    def test_cosine_ranks_by_first_signal(self):
        # Candidate 2 has the highest s1; gold is candidate 2 → top1 hit.
        feats = np.array(
            [[0.1, 9, 9, 9, 9, 9], [0.2, 9, 9, 9, 9, 9], [0.9, 0, 0, 0, 0, 0]],
            dtype=float,
        )
        gold = np.array([0.0, 0.0, 1.0])
        scores = score_fixed([_make_query(feats, gold)], COSINE_WEIGHTS)
        assert scores == [1.0]

    def test_fixed_weights_length(self):
        assert len(AFT_FIXED_WEIGHTS) == N_SIGNALS
        assert len(COSINE_WEIGHTS) == N_SIGNALS


class TestCrossFit:
    def test_heldout_alignment_and_weight_shape(self):
        rng = np.random.default_rng(4)
        qs = [_query_where_signal_separates(0, 8, rng) for _ in range(50)]
        heldout, w_mean, in_sample = cross_fit(qs, k=5)
        assert len(heldout) == len(qs)  # one held-out score per query
        assert w_mean.shape == (N_SIGNALS,)
        assert all(s in (0.0, 1.0) for s in heldout)  # top1 metric
        assert 0.0 <= in_sample <= 1.0

    def test_heldout_beats_chance_on_learnable_signal(self):
        # Signal 0 perfectly determines relevance → held-out top1 well above the
        # 1/8 random baseline.
        rng = np.random.default_rng(5)
        qs = [_query_where_signal_separates(0, 8, rng) for _ in range(120)]
        heldout, _, _ = cross_fit(qs, k=5)
        assert np.mean(heldout) > 0.5

    def test_k_less_than_two_raises(self):
        rng = np.random.default_rng(6)
        qs = [_query_where_signal_separates(0, 6, rng) for _ in range(10)]
        with pytest.raises(ValueError):
            cross_fit(qs, k=1)

    def test_empty_returns_empty(self):
        heldout, w_mean, in_sample = cross_fit([], k=5)
        assert heldout == []
        assert w_mean.shape == (N_SIGNALS,)
        assert np.isnan(in_sample)
