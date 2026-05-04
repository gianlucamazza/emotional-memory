from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.common.statistics import holm_bonferroni
from benchmarks.realistic.analyze_challenge_subsets import (
    SBERT_JSON,
    _challenge_types_present,
    compute_pairwise_by_challenge,
)

_REALISTIC_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "realistic"


def test_compute_pairwise_by_challenge_shape() -> None:
    result = compute_pairwise_by_challenge(SBERT_JSON)

    assert result["system"] == "aft"
    assert result["baseline"] == "naive_cosine"
    challenge_data = result["challenge_types"]

    expected_types = _challenge_types_present(SBERT_JSON)
    assert set(challenge_data.keys()) == set(expected_types)

    import json

    raw = json.loads(SBERT_JSON.read_text(encoding="utf-8"))
    expected_counts = raw["difficulty_profile"]["challenge_type_counts"]

    for ct in expected_types:
        for metric in ("top1", "hit_at_k"):
            m = challenge_data[ct][metric]
            assert "n_queries" in m
            assert m["n_queries"] == expected_counts[ct]
            if m["diff"] is not None:
                assert -1.0 <= m["diff"] <= 1.0
            if m["p_bootstrap"] is not None:
                assert 0.0 <= m["p_bootstrap"] <= 1.0
            if m["p_mcnemar"] is not None:
                assert 0.0 <= m["p_mcnemar"] <= 1.0


def test_compute_pairwise_by_challenge_is_deterministic() -> None:
    r1 = compute_pairwise_by_challenge(SBERT_JSON)
    r2 = compute_pairwise_by_challenge(SBERT_JSON)

    challenge_types = _challenge_types_present(SBERT_JSON)
    for ct in challenge_types:
        for metric in ("top1", "hit_at_k"):
            assert r1["challenge_types"][ct][metric] == r2["challenge_types"][ct][metric]


def test_holm_correction_family() -> None:
    # Holm correction is applied per-metric family (challenge types present)
    # not globally across both metrics at once.
    result = compute_pairwise_by_challenge(SBERT_JSON)
    challenge_types = _challenge_types_present(SBERT_JSON)

    # Collect raw p_bootstrap values per metric family
    for metric in ("top1", "hit_at_k"):
        p_vals = []
        adj_vals = []
        for ct in challenge_types:
            m = result["challenge_types"][ct][metric]
            if m["p_bootstrap"] is not None and m["p_bootstrap_adj_holm"] is not None:
                p_vals.append(m["p_bootstrap"])
                adj_vals.append(m["p_bootstrap_adj_holm"])

        if len(p_vals) < 2:
            continue

        # Holm-adjusted values must be >= their raw counterparts
        for raw_p, adj_p in zip(p_vals, adj_vals, strict=True):
            assert adj_p >= raw_p - 1e-9

        # The minimum adjusted p equals the maximum over Holm-adjusted sequence
        # — check that adjusted p-values are valid (within [0, 1])
        expected_adj = holm_bonferroni(p_vals)
        # Re-derived expected vs actual must match (same order, same inputs)
        # We can't compare directly because the order from the dict may differ,
        # but the set of adjusted values should match.
        assert sorted(adj_vals) == pytest.approx(sorted(expected_adj), abs=1e-4)
