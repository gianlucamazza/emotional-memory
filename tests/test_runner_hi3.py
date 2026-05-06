"""Tests for the Hi3 confirmatory analysis script (benchmarks/ablation/runner_hi3.py).

Uses synthetic per_query_records fixtures to exercise PASS/FAIL/NULL paths
and statistical invariants without requiring real benchmark runs.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from benchmarks.ablation.runner_hi3 import (
    _HI3_ALPHA,
    _HI3_EFFECT_THRESHOLD,
    _PREREG_SEED,
    CHALLENGE_FAMILY,
    HYPOTHESIS_NAMES,
    _align_amp_vectors,
    _build_amp_vector,
    _load_per_query_records,
    run_hi3,
)

# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _make_records(
    *,
    n: int,
    challenge_type: str,
    full_hits: list[bool],
    no_res_hits: list[bool],
    scenario_prefix: str = "sc",
) -> dict[str, list[dict[str, Any]]]:
    """Build a minimal per_query_records dict with two variants (full, no_resonance)."""
    assert len(full_hits) == n
    assert len(no_res_hits) == n

    full_recs: list[dict[str, Any]] = []
    no_res_recs: list[dict[str, Any]] = []
    for i in range(n):
        qid = f"{scenario_prefix}_q{i:04d}_{challenge_type}"
        full_recs.append(
            {
                "query_id": qid,
                "scenario_id": f"{scenario_prefix}_{i}",
                "challenge_type": challenge_type,
                "top1_hit": full_hits[i],
                "hit": full_hits[i],
            }
        )
        no_res_recs.append(
            {
                "query_id": qid,
                "scenario_id": f"{scenario_prefix}_{i}",
                "challenge_type": challenge_type,
                "top1_hit": no_res_hits[i],
                "hit": no_res_hits[i],
            }
        )
    # Include other variants with empty records so the runner doesn't crash
    return {
        "full": full_recs,
        "no_resonance": no_res_recs,
        "no_appraisal": [],
        "no_mood": [],
        "no_momentum": [],
        "no_reconsolidation": [],
        "dual_path": [],
        "aft_keyword_synchronous": [],
    }


def _write_results_json(
    tmp_path: Path,
    records: dict[str, list[dict[str, Any]]],
    filename: str = "results.json",
    link_set_stats: dict[str, Any] | None = None,
) -> Path:
    data: dict[str, Any] = {"per_query_records": records}
    if link_set_stats is not None:
        data["link_set_stats"] = link_set_stats
    p = tmp_path / filename
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _build_amp_vector
# ---------------------------------------------------------------------------


def test_build_amp_vector_basic() -> None:
    records = _make_records(
        n=4,
        challenge_type="semantic_confound",
        full_hits=[True, True, False, False],
        no_res_hits=[False, True, False, True],
    )
    qids, amp = _build_amp_vector(records, challenge_type="semantic_confound")
    # amp[i] = full - no_res: [1-0, 1-1, 0-0, 0-1] = [1, 0, 0, -1]
    assert len(qids) == 4
    assert amp == [1.0, 0.0, 0.0, -1.0]


def test_build_amp_vector_filters_challenge_type() -> None:
    records = _make_records(
        n=3,
        challenge_type="semantic_confound",
        full_hits=[True, True, False],
        no_res_hits=[False, True, True],
    )
    # Ask for a different challenge type — should return empty
    qids, amp = _build_amp_vector(records, challenge_type="affective_arc")
    assert qids == []
    assert amp == []


def test_build_amp_vector_sorted() -> None:
    records = _make_records(
        n=5,
        challenge_type="recency_confound",
        full_hits=[True] * 5,
        no_res_hits=[False] * 5,
        scenario_prefix="zz",
    )
    qids, _ = _build_amp_vector(records, challenge_type="recency_confound")
    assert qids == sorted(qids)


# ---------------------------------------------------------------------------
# _align_amp_vectors
# ---------------------------------------------------------------------------


def test_align_amp_vectors_perfect_overlap() -> None:
    e5 = (["q0", "q1", "q2"], [1.0, 0.0, -1.0])
    sb = (["q0", "q1", "q2"], [0.5, -0.5, 0.0])
    aligned_e5, aligned_sbert = _align_amp_vectors(e5[0], e5[1], sb[0], sb[1])
    assert len(aligned_e5) == 3
    assert len(aligned_sbert) == 3
    # Sorted by query_id: q0, q1, q2
    assert aligned_e5 == [1.0, 0.0, -1.0]
    assert aligned_sbert == [0.5, -0.5, 0.0]


def test_align_amp_vectors_raises_on_large_mismatch() -> None:
    e5_ids = [f"q{i:03d}" for i in range(10)]
    sbert_ids = [f"q{i:03d}" for i in range(2)]  # only 2/10 overlap
    e5_amp = [0.0] * 10
    sbert_amp = [0.0] * 2
    with pytest.raises(RuntimeError, match="mismatch"):
        _align_amp_vectors(e5_ids, e5_amp, sbert_ids, sbert_amp)


def test_align_amp_vectors_small_mismatch_ok() -> None:
    # 9/10 overlap (90%) should NOT raise
    e5_ids = [f"q{i:03d}" for i in range(10)]
    sbert_ids = [f"q{i:03d}" for i in range(9)]
    e5_amp = [1.0] * 10
    sbert_amp = [0.5] * 9
    aligned_e5, aligned_sbert = _align_amp_vectors(e5_ids, e5_amp, sbert_ids, sbert_amp)
    assert len(aligned_e5) == 9
    assert len(aligned_sbert) == 9


# ---------------------------------------------------------------------------
# _load_per_query_records
# ---------------------------------------------------------------------------


def test_load_per_query_records_raises_missing_field(tmp_path: Path) -> None:
    p = tmp_path / "no_records.json"
    p.write_text(json.dumps({"benchmark": "test"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="per_query_records"):
        _load_per_query_records(p)


# ---------------------------------------------------------------------------
# run_hi3 — PASS path
# ---------------------------------------------------------------------------


def test_runner_hi3_pass_path(tmp_path: Path) -> None:
    """Synthetic data where e5 has much stronger resonance amplification on semantic_confound."""
    n = 60

    # e5: resonance always helps on semantic_confound (amp_e5 ~ +1)
    e5_full = [True] * n
    e5_no_res = [False] * n
    # sbert: resonance has no effect (amp_sbert ~ 0)
    sb_full = [True] * (n // 2) + [False] * (n // 2)
    sb_no_res = [True] * (n // 2) + [False] * (n // 2)

    e5_recs = _make_records(
        n=n, challenge_type="semantic_confound", full_hits=e5_full, no_res_hits=e5_no_res
    )
    sb_recs = _make_records(
        n=n, challenge_type="semantic_confound", full_hits=sb_full, no_res_hits=sb_no_res
    )
    # Add other challenge types (empty — Holm still runs with 0-length pairs; that's ok for
    # the other two hypotheses — they'll have NaN delta but verdict FAIL which is valid)
    for ct in ("recency_confound", "affective_arc"):
        e5_recs["full"] += []  # already present via other challenge type
        # add records for the other challenges so family is non-empty
        extra_e5 = _make_records(
            n=10, challenge_type=ct, full_hits=[True] * 10, no_res_hits=[False] * 10
        )
        extra_sb = _make_records(
            n=10, challenge_type=ct, full_hits=[True] * 10, no_res_hits=[True] * 10
        )
        e5_recs["full"] += extra_e5["full"]
        e5_recs["no_resonance"] += extra_e5["no_resonance"]
        sb_recs["full"] += extra_sb["full"]
        sb_recs["no_resonance"] += extra_sb["no_resonance"]

    sbert_path = _write_results_json(tmp_path, sb_recs, "sbert.json")
    e5_path = _write_results_json(tmp_path, e5_recs, "e5.json")

    results = run_hi3(sbert_path, e5_path, n_bootstrap=500, seed=1)

    hi3 = results["hypotheses"]["Hi3"]
    assert hi3["delta"] > _HI3_EFFECT_THRESHOLD, f"delta={hi3['delta']}"
    assert hi3["p_adj_holm"] < _HI3_ALPHA, f"p_adj={hi3['p_adj_holm']}"
    assert hi3["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# run_hi3 — FAIL path (threshold not met)
# ---------------------------------------------------------------------------


def test_runner_hi3_fail_path_threshold(tmp_path: Path) -> None:
    """e5 advantage is tiny (< 0.05) — FAIL on effect threshold."""
    n = 60
    # e5: resonance helps on 3/60 queries (amp ~ 0.05)
    e5_full = [True] * 3 + [False] * 57
    e5_no_res = [False] * 3 + [False] * 57
    # sbert: resonance has no effect
    sb_full = [False] * n
    sb_no_res = [False] * n

    e5_recs = _make_records(
        n=n, challenge_type="semantic_confound", full_hits=e5_full, no_res_hits=e5_no_res
    )
    sb_recs = _make_records(
        n=n, challenge_type="semantic_confound", full_hits=sb_full, no_res_hits=sb_no_res
    )
    for ct in ("recency_confound", "affective_arc"):
        extra = _make_records(
            n=5, challenge_type=ct, full_hits=[False] * 5, no_res_hits=[False] * 5
        )
        for v in ("full", "no_resonance"):
            e5_recs[v] += extra[v]
            sb_recs[v] += extra[v]

    sbert_path = _write_results_json(tmp_path, sb_recs, "sbert.json")
    e5_path = _write_results_json(tmp_path, e5_recs, "e5.json")

    results = run_hi3(sbert_path, e5_path, n_bootstrap=200, seed=1)
    hi3 = results["hypotheses"]["Hi3"]
    # delta = 0.05 exactly — threshold is strict >, so FAIL
    assert hi3["verdict"] == "FAIL"


# ---------------------------------------------------------------------------
# run_hi3 — NULL path (matched vectors)
# ---------------------------------------------------------------------------


def test_runner_hi3_null_path(tmp_path: Path) -> None:
    """Identical e5 and sbert performance — all hypotheses FAIL."""
    n = 40
    full = [True] * (n // 2) + [False] * (n // 2)
    no_res = [False] * (n // 2) + [True] * (n // 2)

    recs = _make_records(
        n=n, challenge_type="semantic_confound", full_hits=full, no_res_hits=no_res
    )
    for ct in ("recency_confound", "affective_arc"):
        extra = _make_records(
            n=10, challenge_type=ct, full_hits=full[:10], no_res_hits=no_res[:10]
        )
        for v in ("full", "no_resonance"):
            recs[v] += extra[v]

    path_a = _write_results_json(tmp_path, recs, "a.json")
    path_b = _write_results_json(tmp_path, recs, "b.json")

    results = run_hi3(path_a, path_b, n_bootstrap=200, seed=1)
    for challenge in CHALLENGE_FAMILY:
        name = HYPOTHESIS_NAMES[challenge]
        h = results["hypotheses"][name]
        assert h["verdict"] == "FAIL", f"{name} should FAIL with identical data"
        # delta should be near zero
        assert abs(h["delta"]) < 0.01, f"{name} delta={h['delta']} should be ~0"


# ---------------------------------------------------------------------------
# Holm correction ordering
# ---------------------------------------------------------------------------


def test_runner_hi3_holm_correction(tmp_path: Path) -> None:
    """Holm p_adj for the smallest p must be <= p_adj for larger p values."""
    n = 80
    # Construct data where Hi3 (semantic_confound) has large effect,
    # Hi3_recency has small effect, Hi3_arc has no effect.
    e5_recs: dict[str, list[dict[str, Any]]] = {
        v: []
        for v in [
            "full",
            "no_resonance",
            "no_appraisal",
            "no_mood",
            "no_momentum",
            "no_reconsolidation",
            "dual_path",
            "aft_keyword_synchronous",
        ]
    }
    sb_recs: dict[str, list[dict[str, Any]]] = {v: [] for v in e5_recs}

    sc = _make_records(
        n=n,
        challenge_type="semantic_confound",
        full_hits=[True] * n,
        no_res_hits=[False] * n,
        scenario_prefix="sc",
    )
    rc = _make_records(
        n=n,
        challenge_type="recency_confound",
        full_hits=[True] * (n // 2) + [False] * (n // 2),
        no_res_hits=[False] * (n // 2) + [False] * (n // 2),
        scenario_prefix="rc",
    )
    ar = _make_records(
        n=n,
        challenge_type="affective_arc",
        full_hits=[False] * n,
        no_res_hits=[False] * n,
        scenario_prefix="ar",
    )
    sb_neutral = _make_records(
        n=n,
        challenge_type="semantic_confound",
        full_hits=[False] * n,
        no_res_hits=[False] * n,
        scenario_prefix="sc",
    )
    sb_rc = _make_records(
        n=n,
        challenge_type="recency_confound",
        full_hits=[False] * n,
        no_res_hits=[False] * n,
        scenario_prefix="rc",
    )
    sb_ar = _make_records(
        n=n,
        challenge_type="affective_arc",
        full_hits=[False] * n,
        no_res_hits=[False] * n,
        scenario_prefix="ar",
    )

    for v in ("full", "no_resonance"):
        e5_recs[v] = sc[v] + rc[v] + ar[v]
        sb_recs[v] = sb_neutral[v] + sb_rc[v] + sb_ar[v]

    sbert_path = _write_results_json(tmp_path, sb_recs, "sbert.json")
    e5_path = _write_results_json(tmp_path, e5_recs, "e5.json")
    results = run_hi3(sbert_path, e5_path, n_bootstrap=300, seed=1)

    hi3_adj = results["hypotheses"]["Hi3"]["p_adj_holm"]
    hi3r_adj = results["hypotheses"]["Hi3_recency"]["p_adj_holm"]
    hi3a_adj = results["hypotheses"]["Hi3_arc"]["p_adj_holm"]
    # Holm step-down: adjusted p-values must be non-decreasing from smallest raw p
    all_adjs = sorted([hi3_adj, hi3r_adj, hi3a_adj])
    assert all_adjs == sorted(all_adjs)


# ---------------------------------------------------------------------------
# Alignment raises on large mismatch
# ---------------------------------------------------------------------------


def test_runner_hi3_alignment_raises_on_mismatch(tmp_path: Path) -> None:
    """run_hi3 must raise if cross-embedder query_ids have <90% overlap."""
    e5_recs = _make_records(
        n=20,
        challenge_type="semantic_confound",
        full_hits=[True] * 20,
        no_res_hits=[False] * 20,
        scenario_prefix="e5x",
    )
    sb_recs = _make_records(
        n=20,
        challenge_type="semantic_confound",
        full_hits=[True] * 20,
        no_res_hits=[False] * 20,
        scenario_prefix="sbx",
    )
    # Disjoint scenario prefixes → disjoint query_ids → <90% overlap
    for ct in ("recency_confound", "affective_arc"):
        extra_e5 = _make_records(
            n=5,
            challenge_type=ct,
            full_hits=[True] * 5,
            no_res_hits=[False] * 5,
            scenario_prefix="e5x",
        )
        extra_sb = _make_records(
            n=5,
            challenge_type=ct,
            full_hits=[True] * 5,
            no_res_hits=[False] * 5,
            scenario_prefix="sbx",
        )
        for v in ("full", "no_resonance"):
            e5_recs[v] += extra_e5[v]
            sb_recs[v] += extra_sb[v]

    sbert_path = _write_results_json(tmp_path, sb_recs, "sbert.json")
    e5_path = _write_results_json(tmp_path, e5_recs, "e5.json")
    with pytest.raises(RuntimeError, match="mismatch"):
        run_hi3(sbert_path, e5_path, n_bootstrap=50, seed=1)


# ---------------------------------------------------------------------------
# seed propagation
# ---------------------------------------------------------------------------


def test_runner_hi3_uses_prescribed_seed(tmp_path: Path) -> None:
    """run_hi3 must propagate seed=1 to paired_bootstrap_diff."""
    n = 20
    e5_recs = _make_records(
        n=n, challenge_type="semantic_confound", full_hits=[True] * n, no_res_hits=[False] * n
    )
    sb_recs = _make_records(
        n=n, challenge_type="semantic_confound", full_hits=[False] * n, no_res_hits=[False] * n
    )
    for ct in ("recency_confound", "affective_arc"):
        extra = _make_records(
            n=5, challenge_type=ct, full_hits=[True] * 5, no_res_hits=[False] * 5
        )
        for v in ("full", "no_resonance"):
            e5_recs[v] += extra[v]
            sb_recs[v] += extra[v]

    sbert_path = _write_results_json(tmp_path, sb_recs, "sbert.json")
    e5_path = _write_results_json(tmp_path, e5_recs, "e5.json")

    seeds_used: list[int] = []
    original = __import__(
        "benchmarks.common.statistics", fromlist=["paired_bootstrap_diff"]
    ).paired_bootstrap_diff

    def _capture_seed(*args: Any, **kwargs: Any) -> Any:
        seeds_used.append(kwargs.get("seed", 0))
        return original(*args, **kwargs)

    with patch("benchmarks.ablation.runner_hi3.paired_bootstrap_diff", side_effect=_capture_seed):
        run_hi3(sbert_path, e5_path, n_bootstrap=50, seed=_PREREG_SEED)

    assert all(s == _PREREG_SEED for s in seeds_used), f"Non-prescribed seeds: {seeds_used}"


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_runner_hi3_output_structure(tmp_path: Path) -> None:
    n = 30
    recs = _make_records(
        n=n, challenge_type="semantic_confound", full_hits=[True] * n, no_res_hits=[False] * n
    )
    for ct in ("recency_confound", "affective_arc"):
        extra = _make_records(
            n=10, challenge_type=ct, full_hits=[True] * 10, no_res_hits=[False] * 10
        )
        for v in ("full", "no_resonance"):
            recs[v] += extra[v]

    path = _write_results_json(tmp_path, recs, "results.json")
    results = run_hi3(path, path, n_bootstrap=100, seed=1)

    assert "study" in results
    assert "hypotheses" in results
    assert "protocol" in results
    assert "mechanism_exploratory" in results
    assert "run_timestamp" in results

    hyps = results["hypotheses"]
    assert set(hyps.keys()) == {"Hi3", "Hi3_recency", "Hi3_arc"}
    for name, h in hyps.items():
        assert "delta" in h
        assert "p_adj_holm" in h
        assert "verdict" in h
        assert h["verdict"] in ("PASS", "FAIL")
        assert not math.isnan(h["delta"]), f"{name} delta is NaN"

    proto = results["protocol"]
    assert proto["n_bootstrap"] == 100
    assert proto["seed"] == 1
    assert proto["alpha"] == _HI3_ALPHA
    assert proto["effect_threshold"] == _HI3_EFFECT_THRESHOLD
    assert proto["family_m"] == 3
