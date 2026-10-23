"""Check scientific reporting with small, hand-calculated examples."""

import json
import math
from copy import deepcopy

import pytest

from benchmarks.realistic.evaluation import ranking_metrics, state_divergence
from benchmarks.realistic.runner import (
    _build_pairwise_comparisons,
    load_dataset,
    validate_dataset_difficulty,
    write_results,
)
from benchmarks.realistic.shared_history import run_shared_history


def test_binary_metrics_multiple_relevant_and_duplicate_results():
    scores = ranking_metrics(["x", "a", "a", "b"], {"a", "b", "c"}, 3)
    assert scores["recall_at_k"] == pytest.approx(1 / 3)
    assert scores["reciprocal_rank_at_k"] == 0.5
    assert scores["ndcg_at_k"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3) + 0.5))
    assert ranking_metrics([], {"a"}, 3) == {
        "recall_at_k": 0,
        "ndcg_at_k": 0,
        "reciprocal_rank_at_k": 0,
    }
    assert ranking_metrics(["a"], {"a"}, 3)["recall_at_k"] == 1
    assert ranking_metrics(["a", "b"], {"a", "b"}, 2)["ndcg_at_k"] == 1


@pytest.mark.parametrize("relevant,k", [(set(), 1), ({"a"}, 0), ({"a"}, -1)])
def test_metrics_reject_undefined_inputs(relevant, k):
    with pytest.raises(ValueError):
        ranking_metrics([], relevant, k)


def test_partial_missing_relevance_is_rejected():
    dataset = load_dataset()
    dataset.scenarios[0].sessions[-1].queries[0].expected_memory_ids.append("nonexistent")
    with pytest.raises(ValueError, match="Every expected memory"):
        validate_dataset_difficulty(dataset, top_k=dataset.default_top_k)


def test_pairing_uses_scenario_and_session_not_only_query_id():
    systems = [
        {
            "system": system,
            "scenarios": [
                {
                    "scenario_id": scenario,
                    "sessions": [
                        {
                            "session_id": "same-session",
                            "queries": [
                                {
                                    "query_id": "same-query",
                                    "top1_hit": system == "aft",
                                    "hit": True,
                                }
                            ],
                        }
                    ],
                }
                for scenario in ("first", "second")
            ],
        }
        for system in ("naive_cosine", "aft")
    ]
    rows = _build_pairwise_comparisons(systems, n_bootstrap=20)
    assert all(row["n_queries"] == 2 for row in rows)
    assert rows[0]["diff"] == 1


def test_manifest_divergence_and_export(tmp_path):
    dataset = load_dataset()
    dataset.scenarios = dataset.scenarios[:1]
    result = run_shared_history(dataset, seed=7, n_bootstrap=20)
    assert result["manifest"]["seed"] == 7
    assert result["manifest"]["engine_config"]
    assert result["manifest"]["source_files"]
    assert len(result["manifest"]["dataset_sha256"]) == 64
    assert result["state_divergence"]
    assert all("mrr_at_k" in system["aggregate_metrics"] for system in result["systems"])
    write_results(
        result,
        out_json=tmp_path / "r.json",
        out_md=tmp_path / "r.md",
        out_protocol=tmp_path / "p.json",
    )
    assert json.loads((tmp_path / "r.json").read_text()) == result
    assert json.loads((tmp_path / "p.json").read_text())["manifest"] == result["manifest"]
    assert "Pairwise vs cosine" in (tmp_path / "r.md").read_text()
    # Detect a changed memory even if affect remains identical.
    changed = deepcopy(result["systems"])
    ref = changed[0]["scenarios"][0]["trace"]
    target = changed[1]["scenarios"][0]["trace"]
    index = next(i for i, t in enumerate(ref) if t["snapshot"]["memories"])
    target[index] = deepcopy(ref[index])
    target[index]["snapshot"]["memories"][0]["content"] = "changed"
    target[index]["state_sha256"] = "changed digest"
    row = state_divergence(changed)[index]
    assert not row["state_equal"]
    assert row["affective_state_equal"]
    assert row["different_memory_count"] == 1
    target[index]["input_sha256"] = "different event"
    with pytest.raises(ValueError, match="alignment"):
        state_divergence(changed)
