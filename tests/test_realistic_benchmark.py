from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from benchmarks.realistic.runner import (
    analyze_dataset_difficulty,
    build_protocol_metadata,
    load_dataset,
    run_benchmark,
    validate_dataset_difficulty,
    write_results,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_realistic_dataset_loads() -> None:
    dataset = load_dataset()

    assert dataset.name == "realistic_recall_v1"
    assert dataset.default_top_k == 2
    assert dataset.version == "1.3"
    assert dataset.scenarios
    assert len(dataset.scenarios) >= 10
    assert all(
        session.queries for scenario in dataset.scenarios for session in scenario.sessions[1:]
    )


def test_realistic_protocol_metadata_is_complete() -> None:
    dataset = load_dataset()
    difficulty = validate_dataset_difficulty(dataset, top_k=dataset.default_top_k)
    protocol = build_protocol_metadata(
        dataset,
        system_names=["aft", "naive_cosine", "recency"],
        difficulty_profile=difficulty,
    )

    assert protocol["benchmark"] == "realistic_recall_v1"
    assert protocol["systems"] == ["aft", "naive_cosine", "recency"]
    assert protocol["dataset"]["scenario_count"] == len(dataset.scenarios)
    assert protocol["primary_metrics"] == ["top1_accuracy", "hit@k"]
    assert protocol["difficulty_profile"]["minimum_candidate_count"] > dataset.default_top_k
    assert protocol["difficulty_profile"]["nontrivial_query_rate"] >= 0.5
    assert protocol["difficulty_profile"]["challenge_type_counts"]["semantic_confound"] >= 8
    assert any(
        "controlled replay benchmark" in item for item in protocol["interpretation_guardrails"]
    )


def test_realistic_dataset_validation_rejects_trivial_candidate_windows() -> None:
    dataset = load_dataset().model_copy(update={"default_top_k": 6})

    with pytest.raises(ValueError, match="candidate_count <= top_k"):
        validate_dataset_difficulty(dataset, top_k=dataset.default_top_k)


def test_realistic_dataset_contains_nontrivial_queries_per_scenario() -> None:
    dataset = load_dataset()
    difficulty = analyze_dataset_difficulty(dataset, top_k=dataset.default_top_k)

    assert difficulty["nontrivial_query_rate"] >= 0.5
    assert difficulty["challenge_type_counts"]["affective_arc"] >= 4
    assert difficulty["challenge_type_counts"]["semantic_confound"] >= 8
    assert all(report["nontrivial_query_count"] >= 1 for report in difficulty["scenario_reports"])


def test_realistic_benchmark_runs_and_writes_outputs(tmp_path: Path) -> None:
    dataset = load_dataset()
    results = run_benchmark(dataset)

    assert [system["system"] for system in results["systems"]] == [
        "aft",
        "naive_cosine",
        "recency",
    ]
    aft = next(system for system in results["systems"] if system["system"] == "aft")
    naive = next(system for system in results["systems"] if system["system"] == "naive_cosine")
    recency = next(system for system in results["systems"] if system["system"] == "recency")
    assert aft["aggregate_metrics"]["query_count"] >= 20
    assert aft["aggregate_metrics"]["stateful_session_rate"] > 0.0
    assert aft["aggregate_metrics"]["top1_accuracy"] > naive["aggregate_metrics"]["top1_accuracy"]
    assert recency["aggregate_metrics"]["hit_at_k"] < 1.0
    assert (
        recency["aggregate_metrics"]["top1_accuracy"] < aft["aggregate_metrics"]["top1_accuracy"]
    )
    assert results["difficulty_profile"]["nontrivial_query_rate"] >= 0.5
    assert all(system["challenge_type_metrics"] for system in results["systems"])
    aft_affective = next(
        metrics
        for metrics in aft["challenge_type_metrics"]
        if metrics["challenge_type"] == "affective_arc"
    )
    naive_affective = next(
        metrics
        for metrics in naive["challenge_type_metrics"]
        if metrics["challenge_type"] == "affective_arc"
    )
    assert aft_affective["top1_accuracy"] > naive_affective["top1_accuracy"]
    assert all(
        query["candidate_count"] > query["top_k"]
        for system in results["systems"]
        for scenario in system["scenarios"]
        for session in scenario["sessions"]
        for query in session["queries"]
    )

    out_json = tmp_path / "results.json"
    out_md = tmp_path / "results.md"
    out_protocol = tmp_path / "results.protocol.json"
    write_results(results, out_json=out_json, out_md=out_md, out_protocol=out_protocol)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    protocol = json.loads(out_protocol.read_text(encoding="utf-8"))

    assert payload["benchmark"] == "realistic_recall_v1"
    assert protocol["benchmark"] == "realistic_recall_v1"
    assert len(payload["systems"]) == 3
    assert "challenge_type_counts" in protocol["difficulty_profile"]
    assert "# Realistic Replay Benchmark" in out_md.read_text(encoding="utf-8")
    assert "## By Challenge Type" in out_md.read_text(encoding="utf-8")
    assert "Headline metric: `top1_accuracy`." in out_md.read_text(encoding="utf-8")
