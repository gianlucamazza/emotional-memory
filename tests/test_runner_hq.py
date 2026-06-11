"""Smoke tests for the Addendum Q runner — no real LLM, hash embedder only."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest

pytest.importorskip("tqdm")

from benchmarks.appraisal_confound.runner_hq import (
    _gate_records,
    _GateDataset,
    _LLMGateClassifier,
    _oracle_mapping,
    _OracleGateClassifier,
    _validate_dataset,
    dry_run_oracle,
)
from benchmarks.datasets.generate_v5_gate import _leakage_ok


def _synthetic_scenario(idx: int) -> dict[str, Any]:
    sid = f"q{idx:02d}_synth"
    topics = [
        "harbour ledger",
        "garden gate",
        "engine bay",
        "paper lanterns",
        "stone arch",
        "radio mast",
    ]
    events = [
        f"Scenario {idx} slot {n}: the {topics[n]} work, take {idx * 10 + n}." for n in range(6)
    ]
    event_ids = [f"{sid}_e{n}" for n in range(1, 7)]
    queries = [
        {
            "query_id": f"{sid}_q1",
            "query": f"What of the {topics[4]} work in scenario {idx}, take {idx * 10 + 4}?",
            "expected_memory_ids": [event_ids[4]],
            "challenge_type": "semantic_confound",
            "gate_label": "affect_free",
        },
        {
            "query_id": f"{sid}_q2",
            "query": f"The earlier {topics[1]} moment in scenario {idx}, take {idx * 10 + 1}?",
            "expected_memory_ids": [event_ids[1]],
            "challenge_type": "recency_confound",
            "gate_label": "affect_free",
        },
        {
            "query_id": f"{sid}_q3",
            "query": f"Which moment in scenario {idx} matched how it all ended up?",
            "expected_memory_ids": [event_ids[3]],
            "challenge_type": "affect_congruent_tiebreak",
            "gate_label": "affective",
        },
        {
            "query_id": f"{sid}_q4",
            "query": f"When did the long slog finally break in scenario {idx}?",
            "expected_memory_ids": [event_ids[3]],
            "challenge_type": "affective_arc_blind",
            "gate_label": "affective",
        },
    ]
    return {
        "scenario_id": sid,
        "description": f"Synthetic scenario {idx}",
        "sessions": [
            {
                "session_id": "session_1",
                "description": "Opening.",
                "events": [
                    {"memory_id": event_ids[n], "content": events[n], "metadata": {}}
                    for n in range(3)
                ],
                "queries": [],
            },
            {
                "session_id": "session_2",
                "description": "Later.",
                "events": [
                    {"memory_id": event_ids[n], "content": events[n], "metadata": {}}
                    for n in range(3, 6)
                ],
                "queries": queries,
            },
        ],
    }


def _synthetic_dataset(n: int = 3) -> dict[str, Any]:
    return {
        "name": "synthetic_v5_gate",
        "version": "0.0.1",
        "description": "Synthetic mixed-gate dataset for smoke tests.",
        "default_top_k": 2,
        "scenarios": [_synthetic_scenario(i + 1) for i in range(n)],
    }


def test_validate_dataset_accepts_synthetic() -> None:
    ds = _GateDataset.model_validate(_synthetic_dataset())
    _validate_dataset(ds, top_k=ds.default_top_k)


def test_validate_dataset_rejects_label_imbalance() -> None:
    raw = _synthetic_dataset(1)
    raw["scenarios"][0]["sessions"][1]["queries"][0]["challenge_type"] = "affective_arc_blind"
    raw["scenarios"][0]["sessions"][1]["queries"][0]["gate_label"] = "affective"
    ds = _GateDataset.model_validate(raw)
    with pytest.raises(ValueError, match="gate-label balance"):
        _validate_dataset(ds, top_k=ds.default_top_k)


def test_validate_dataset_rejects_label_challenge_mismatch() -> None:
    raw = _synthetic_dataset(1)
    raw["scenarios"][0]["sessions"][1]["queries"][2]["gate_label"] = "affect_free"
    ds = _GateDataset.model_validate(raw)
    with pytest.raises(ValueError, match="inconsistent"):
        _validate_dataset(ds, top_k=ds.default_top_k)


def test_leakage_gate() -> None:
    target = "The crane lifted the cracked bell out of the tower at dawn."
    distractor = "The crane dropped the spare bell against the tower wall."
    # Neutral phrasing overlapping both: OK.
    assert _leakage_ok("the crane and bell moment at the tower", target, distractor)
    # Phrasing lexically loaded toward the target: rejected.
    assert not _leakage_ok(
        "the crane lifted cracked bell out of tower at dawn moment", target, distractor
    )


def test_oracle_gate_classifier_logs_and_maps() -> None:
    clf = _OracleGateClassifier({"q one": "affect_free"})
    assert clf.classify("q one") == "affect_free"
    assert clf.classify("unseen") == "affective"  # pre-reg fallback
    assert clf.log == [("q one", "affect_free"), ("unseen", "affective")]


def test_llm_gate_classifier_parses_and_falls_back() -> None:
    def good_llm(prompt: str, schema: dict[str, Any]) -> str:
        return json.dumps({"gate_label": "affect_free"})

    clf = _LLMGateClassifier(good_llm)
    assert clf.classify("factual?") == "affect_free"
    # Cached second call, log grows by query text.
    assert clf.classify("factual?") == "affect_free"
    assert len(clf.log) == 2

    def bad_llm(prompt: str, schema: dict[str, Any]) -> str:
        raise RuntimeError("boom")

    clf2 = _LLMGateClassifier(bad_llm)
    assert clf2.classify("anything") == "affective"  # pre-reg fallback


def test_gate_records_match_by_query_text() -> None:
    # Hl3 lesson: records must survive missing/extra log entries.
    records = _gate_records(
        [("q A", "affective")],
        {"q A": "affective", "q B": "affect_free"},
    )
    by_query = {r["query"]: r for r in records}
    assert by_query["q A"]["predicted_label"] == "affective"
    assert by_query["q B"]["predicted_label"] == "unknown"


def test_dry_run_oracle_equivalence(tmp_path: Path) -> None:
    """Gate-off path must reproduce naive_cosine top1 on every affect-free query."""
    ds_path = tmp_path / "synthetic_v5_gate.json"
    ds_path.write_text(json.dumps(_synthetic_dataset(3)), encoding="utf-8")
    # Raises AssertionError on any affect-free top1 mismatch.
    dry_run_oracle(ds_path, workdir=tmp_path / "work")


def test_oracle_mapping_covers_all_queries() -> None:
    ds = _GateDataset.model_validate(_synthetic_dataset(2))
    mapping = _oracle_mapping(ds)
    assert len(mapping) == 8
    assert set(mapping.values()) == {"affective", "affect_free"}
