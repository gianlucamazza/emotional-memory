from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.human_eval.pipeline import (
    build_packets,
    krippendorff_alpha,
    load_ratings,
    summarize_ratings,
    write_packets,
    write_summary,
)
from benchmarks.realistic.runner import load_dataset, run_benchmark, write_results

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _realistic_results(tmp_path: Path) -> Path:
    dataset = load_dataset()
    results = run_benchmark(dataset)
    out_json = tmp_path / "realistic-results.json"
    write_results(
        results,
        out_json=out_json,
        out_md=tmp_path / "realistic-results.md",
        out_protocol=tmp_path / "realistic-results.protocol.json",
    )
    return out_json


def test_build_packets_from_realistic_results(tmp_path: Path) -> None:
    results_path = _realistic_results(tmp_path)
    packets = build_packets(results_path=results_path)

    assert packets["name"] == "human_eval_pilot_v1"
    assert packets["conditions"] == ["aft", "naive_cosine"]
    assert packets["packet_count"] == len(packets["packets"]) == 10
    assert all(len(packet["conditions"]) == 2 for packet in packets["packets"])
    assert all(
        condition["query_cards"]
        for packet in packets["packets"]
        for condition in packet["conditions"]
    )


def test_write_packets_creates_template(tmp_path: Path) -> None:
    results_path = _realistic_results(tmp_path)
    payload = build_packets(results_path=results_path)
    packets_path = tmp_path / "packets.json"
    template_path = tmp_path / "ratings_template.jsonl"
    ratings_path = tmp_path / "ratings.jsonl"

    write_packets(
        payload,
        out_path=packets_path,
        template_path=template_path,
    )

    assert json.loads(packets_path.read_text(encoding="utf-8"))["name"] == "human_eval_pilot_v1"
    lines = [line for line in template_path.read_text(encoding="utf-8").splitlines() if line]
    assert lines
    first = json.loads(lines[0])
    assert "ratings" in first
    assert first["rater_id"] == ""
    assert not ratings_path.exists()


def test_human_eval_instruction_sheet_exists() -> None:
    instructions = FIXTURES.parents[1] / "benchmarks" / "human_eval" / "rater_instructions.md"

    assert instructions.exists()
    assert "Human-Eval Pilot v1 Instructions" in instructions.read_text(encoding="utf-8")


def test_load_ratings_requires_filled_ratings_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "ratings.jsonl"

    with pytest.raises(FileNotFoundError, match="ratings file not found"):
        load_ratings(missing_path)


def test_repo_does_not_ship_placeholder_human_eval_summaries() -> None:
    root = Path(__file__).resolve().parents[1]

    assert not (root / "benchmarks" / "human_eval" / "summary.json").exists()
    assert not (root / "benchmarks" / "human_eval" / "summary.md").exists()


def test_summarize_ratings_rejects_placeholder_template_records() -> None:
    template_records = [
        {
            "scenario_id": "customer_repair_arc",
            "condition": "aft",
            "rater_id": "",
            "ratings": {
                "affective_coherence": None,
                "usefulness": None,
                "continuity": None,
                "plausibility": None,
            },
            "note": "",
        }
    ]

    with pytest.raises(ValueError, match="No completed ratings found"):
        summarize_ratings(template_records)


def test_summarize_ratings_and_write_summary(tmp_path: Path) -> None:
    fixture_path = FIXTURES / "human_eval_completed.jsonl"
    ratings = load_ratings(fixture_path)
    ratings.append(
        {
            "scenario_id": "family_health_reassurance",
            "condition": "aft",
            "rater_id": "r2",
            "ratings": {
                "affective_coherence": 4,
                "usefulness": None,
                "continuity": 4,
                "plausibility": None,
            },
            "note": "Missing two dimensions.",
        }
    )

    summary = summarize_ratings(ratings)
    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "summary.md"
    write_summary(summary, out_json=out_json, out_md=out_md)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["conditions"]
    assert payload["completed_ratings_count"] == 2
    assert payload["unique_rater_count"] == 1
    assert len(payload["incomplete_records"]) == 1
    assert not payload["invalid_records"]
    assert any(item["condition"] == "aft" for item in payload["conditions"])
    assert "# Human Evaluation Pilot Summary" in out_md.read_text(encoding="utf-8")
    assert "Unique raters: 1" in out_md.read_text(encoding="utf-8")
    assert "Incomplete records: 1" in out_md.read_text(encoding="utf-8")
    # Single rater → alpha should be None for all dims
    assert all(v is None for v in payload["krippendorff_alpha_by_dimension"].values())


# ---------------------------------------------------------------------------
# Krippendorff's alpha
# ---------------------------------------------------------------------------


def test_krippendorff_alpha_perfect_agreement() -> None:
    # 3 raters, 3 units, all identical → alpha = 1.0
    data: list[list[float | None]] = [
        [1.0, 1.0, 1.0],
        [3.0, 3.0, 3.0],
        [5.0, 5.0, 5.0],
    ]
    assert krippendorff_alpha(data) == pytest.approx(1.0)


def test_krippendorff_alpha_known_value() -> None:
    # 3 raters, 3 units, ordinal {1,2,3}
    # Item 1: [1,2,3] (all different); Items 2,3: perfect agreement
    # Hand-computed: alpha = 2/3 ≈ 0.6667 (Krippendorff coincidence-matrix)
    data: list[list[float | None]] = [
        [1.0, 2.0, 3.0],
        [1.0, 1.0, 1.0],
        [3.0, 3.0, 3.0],
    ]
    result = krippendorff_alpha(data, level="ordinal")
    assert result == pytest.approx(2 / 3, abs=1e-6)


def test_krippendorff_alpha_handles_missing_values() -> None:
    # Mix of missing and present values; should return a finite float
    data: list[list[float | None]] = [
        [4.0, None, 5.0],
        [2.0, 3.0, None],
        [None, 1.0, 2.0],
        [5.0, 4.0, 5.0],
    ]
    result = krippendorff_alpha(data)
    assert result is not None
    assert -1.5 < result <= 1.0


def test_krippendorff_alpha_single_rater_returns_none() -> None:
    data: list[list[float | None]] = [[3.0], [4.0], [5.0]]
    assert krippendorff_alpha(data) is None


def test_summarize_ratings_emits_alpha_per_dimension(tmp_path: Path) -> None:
    fixture_path = FIXTURES / "human_eval_three_raters.jsonl"
    ratings = load_ratings(fixture_path)
    summary = summarize_ratings(ratings)

    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "summary.md"
    write_summary(summary, out_json=out_json, out_md=out_md)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    alpha_map = payload["krippendorff_alpha_by_dimension"]

    assert set(alpha_map.keys()) == {
        "affective_coherence",
        "usefulness",
        "continuity",
        "plausibility",
    }
    # With 3 raters, alpha should be defined (not None) for all dims
    assert all(v is not None for v in alpha_map.values())
    # Values must be in a reasonable range
    assert all(-1.5 < v <= 1.0 for v in alpha_map.values())

    md = out_md.read_text(encoding="utf-8")
    assert "## Inter-Rater Agreement" in md
    assert "| Dimension | alpha |" in md
    assert "affective_coherence" in md
