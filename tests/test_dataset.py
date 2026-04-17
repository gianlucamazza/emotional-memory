"""Validate benchmarks/datasets/affect_reference_v1.jsonl schema and distribution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATASET = Path(__file__).parent.parent / "benchmarks" / "datasets" / "affect_reference_v1.jsonl"
VALID_LABELS = {"joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"}


@pytest.fixture(scope="module")
def examples() -> list[dict]:  # type: ignore[type-arg]
    assert DATASET.exists(), f"Dataset not found: {DATASET}"
    return [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]


def test_minimum_count(examples: list) -> None:  # type: ignore[type-arg]
    assert len(examples) >= 200


def test_schema_fields(examples: list) -> None:  # type: ignore[type-arg]
    required = {"id", "text", "valence", "arousal", "dominance", "expected_label", "source"}
    for ex in examples:
        assert required <= ex.keys(), f"Missing fields in {ex['id']}"


def test_valence_range(examples: list) -> None:  # type: ignore[type-arg]
    for ex in examples:
        assert -1.0 <= ex["valence"] <= 1.0, f"valence out of range: {ex['id']}"


def test_arousal_range(examples: list) -> None:  # type: ignore[type-arg]
    for ex in examples:
        assert 0.0 <= ex["arousal"] <= 1.0, f"arousal out of range: {ex['id']}"


def test_dominance_range(examples: list) -> None:  # type: ignore[type-arg]
    for ex in examples:
        assert -1.0 <= ex["dominance"] <= 1.0, f"dominance out of range: {ex['id']}"


def test_valid_labels(examples: list) -> None:  # type: ignore[type-arg]
    for ex in examples:
        assert ex["expected_label"] in VALID_LABELS, f"Unknown label: {ex['expected_label']}"


def test_no_empty_text(examples: list) -> None:  # type: ignore[type-arg]
    for ex in examples:
        assert ex["text"].strip(), f"Empty text in {ex['id']}"


def test_unique_ids(examples: list) -> None:  # type: ignore[type-arg]
    ids = [ex["id"] for ex in examples]
    assert len(ids) == len(set(ids)), "Duplicate IDs found"


def test_quadrant_coverage(examples: list) -> None:  # type: ignore[type-arg]
    """All four Russell quadrants must be represented with ≥20 examples each."""
    q = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    for ex in examples:
        v, a = ex["valence"], ex["arousal"]
        if v >= 0 and a >= 0.5:
            q["Q1"] += 1
        elif v < 0 and a >= 0.5:
            q["Q2"] += 1
        elif v < 0:
            q["Q3"] += 1
        else:
            q["Q4"] += 1
    for quad, count in q.items():
        assert count >= 20, f"Quadrant {quad} underrepresented: {count} examples"


def test_label_distribution(examples: list) -> None:  # type: ignore[type-arg]
    """At least 3 distinct Plutchik labels must be present."""
    labels = {ex["expected_label"] for ex in examples}
    assert len(labels) >= 3
