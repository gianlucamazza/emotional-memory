"""Regression tests for pareto_runner pure functions (Add. J)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("tqdm")

from benchmarks.locomo.dataset import (
    Conversation,
    LoCoMoDataset,
    QAPair,
    Session,
)
from benchmarks.locomo.pareto_runner import (
    PARETO_CATEGORIES,
    _compute_pareto_table,
    _stratified_subsample,
)

_CATEGORY_NAMES = ["multi_hop", "temporal", "open_domain", "single_hop"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    config_id: str,
    aggregate_f1: float,
    category_f1: dict[str, float],
) -> dict[str, Any]:
    return {
        "config_id": config_id,
        "scores": {
            "aggregate": {"f1": aggregate_f1, "judge_acc": 0.0},
            "by_category": {
                name: {"f1": f1, "judge_acc": 0.0} for name, f1 in category_f1.items()
            },
        },
        "predictions": [],
    }


def _uniform_cat(f1: float) -> dict[str, float]:
    return dict.fromkeys(_CATEGORY_NAMES, f1)


def _make_dataset(qa_per_category: dict[int, int]) -> LoCoMoDataset:
    """Synthetic LoCoMoDataset with qa_per_category[cat] QA pairs per category."""
    qa_pairs: list[QAPair] = [
        QAPair(
            question=f"Q cat{cat} n{i}",
            answer=f"A cat{cat} n{i}",
            category=cat,
            evidence_dia_ids=[],
        )
        for cat, count in qa_per_category.items()
        for i in range(count)
    ]
    conv = Conversation(
        sample_id="conv0",
        speaker_a="A",
        speaker_b="B",
        sessions=[Session(session_num=1, date_time="2024-01-01", turns=[])],
        qa_pairs=qa_pairs,
    )
    return LoCoMoDataset(conversations=[conv])


# ---------------------------------------------------------------------------
# _compute_pareto_table — Hj1 criterion
# ---------------------------------------------------------------------------


def test_compute_pareto_table_hj1_fail() -> None:
    """Hj1=FAIL when no AFT config reaches naive_rag F1 on any category."""
    naive_f1 = 0.25
    results: dict[str, Any] = {
        "configs": [
            _make_config("W0", 0.13, _uniform_cat(0.10)),
            _make_config("W1", 0.14, _uniform_cat(0.12)),
            # W2 best AFT but still below naive_rag on every category (0.20 < 0.25)
            _make_config("W2", 0.18, _uniform_cat(0.20)),
            _make_config("naive_rag", 0.21, _uniform_cat(naive_f1)),
        ]
    }
    table = _compute_pareto_table(results)
    assert table["hj1_verdict"] == "FAIL"
    # No non-W0 row should have closes_gap_to_naive=True on any category
    for row in table["rows"]:
        if row["config_id"] in ("W0", "naive_rag"):
            continue
        for cat_name in _CATEGORY_NAMES:
            assert not row["by_category"].get(cat_name, {}).get("closes_gap_to_naive", False)


def test_compute_pareto_table_hj1_pass() -> None:
    """Hj1=PASS when at least one (W, C) pair has aft_W.F1 >= naive_rag.F1."""
    naive_f1 = 0.20
    # W2 beats naive_rag on multi_hop only (0.22 > 0.20)
    w2_cat = _uniform_cat(0.10)
    w2_cat["multi_hop"] = 0.22
    results: dict[str, Any] = {
        "configs": [
            _make_config("W0", 0.13, _uniform_cat(0.10)),
            _make_config("W2", 0.15, w2_cat),
            _make_config("naive_rag", 0.20, _uniform_cat(naive_f1)),
        ]
    }
    table = _compute_pareto_table(results)
    assert table["hj1_verdict"] == "PASS"
    w2_row = next(r for r in table["rows"] if r["config_id"] == "W2")
    assert w2_row["by_category"]["multi_hop"]["closes_gap_to_naive"] is True
    # Other categories below naive_rag → closes_gap_to_naive=False
    for cat in ("temporal", "open_domain", "single_hop"):
        assert w2_row["by_category"][cat]["closes_gap_to_naive"] is False


# ---------------------------------------------------------------------------
# _stratified_subsample
# ---------------------------------------------------------------------------


def test_stratified_subsample_deterministic() -> None:
    """Same seed produces identical sample on two independent calls."""
    dataset = _make_dataset({1: 100, 2: 100, 3: 100, 4: 100})
    s1 = _stratified_subsample(dataset, seed=42, per_category=50)
    s2 = _stratified_subsample(dataset, seed=42, per_category=50)
    ids1 = sorted((qa.question, qa.answer) for c in s1.conversations for qa in c.qa_pairs)
    ids2 = sorted((qa.question, qa.answer) for c in s2.conversations for qa in c.qa_pairs)
    assert ids1 == ids2


def test_stratified_subsample_per_category_count() -> None:
    """Subsample returns exactly min(per_category, available) QA per category."""
    # 100 per category, per_category=50 → exactly 50 each
    dataset = _make_dataset({1: 100, 2: 100, 3: 100, 4: 100})
    subsample = _stratified_subsample(dataset, seed=42, per_category=50)
    per_cat = dict.fromkeys(PARETO_CATEGORIES, 0)
    for conv in subsample.conversations:
        for qa in conv.qa_pairs:
            per_cat[qa.category] += 1
    assert all(count == 50 for count in per_cat.values()), per_cat

    # 20 per category (< per_category=50) → takes all available (all-if-fewer)
    dataset_small = _make_dataset({1: 20, 2: 20, 3: 20, 4: 20})
    subsample_small = _stratified_subsample(dataset_small, seed=42, per_category=50)
    per_cat_small = dict.fromkeys(PARETO_CATEGORIES, 0)
    for conv in subsample_small.conversations:
        for qa in conv.qa_pairs:
            per_cat_small[qa.category] += 1
    assert all(count == 20 for count in per_cat_small.values()), per_cat_small
