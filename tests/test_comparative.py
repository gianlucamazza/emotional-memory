"""Smoke tests for the comparative benchmark harness."""

from __future__ import annotations

from benchmarks.comparative.adapters.aft import AFTAdapter
from benchmarks.comparative.adapters.naive_cosine import NaiveCosineAdapter
from benchmarks.comparative.runner import QUERIES, _is_congruent, run_adapter


def _mini_dataset() -> list[dict]:  # type: ignore[type-arg]
    return [
        {
            "id": "t0",
            "text": "I am ecstatic and thrilled!",
            "valence": 0.9,
            "arousal": 0.9,
            "dominance": 0.6,
            "expected_label": "joy",
            "source": "test",
        },
        {
            "id": "t1",
            "text": "Terrified and shaking with panic.",
            "valence": -0.8,
            "arousal": 0.8,
            "dominance": -0.5,
            "expected_label": "fear",
            "source": "test",
        },
        {
            "id": "t2",
            "text": "Deep sadness, utterly hopeless.",
            "valence": -0.8,
            "arousal": 0.2,
            "dominance": -0.6,
            "expected_label": "sadness",
            "source": "test",
        },
        {
            "id": "t3",
            "text": "Peaceful and serenely content.",
            "valence": 0.8,
            "arousal": 0.2,
            "dominance": 0.5,
            "expected_label": "trust",
            "source": "test",
        },
        {
            "id": "t4",
            "text": "Excited about the new project.",
            "valence": 0.7,
            "arousal": 0.7,
            "dominance": 0.4,
            "expected_label": "joy",
            "source": "test",
        },
        {
            "id": "t5",
            "text": "Calm and grateful for everything.",
            "valence": 0.6,
            "arousal": 0.3,
            "dominance": 0.3,
            "expected_label": "trust",
            "source": "test",
        },
    ]


def test_aft_adapter_encode_retrieve() -> None:
    adapter = AFTAdapter()
    adapter.reset()
    ds = _mini_dataset()
    for ex in ds:
        adapter.encode(ex["text"], valence=ex["valence"], arousal=ex["arousal"])
    results = adapter.retrieve("feeling joyful and excited", top_k=3)
    assert len(results) <= 3
    assert all(hasattr(r, "id") and hasattr(r, "text") for r in results)


def test_naive_cosine_adapter_encode_retrieve() -> None:
    adapter = NaiveCosineAdapter()
    adapter.reset()
    ds = _mini_dataset()
    for ex in ds:
        adapter.encode(ex["text"])
    results = adapter.retrieve("hopeless sadness", top_k=2)
    assert len(results) == 2


def test_reset_clears_store() -> None:
    adapter = NaiveCosineAdapter()
    adapter.encode("hello", valence=0.0)
    adapter.reset()
    results = adapter.retrieve("hello", top_k=5)
    assert results == []


def test_run_adapter_returns_metrics() -> None:
    adapter = AFTAdapter()
    metrics = run_adapter(adapter, _mini_dataset(), top_k=3)
    assert metrics["system"] == "aft"
    assert "recall@3" in metrics
    assert 0.0 <= metrics["recall@3"] <= 1.0
    assert metrics["status"] == "ok"


def test_is_congruent_q1() -> None:
    q = QUERIES[0]  # Q1 joy
    assert _is_congruent({"valence": 0.8, "arousal": 0.8}, q)
    assert not _is_congruent({"valence": -0.5, "arousal": 0.8}, q)
    assert not _is_congruent({"valence": 0.8, "arousal": 0.3}, q)


def test_is_congruent_q3() -> None:
    q = QUERIES[2]  # Q3 sadness
    assert _is_congruent({"valence": -0.7, "arousal": 0.2}, q)
    assert not _is_congruent({"valence": 0.5, "arousal": 0.2}, q)
