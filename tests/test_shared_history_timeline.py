"""Interleaved shared-history dataset: timeline order, gate share, read-only control."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.realistic.runner import (
    InterleavedReplayDataset,
    analyze_dataset_difficulty,
    count_boundary_queries,
    count_encode_after_recall,
    count_neutral_queries,
    dataset_declares_interleave,
    iter_dataset_queries,
    load_dataset,
    run_benchmark,
    session_timeline,
    validate_dataset_difficulty,
)
from benchmarks.realistic.shared_history import run_shared_history
from emotional_memory import EmotionalMemory
from emotional_memory.engine_shared import is_query_affect_neutral

ROOT = Path(__file__).resolve().parents[1]
REPLAY_V1_PATH = ROOT / "benchmarks" / "datasets" / "realistic_recall_v1.json"
TIMELINE_PATH = ROOT / "benchmarks" / "datasets" / "shared_history_recall_v1.json"
GATE_TAU = 0.2


def load_timeline_dataset() -> InterleavedReplayDataset:
    dataset = load_dataset(TIMELINE_PATH)
    assert isinstance(dataset, InterleavedReplayDataset)
    return dataset


def test_v1_shared_history_protocol_mode_unchanged() -> None:
    dataset = load_dataset(REPLAY_V1_PATH)
    dataset = dataset.model_copy(update={"scenarios": dataset.scenarios[:1]})
    result = run_shared_history(dataset, seed=7, n_bootstrap=8, dataset_path=REPLAY_V1_PATH)
    assert result["protocol"]["mode"] == "shared_history_v1"
    assert "event_order" not in result["protocol"]
    assert "read_only_retrieves" not in result["protocol"]


def test_v1_dump_omits_timeline_and_interleave_extensions() -> None:
    dataset = load_dataset(REPLAY_V1_PATH)
    dumped = dataset.model_dump(mode="json")
    assert set(dumped) == {"name", "version", "description", "default_top_k", "scenarios"}
    assert set(dumped["scenarios"][0]["sessions"][0]) == {
        "session_id",
        "description",
        "events",
        "queries",
    }
    assert not dataset_declares_interleave(dataset)
    assert count_encode_after_recall(dataset) == 0


def test_shared_history_timeline_dataset_has_encode_after_recall() -> None:
    dataset = load_timeline_dataset()
    encode_after_recall = count_encode_after_recall(dataset)
    assert encode_after_recall > 0
    assert any(
        session_timeline(session)
        for scenario in dataset.scenarios
        for session in scenario.sessions
    )
    assert dataset.interleave is True
    assert dataset.name == "shared_history_recall_v1"


def test_shared_history_timeline_neutral_share_matches_declaration() -> None:
    dataset = load_timeline_dataset()
    assert dataset.query_affect_gate_tau == GATE_TAU
    assert count_neutral_queries(dataset, tau=GATE_TAU) == dataset.declared_neutral_query_count
    queries = iter_dataset_queries(dataset)
    assert dataset.declared_neutral_query_count / len(queries) == pytest.approx(
        dataset.declared_neutral_query_share, abs=1e-4
    )
    assert count_boundary_queries(dataset, tau=GATE_TAU) == dataset.declared_boundary_query_count
    assert dataset.declared_neutral_query_count > 0
    assert dataset.declared_boundary_query_count > 0
    assert any(query.state is not None and abs(query.state.valence) >= 0.5 for query in queries)


def test_shared_history_timeline_expected_ids_already_encoded() -> None:
    dataset = load_timeline_dataset()
    profile = validate_dataset_difficulty(dataset, top_k=dataset.default_top_k)
    assert profile["minimum_candidate_count"] > dataset.default_top_k
    for scenario in dataset.scenarios:
        encoded: set[str] = set()
        for session in scenario.sessions:
            timeline = session_timeline(session)
            assert timeline is not None
            events = {event.memory_id: event for event in session.events}
            queries = {query.query_id: query for query in session.queries}
            for step in timeline:
                if step.kind == "encode":
                    encoded.add(events[step.id].memory_id)
                    continue
                query = queries[step.id]
                assert set(query.expected_memory_ids) <= encoded


def test_shared_history_timeline_repeat_reports_seed_7() -> None:
    dataset = load_timeline_dataset()
    before = dataset.model_dump(mode="json")
    first = run_shared_history(dataset, seed=7, n_bootstrap=20, dataset_path=TIMELINE_PATH)
    assert first == run_shared_history(dataset, seed=7, n_bootstrap=20, dataset_path=TIMELINE_PATH)
    assert dataset.model_dump(mode="json") == before
    assert first["protocol"]["mode"] == "shared_history_timeline"
    assert first["protocol"]["event_order"] == "dataset_timeline"
    kinds = [
        event["kind"]
        for scenario in first["systems"][0]["scenarios"]
        for event in scenario["trace"]
    ]
    assert "retrieve" in kinds and "encode" in kinds
    first_retrieve = kinds.index("retrieve")
    assert "encode" in kinds[first_retrieve + 1 :]


def test_shared_history_timeline_gated_always_on_routes_diverge_on_neutral_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_timeline_dataset()
    calls: list[tuple[float, bool]] = []
    original = EmotionalMemory.retrieve_with_explanations

    def spy(self, query: str, top_k: int = 5, **kwargs):
        query_affect = kwargs.get("query_affect")
        valence = 0.0 if query_affect is None else float(query_affect.valence)
        calls.append((valence, kwargs.get("precomputed_weights") is not None))
        return original(self, query, top_k=top_k, **kwargs)

    monkeypatch.setattr(EmotionalMemory, "retrieve_with_explanations", spy)
    run_shared_history(dataset, seed=7, n_bootstrap=8, dataset_path=TIMELINE_PATH)
    query_count = len(iter_dataset_queries(dataset))
    assert len(calls) == 3 * query_count
    gated = calls[query_count : 2 * query_count]
    always_on = calls[2 * query_count :]
    queries = iter_dataset_queries(dataset)
    neutral_diverged = 0
    for query, gated_call, always_call in zip(queries, gated, always_on, strict=True):
        assert query.state is not None
        valence = query.state.valence
        if is_query_affect_neutral(valence, GATE_TAU):
            assert gated_call[1] is True
            assert always_call[1] is False
            neutral_diverged += 1
        else:
            assert gated_call[1] is False
            assert always_call[1] is False
    assert neutral_diverged == dataset.declared_neutral_query_count


def test_shared_history_read_only_retrieves_do_not_persist_writebacks() -> None:
    dataset = load_timeline_dataset()
    writable = run_shared_history(dataset, seed=7, n_bootstrap=8, dataset_path=TIMELINE_PATH)
    readonly = run_shared_history(
        dataset,
        seed=7,
        n_bootstrap=8,
        dataset_path=TIMELINE_PATH,
        read_only_retrieves=True,
    )
    assert "read_only_retrieves" not in writable["protocol"]
    assert readonly["protocol"]["read_only_retrieves"] is True

    def counts_after_first_retrieve(result: dict) -> list[int]:
        for scenario in result["systems"][0]["scenarios"]:
            seen_retrieve = False
            for event in scenario["trace"]:
                if event["kind"] == "retrieve":
                    seen_retrieve = True
                    continue
                if seen_retrieve and event["kind"] == "encode":
                    return [
                        memory["tag"]["retrieval_count"]
                        for memory in event["snapshot"]["memories"]
                    ]
        raise AssertionError("no encode after retrieve")

    writable_counts = counts_after_first_retrieve(writable)
    readonly_counts = counts_after_first_retrieve(readonly)
    assert any(count > 0 for count in writable_counts)
    assert all(count == 0 for count in readonly_counts)


def test_comparative_runner_rejects_interleaved_dataset() -> None:
    dataset = load_timeline_dataset()
    with pytest.raises(ValueError, match="require --shared-history"):
        run_benchmark(dataset, systems=["recency"], n_bootstrap=8, seed=7)


def test_shared_history_timeline_difficulty_profile_uses_query_time_candidates() -> None:
    dataset = load_timeline_dataset()
    profile = analyze_dataset_difficulty(dataset, top_k=dataset.default_top_k)
    first = profile["queries"][0]
    assert first["candidate_count"] == 3
    assert first["expected_recency_ranks"]
    assert all(rank > dataset.default_top_k for rank in first["expected_recency_ranks"])
