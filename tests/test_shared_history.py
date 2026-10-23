"""Behavioral checks for the offline longitudinal replay contract."""

from datetime import UTC, datetime

import pytest

from benchmarks.realistic.runner import load_dataset
from benchmarks.realistic.shared_history import (
    ARMS,
    EPOCH,
    SharedHistoryAdapter,
    controlled_runtime,
    run_shared_history,
)


def small_dataset():
    dataset = load_dataset()
    return dataset.model_copy(update={"scenarios": dataset.scenarios[:1]})


def test_shared_replay_exact_repeat_and_common_inputs():
    dataset = small_dataset()
    before = dataset.model_dump(mode="json")
    first = run_shared_history(dataset, seed=7, n_bootstrap=20)
    assert first == run_shared_history(dataset, seed=7, n_bootstrap=20)
    assert dataset.model_dump(mode="json") == before
    reports = [system["scenarios"][0] for system in first["systems"]]
    assert all(r["checkpoint"] == reports[0]["checkpoint"] for r in reports)
    inputs = [
        [(t["index"], t["timestamp"], t["input_sha256"]) for t in r["trace"]] for r in reports
    ]
    assert all(events == inputs[0] for events in inputs)
    for report in reports:
        queries = [t for t in report["trace"] if t["kind"] == "retrieve"]
        assert queries
        assert any(
            m["tag"]["retrieval_count"] > 0 for t in queries for m in t["snapshot"]["memories"]
        )


def test_shared_arms_do_not_share_store_config_or_tags():
    arms = [SharedHistoryAdapter(arm, "scenario", 0) for arm in ARMS]
    try:
        for arm in arms:
            arm.begin_session("one")
            arm.encode(
                memory_alias="a",
                content="same text",
                valence=0.8,
                arousal=0.8,
                metadata={"nested": [1]},
            )
        before = arms[1].snapshot()
        assert arms[0].config is not arms[1].config
        arms[0].retrieve("same text", top_k=1, valence=0.8, arousal=0.8)
        arms[0].engine.list_all()[0].metadata["nested"].append(2)
        assert arms[1].snapshot() == before
        assert arms[2].snapshot() == before
    finally:
        for arm in arms:
            arm.close()


def test_runtime_restored_on_exception():
    import emotional_memory.engine as engine
    import emotional_memory.models as models

    old_datetime, old_uuid = engine.datetime, models.uuid
    with pytest.raises(RuntimeError), controlled_runtime(EPOCH, "fixture-id"):
        assert engine.datetime.now(tz=UTC) == EPOCH
        raise RuntimeError("failure")
    assert engine.datetime is old_datetime
    assert models.uuid is old_uuid
    assert datetime.now(tz=UTC) != EPOCH


def test_shared_history_requires_recorded_query_affect():
    dataset = small_dataset().model_copy(deep=True)
    dataset.scenarios[0].sessions[-1].queries[0].state = None
    with pytest.raises(ValueError, match="recorded query affect"):
        run_shared_history(dataset, n_bootstrap=20)


def test_shared_history_resets_each_scenario():
    dataset = small_dataset()
    duplicate = dataset.scenarios[0].model_copy(deep=True)
    duplicate.scenario_id = "independent"
    dataset.scenarios.append(duplicate)
    result = run_shared_history(dataset, n_bootstrap=20)
    for system in result["systems"]:
        first, second = system["scenarios"]
        assert first["checkpoint"] == second["checkpoint"]
        assert second["sessions"][0]["memory_count_start"] == 0
        assert not second["sessions"][0]["state_loaded_from_store"]
