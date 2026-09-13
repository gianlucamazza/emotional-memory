"""Tests for FlyAffectHost (requires affective-fly).

Skipped automatically when affective-fly is not installed. Main CI does
not install it (the package is not on PyPI). Run locally after:

    make install-fly
    uv run python -m pytest tests/test_fly_adapter.py
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("affective_fly")

from affective_fly import (
    HYPOTHESIS_TAU_APPROACH,
    HYPOTHESIS_TAU_AROUSAL,
    HYPOTHESIS_TAU_VALENCE,
    HostFrame,
    MockFlyCircuit,
    load_measurement_records,
)

from emotional_memory import EmotionalMemory, InMemoryStore
from emotional_memory.integrations.fly import FlyAffectHost


class _FixedEmbedder:
    """Returns the same 4-dim unit vector for every input — enough for tests."""

    def embed(self, text: str) -> list[float]:
        return [0.25, 0.25, 0.25, 0.25]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _journal_frames() -> list[Any]:
    return [
        HostFrame(
            timestamp="2026-09-12T10:00:00Z",
            visual_hash="note-a",
            context={"context": "journal", "note_id": "n1", "sentiment": 0.6},
        ),
        HostFrame(
            timestamp="2026-09-12T10:03:00Z",
            visual_hash="note-b",
            context={
                "context": "review",
                "note_id": "n1",
                "sentiment": -0.5,
                "outcome": -0.4,
            },
        ),
    ]


def test_importable_from_integrations_subpackage() -> None:
    from emotional_memory.integrations import FlyAffect
    from emotional_memory.integrations import FlyAffectHost as Host

    assert Host is FlyAffectHost
    assert FlyAffect.__name__ == "FlyAffect"


def test_host_constructs_its_own_store_and_memory() -> None:
    host = FlyAffectHost(fly_circuit=MockFlyCircuit(seed=42))

    assert host.loop.emotional_memory is host.memory
    assert isinstance(host.memory, EmotionalMemory)
    assert host.mood.tau_valence == HYPOTHESIS_TAU_VALENCE
    assert host.mood.tau_arousal == HYPOTHESIS_TAU_AROUSAL
    assert host.mood.tau_approach == HYPOTHESIS_TAU_APPROACH
    assert host.loop.policy.threshold_act == 0.2
    assert host.loop.launch_gate.threshold_approach == 0.2
    assert host.loop.policy.threshold_calm == 0.0


def test_host_reuses_caller_memory_store_and_embedder() -> None:
    store = InMemoryStore()
    embedder = _FixedEmbedder()
    memory = EmotionalMemory(store=store, embedder=embedder)
    host = FlyAffectHost(
        store=store,
        embedder=embedder,
        memory=memory,
        fly_circuit=MockFlyCircuit(seed=42),
    )

    assert host.memory is memory
    assert host.store is store
    assert host.embedder is embedder
    assert host.loop.emotional_memory is memory


def test_memory_without_store_and_embedder_raises() -> None:
    store = InMemoryStore()
    embedder = _FixedEmbedder()
    memory = EmotionalMemory(store=store, embedder=embedder)
    with pytest.raises(ValueError, match="store= and embedder="):
        FlyAffectHost(memory=memory, fly_circuit=MockFlyCircuit(seed=42))


def test_host_owns_timestamp_mood_dt() -> None:
    host = FlyAffectHost(fly_circuit=MockFlyCircuit(seed=42))
    affects = host.replay(_journal_frames())

    assert [a.mood_dt for a in affects] == [0.0, 180.0]
    assert host.loop.last_measurement is not None
    assert host.loop.last_measurement.mood_dt == 180.0
    for affect in affects:
        assert -1.0 <= affect.valence <= 1.0
        assert 0.0 <= affect.arousal <= 1.0
        assert -1.0 <= affect.approach <= 1.0
    assert len(host.memory) >= 1


def test_host_owns_wall_clock_mood_dt() -> None:
    host = FlyAffectHost(fly_circuit=MockFlyCircuit(seed=42))
    frame = HostFrame(
        visual_hash="live-1",
        context={"context": "journal", "note_id": "live", "sentiment": 0.2},
    )

    first = host.tick_wall_clock(frame, now=10.0)
    second = host.tick_wall_clock(frame, now=12.5)

    assert first.mood_dt == 0.0
    assert second.mood_dt == 2.5
    assert host.loop.last_measurement is not None
    assert host.loop.last_measurement.mood_dt == 2.5


def test_host_writes_measure_jsonl(tmp_path: Any) -> None:
    path = tmp_path / "measure.jsonl"
    host = FlyAffectHost(fly_circuit=MockFlyCircuit(seed=42), measure_path=path)
    host.replay(_journal_frames())

    records = load_measurement_records(path)
    assert [r.mood_dt for r in records] == [0.0, 180.0]
    assert {r.tau_set for r in records} == {"hypothesis"}
    assert records[0].tau_valence == HYPOTHESIS_TAU_VALENCE
    assert records[0].tau_arousal == HYPOTHESIS_TAU_AROUSAL
    assert records[0].tau_approach == HYPOTHESIS_TAU_APPROACH
    assert records[0].threshold_act == 0.2
    assert records[0].threshold_approach == 0.2
    assert records[0].threshold_calm == 0.0
    assert records[0].approach_denominator == 20.0


def test_host_writes_measure_on_wall_clock(tmp_path: Any) -> None:
    path = tmp_path / "measure.jsonl"
    host = FlyAffectHost(fly_circuit=MockFlyCircuit(seed=42), measure_path=path)
    frame = HostFrame(
        visual_hash="live-1",
        context={"context": "journal", "note_id": "live", "sentiment": 0.2},
    )

    host.tick_wall_clock(frame, now=10.0)
    host.tick_wall_clock(frame, now=12.5)

    records = load_measurement_records(path)
    assert [r.mood_dt for r in records] == [0.0, 2.5]
    assert records[-1].tau_set == "hypothesis"


def test_no_measure_path_does_not_write_jsonl(tmp_path: Any) -> None:
    host = FlyAffectHost(fly_circuit=MockFlyCircuit(seed=42))
    host.replay(_journal_frames())

    assert host.measure_log is None
    assert not (tmp_path / "measure.jsonl").exists()
