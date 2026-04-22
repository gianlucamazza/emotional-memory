from __future__ import annotations

import pytest

from emotional_memory.affect import CoreAffect
from emotional_memory.engine import EmotionalMemory
from emotional_memory.state import AffectiveState
from emotional_memory.state_stores.in_memory import InMemoryAffectiveStateStore
from emotional_memory.state_stores.redis import RedisAffectiveStateStore
from emotional_memory.state_stores.sqlite import SQLiteAffectiveStateStore
from emotional_memory.stores.in_memory import InMemoryStore


def _sample_state() -> AffectiveState:
    state = AffectiveState.initial()
    state = state.update(CoreAffect(valence=0.2, arousal=0.4))
    return state.update(CoreAffect(valence=0.6, arousal=0.7))


class _FixedEmbedder:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def test_in_memory_state_store_round_trip_returns_copy() -> None:
    store = InMemoryAffectiveStateStore()
    state = _sample_state()

    store.save(state)
    loaded = store.load()

    assert loaded is not None
    assert loaded.core_affect == state.core_affect
    assert loaded is not state


def test_in_memory_state_store_clear() -> None:
    store = InMemoryAffectiveStateStore()
    store.save(_sample_state())

    store.clear()

    assert store.load() is None


def test_sqlite_state_store_round_trip_preserves_history(tmp_path) -> None:
    path = tmp_path / "affective-state.sqlite"
    store = SQLiteAffectiveStateStore(path)
    state = _sample_state()

    store.save(state)
    loaded = store.load()

    assert loaded is not None
    assert loaded.core_affect == state.core_affect

    next_original = state.update(CoreAffect(valence=0.8, arousal=0.9))
    next_loaded = loaded.update(CoreAffect(valence=0.8, arousal=0.9))
    assert next_loaded.momentum == next_original.momentum


def test_sqlite_state_store_clear(tmp_path) -> None:
    path = tmp_path / "affective-state.sqlite"
    store = SQLiteAffectiveStateStore(path)
    store.save(_sample_state())

    store.clear()

    assert store.load() is None


class _FakeRedisClient:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self.closed = False

    def set(self, key: str, value: str) -> None:
        self._store[key] = value

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def close(self) -> None:
        self.closed = True


def test_redis_state_store_round_trip() -> None:
    client = _FakeRedisClient()
    store = RedisAffectiveStateStore(client=client, key="test-state")
    state = _sample_state()

    store.save(state)
    loaded = store.load()

    assert loaded is not None
    assert loaded.core_affect == state.core_affect


def test_redis_state_store_clear_and_close() -> None:
    client = _FakeRedisClient()
    store = RedisAffectiveStateStore(client=client, key="test-state")
    store.save(_sample_state())

    store.clear()
    store.close()

    assert store.load() is None
    assert client.closed


def test_redis_state_store_supports_shared_engine_state_continuity() -> None:
    client = _FakeRedisClient()
    state_store = RedisAffectiveStateStore(client=client, key="shared-state")
    engine_a = EmotionalMemory(
        store=InMemoryStore(),
        embedder=_FixedEmbedder(),
        state_store=state_store,
    )
    engine_a.set_affect(CoreAffect(valence=0.7, arousal=0.3))
    engine_a.observe("A difficult review left the room feeling tense.")

    engine_b = EmotionalMemory(
        store=InMemoryStore(),
        embedder=_FixedEmbedder(),
        state_store=RedisAffectiveStateStore(client=client, key="shared-state"),
    )

    restored = engine_b.restore_persisted_state()
    next_memory = engine_b.encode("We reopened the conversation after a calmer start.")

    assert restored is True
    assert engine_b.get_state().core_affect.valence == pytest.approx(
        next_memory.tag.core_affect.valence
    )
    assert engine_b.get_state().core_affect.arousal == pytest.approx(
        next_memory.tag.core_affect.arousal
    )


def test_redis_state_store_clear_removes_shared_snapshot_for_new_engine() -> None:
    client = _FakeRedisClient()
    state_store = RedisAffectiveStateStore(client=client, key="shared-state")
    engine = EmotionalMemory(
        store=InMemoryStore(),
        embedder=_FixedEmbedder(),
        state_store=state_store,
    )
    engine.set_affect(CoreAffect(valence=0.5, arousal=0.5))
    engine.clear_persisted_state()

    restored = EmotionalMemory(
        store=InMemoryStore(),
        embedder=_FixedEmbedder(),
        state_store=RedisAffectiveStateStore(client=client, key="shared-state"),
    )

    assert restored.get_state().core_affect == CoreAffect.neutral()
