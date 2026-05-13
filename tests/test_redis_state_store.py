"""Tests for RedisAffectiveStateStore.

Two levels:
- Mock tests: use a dict-backed _MockRedisClient (zero deps, fast).
- Fakeredis tests: use fakeredis.FakeRedis to validate against a real Redis protocol
  implementation (no live server needed).
"""

from __future__ import annotations

import json

import fakeredis

from emotional_memory.affect import CoreAffect
from emotional_memory.state import AffectiveState
from emotional_memory.state_stores.redis import RedisAffectiveStateStore

# ---------------------------------------------------------------------------
# Minimal Redis client mock
# ---------------------------------------------------------------------------


class _MockRedisClient:
    """Dict-backed mock that implements the subset of Redis used by the store."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self.closed = False

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def close(self) -> None:
        self.closed = True


def _make_store(key: str = "test:state") -> tuple[RedisAffectiveStateStore, _MockRedisClient]:
    client = _MockRedisClient()
    store = RedisAffectiveStateStore(client=client, key=key)
    return store, client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_load_roundtrip() -> None:
    store, _ = _make_store()
    state = AffectiveState.initial()
    state = state.update(CoreAffect(valence=0.5, arousal=0.7, dominance=0.6))

    store.save(state)
    loaded = store.load()

    assert loaded is not None
    assert abs(loaded.core_affect.valence - 0.5) < 1e-6
    assert abs(loaded.core_affect.arousal - 0.7) < 1e-6


def test_load_returns_none_when_empty() -> None:
    store, _ = _make_store()
    result = store.load()
    assert result is None


def test_clear_removes_key() -> None:
    store, client = _make_store(key="em:state")
    state = AffectiveState.initial()
    store.save(state)

    assert client.get("em:state") is not None
    store.clear()
    assert client.get("em:state") is None
    assert store.load() is None


def test_save_overwrites_previous() -> None:
    store, _ = _make_store()
    state1 = AffectiveState.initial()
    state1 = state1.update(CoreAffect(valence=0.3, arousal=0.4, dominance=0.5))
    store.save(state1)

    state2 = AffectiveState.initial()
    state2 = state2.update(CoreAffect(valence=-0.5, arousal=0.8, dominance=0.3))
    store.save(state2)

    loaded = store.load()
    assert loaded is not None
    assert abs(loaded.core_affect.valence - (-0.5)) < 1e-6


def test_close_calls_client_close() -> None:
    store, client = _make_store()
    assert not client.closed
    store.close()
    assert client.closed


def test_close_without_close_method_does_not_raise() -> None:
    class _NoClose:
        def set(self, k: str, v: str) -> None: ...
        def get(self, k: str) -> str | None:
            return None

        def delete(self, k: str) -> None: ...

    store = RedisAffectiveStateStore(client=_NoClose(), key="test")
    # Must not raise
    store.close()


def test_snapshot_roundtrip_preserves_history() -> None:
    """save/load preserves AffectiveState including momentum history."""
    store, client = _make_store()
    state = AffectiveState.initial()
    state = state.update(CoreAffect(valence=0.1, arousal=0.2, dominance=0.5))
    state = state.update(CoreAffect(valence=0.4, arousal=0.6, dominance=0.4))

    store.save(state)
    raw = client.get("test:state")
    assert raw is not None
    snapshot = json.loads(raw)
    assert "core_affect" in snapshot

    loaded = store.load()
    assert loaded is not None
    assert abs(loaded.core_affect.valence - state.core_affect.valence) < 1e-6


# ---------------------------------------------------------------------------
# Fakeredis tests — validates against real Redis protocol (no server needed)
# ---------------------------------------------------------------------------


def _make_fake_store(key: str = "test:state") -> RedisAffectiveStateStore:
    client = fakeredis.FakeRedis(decode_responses=True)
    return RedisAffectiveStateStore(client=client, key=key)


def test_fakeredis_save_load_roundtrip() -> None:
    store = _make_fake_store()
    state = AffectiveState.initial()
    state = state.update(CoreAffect(valence=0.6, arousal=0.8, dominance=0.4))

    store.save(state)
    loaded = store.load()

    assert loaded is not None
    assert abs(loaded.core_affect.valence - 0.6) < 1e-6
    assert abs(loaded.core_affect.arousal - 0.8) < 1e-6


def test_fakeredis_clear() -> None:
    store = _make_fake_store(key="fake:em:state")
    state = AffectiveState.initial()
    store.save(state)
    assert store.load() is not None

    store.clear()
    assert store.load() is None


def test_fakeredis_close_does_not_raise() -> None:
    store = _make_fake_store()
    store.close()  # FakeRedis has a close() method; should not raise
