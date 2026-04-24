"""Redis-backed persistence for the current affective state snapshot."""

from __future__ import annotations

import importlib
import json
import logging
from typing import Any

from emotional_memory.state import AffectiveState

logger = logging.getLogger(__name__)


class RedisAffectiveStateStore:
    """Persist the current affective state in Redis.

    The class lazily imports ``redis`` so the package remains importable
    without the optional dependency installed. Pass a preconfigured client in
    tests or advanced deployments to avoid URL-based construction.
    """

    __slots__ = ("_client", "_key", "_url")

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        key: str = "emotional_memory:affective_state",
        client: Any | None = None,
    ) -> None:
        self._url = url
        self._key = key
        self._client = client if client is not None else self._create_client(url)

    @staticmethod
    def _create_client(url: str) -> Any:
        try:
            redis_module = importlib.import_module("redis")
        except ImportError as exc:
            raise ImportError(
                "redis is required for RedisAffectiveStateStore.\n"
                "Install with: pip install 'emotional-memory[redis]'"
            ) from exc

        return redis_module.Redis.from_url(url, decode_responses=True)

    def save(self, state: AffectiveState) -> None:
        try:
            self._client.set(self._key, json.dumps(state.snapshot()))
        except Exception as exc:
            logger.warning("RedisAffectiveStateStore.save failed: %s", exc)

    def load(self) -> AffectiveState | None:
        try:
            raw = self._client.get(self._key)
        except Exception as exc:
            logger.warning("RedisAffectiveStateStore.load failed, returning None: %s", exc)
            return None
        if raw is None:
            return None
        return AffectiveState.restore(json.loads(raw))

    def clear(self) -> None:
        try:
            self._client.delete(self._key)
        except Exception as exc:
            logger.warning("RedisAffectiveStateStore.clear failed: %s", exc)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.warning("RedisAffectiveStateStore.close failed: %s", exc)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(url={self._url!r}, key={self._key!r})"
