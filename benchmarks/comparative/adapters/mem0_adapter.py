"""Mem0 adapter — wraps mem0ai if installed, otherwise marks as not_evaluated."""

from __future__ import annotations

import uuid

from .base import MemoryAdapter, RetrievedItem

_NOT_INSTALLED_REASON = "mem0ai not installed (pip install mem0ai)"


class Mem0Adapter(MemoryAdapter):
    """mem0 adapter (requires: pip install mem0ai)."""

    name = "mem0"

    def __init__(self) -> None:
        try:
            import mem0  # noqa: F401

            self._available = True
            # mem0 requires an OpenAI key by default; we use an in-memory config
            from mem0 import Memory  # type: ignore[import-untyped]

            self._mem = Memory.from_config(
                {
                    "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
                    "embedder": {"provider": "openai"},
                    "vector_store": {"provider": "chroma", "config": {"collection_name": "bench"}},
                }
            )
            self._user_id = "benchmark"
        except ImportError:
            self._available = False
            self._mem = None

    def encode(self, text: str, valence: float = 0.0, arousal: float = 0.5) -> str:
        if not self._available or self._mem is None:
            return str(uuid.uuid4())
        result = self._mem.add(text, user_id=self._user_id)
        return (
            result.get("id", str(uuid.uuid4())) if isinstance(result, dict) else str(uuid.uuid4())
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        valence: float = 0.0,
        arousal: float = 0.5,
    ) -> list[RetrievedItem]:
        if not self._available or self._mem is None:
            return []
        results = self._mem.search(query, user_id=self._user_id, limit=top_k)
        items = results if isinstance(results, list) else results.get("results", [])
        return [
            RetrievedItem(
                id=str(r.get("id", i)),
                text=r.get("memory", ""),
                score=r.get("score", 0.0),
            )
            for i, r in enumerate(items)
        ]

    def reset(self) -> None:
        import contextlib

        if self._available and self._mem is not None:
            with contextlib.suppress(Exception):
                self._mem.delete_all(user_id=self._user_id)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def not_available_reason(self) -> str:
        return "" if self._available else _NOT_INSTALLED_REASON
