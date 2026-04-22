"""Recency-only replay adapter for realistic benchmarks."""

from __future__ import annotations

import uuid
from typing import Any

from .base import ReplayAdapter, ReplayRetrievedItem, ReplaySessionEnd, ReplaySessionStart


class RecencyReplayAdapter(ReplayAdapter):
    name = "recency"

    def __init__(self) -> None:
        self._store: list[tuple[str, str]] = []

    def reset(self) -> None:
        self._store = []

    def begin_session(self, session_id: str) -> ReplaySessionStart:
        return ReplaySessionStart(
            state_loaded=False,
            memory_count_start=len(self._store),
        )

    def encode(
        self,
        *,
        memory_alias: str,
        content: str,
        valence: float,
        arousal: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        item_id = str(uuid.uuid4())
        self._store.append((item_id, content))
        return item_id

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        valence: float | None = None,
        arousal: float | None = None,
    ) -> list[ReplayRetrievedItem]:
        recent = list(reversed(self._store))[:top_k]
        return [ReplayRetrievedItem(id=item_id, text=text, score=0.0) for item_id, text in recent]

    def end_session(self) -> ReplaySessionEnd:
        return ReplaySessionEnd(memory_count_end=len(self._store))

    def close(self) -> None:
        return None
