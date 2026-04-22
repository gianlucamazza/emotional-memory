"""Semantic-only replay adapter for realistic benchmarks."""

from __future__ import annotations

import uuid
from typing import Any

from .base import (
    ReplayAdapter,
    ReplayRetrievedItem,
    ReplaySessionEnd,
    ReplaySessionStart,
    TokenHashEmbedder,
    cosine_similarity,
)


class NaiveCosineReplayAdapter(ReplayAdapter):
    name = "naive_cosine"

    def __init__(self) -> None:
        self._embedder = TokenHashEmbedder()
        self._store: list[tuple[str, str, list[float]]] = []

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
        self._store.append((item_id, content, self._embedder.embed(content)))
        return item_id

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        valence: float | None = None,
        arousal: float | None = None,
    ) -> list[ReplayRetrievedItem]:
        qvec = self._embedder.embed(query)
        scored = sorted(
            ((item_id, text, cosine_similarity(qvec, emb)) for item_id, text, emb in self._store),
            key=lambda item: item[2],
            reverse=True,
        )
        return [
            ReplayRetrievedItem(id=item_id, text=text, score=score)
            for item_id, text, score in scored[:top_k]
        ]

    def end_session(self) -> ReplaySessionEnd:
        return ReplaySessionEnd(memory_count_end=len(self._store))

    def close(self) -> None:
        return None
