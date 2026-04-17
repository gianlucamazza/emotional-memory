"""Recency baseline — always returns the most recently encoded items.

Serves as the lower-bound baseline: demonstrates what a system achieves
with zero semantic or affective reasoning.  Any useful memory system should
substantially outperform this baseline on mood-congruent recall.
"""

from __future__ import annotations

import uuid

from .base import MemoryAdapter, RetrievedItem


class RecencyAdapter(MemoryAdapter):
    """Returns the N most recently encoded items, regardless of query."""

    name = "recency"

    def __init__(self) -> None:
        self._store: list[tuple[str, str]] = []  # (id, text) in insertion order

    def encode(self, text: str, valence: float = 0.0, arousal: float = 0.5) -> str:
        item_id = str(uuid.uuid4())
        self._store.append((item_id, text))
        return item_id

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        valence: float = 0.0,
        arousal: float = 0.5,
    ) -> list[RetrievedItem]:
        recent = list(reversed(self._store))[:top_k]
        return [RetrievedItem(id=i, text=t, score=0.0) for i, t in recent]

    def reset(self) -> None:
        self._store = []
