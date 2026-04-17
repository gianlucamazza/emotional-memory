"""Naive cosine-similarity baseline — pure semantic retrieval, no affect.

This adapter serves as the intra-paper baseline: it answers the question
"how much does the AFT multi-signal scorer outperform a plain cosine search?"
Pass the same embedder as AFTAdapter for a fair comparison (same embedding
quality, different scoring).
"""

from __future__ import annotations

import hashlib
import math
import uuid
from typing import Any

from .base import MemoryAdapter, RetrievedItem


class _HashEmbedder:
    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [(b / 127.5) - 1.0 for b in digest[:64]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


class NaiveCosineAdapter(MemoryAdapter):
    """Semantic-only retrieval baseline (no emotional state).

    Pass an embedder instance to use semantic embeddings::

        from emotional_memory.embedders import SentenceTransformerEmbedder
        adapter = NaiveCosineAdapter(embedder=SentenceTransformerEmbedder())
    """

    name = "naive_cosine"

    def __init__(self, embedder: Any = None) -> None:
        self._embedder = embedder if embedder is not None else _HashEmbedder()
        self._store: list[tuple[str, str, list[float]]] = []

    def encode(self, text: str, valence: float = 0.0, arousal: float = 0.5) -> str:
        item_id = str(uuid.uuid4())
        self._store.append((item_id, text, self._embedder.embed(text)))
        return item_id

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        valence: float = 0.0,
        arousal: float = 0.5,
    ) -> list[RetrievedItem]:
        qvec = self._embedder.embed(query)
        scored = sorted(
            ((item_id, text, _cosine(qvec, emb)) for item_id, text, emb in self._store),
            key=lambda x: x[2],
            reverse=True,
        )
        return [RetrievedItem(id=i, text=t, score=s) for i, t, s in scored[:top_k]]

    def reset(self) -> None:
        self._store = []
