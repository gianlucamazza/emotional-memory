"""AFT adapter — wraps emotional-memory EmotionalMemory."""

from __future__ import annotations

import hashlib
from typing import Any

from emotional_memory import CoreAffect, EmotionalMemory, InMemoryStore

from .base import MemoryAdapter, RetrievedItem


class _HashEmbedder:
    """Deterministic 64-dim embedder for benchmark reproducibility."""

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [(b / 127.5) - 1.0 for b in digest[:64]]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _make_embedder(embedder: Any) -> Any:
    if embedder is not None:
        return embedder
    return _HashEmbedder()


class AFTAdapter(MemoryAdapter):
    """emotional-memory (AFT) adapter.

    Pass an embedder instance to use semantic embeddings instead of the
    default SHA-256 hash embedder::

        from emotional_memory.embedders import SentenceTransformerEmbedder
        adapter = AFTAdapter(embedder=SentenceTransformerEmbedder())
    """

    name = "aft"

    def __init__(self, embedder: Any = None) -> None:
        self._embedder = _make_embedder(embedder)
        self._em = EmotionalMemory(store=InMemoryStore(), embedder=self._embedder)

    def encode(self, text: str, valence: float = 0.0, arousal: float = 0.5) -> str:
        self._em.set_affect(CoreAffect(valence=valence, arousal=arousal))
        mem = self._em.encode(text)
        return mem.id

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        valence: float = 0.0,
        arousal: float = 0.5,
    ) -> list[RetrievedItem]:
        # Set affect to query quadrant so mood-congruent scoring is active
        self._em.set_affect(CoreAffect(valence=valence, arousal=arousal))
        results = self._em.retrieve(query, top_k=top_k)
        return [RetrievedItem(id=m.id, text=m.content, score=0.0) for m in results]

    def reset(self) -> None:
        self._em = EmotionalMemory(store=InMemoryStore(), embedder=self._embedder)
