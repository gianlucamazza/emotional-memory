"""AFT adapter — wraps emotional-memory EmotionalMemory."""

from __future__ import annotations

import hashlib

from emotional_memory import CoreAffect, EmotionalMemory, InMemoryStore

from .base import MemoryAdapter, RetrievedItem


class _HashEmbedder:
    """Deterministic 64-dim embedder for benchmark reproducibility."""

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [(b / 127.5) - 1.0 for b in digest[:64]]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class AFTAdapter(MemoryAdapter):
    """emotional-memory (AFT) adapter."""

    name = "aft"

    def __init__(self) -> None:
        self._em = EmotionalMemory(store=InMemoryStore(), embedder=_HashEmbedder())

    def encode(self, text: str, valence: float = 0.0, arousal: float = 0.5) -> str:
        self._em.set_affect(CoreAffect(valence=valence, arousal=arousal))
        mem = self._em.encode(text)
        return mem.id

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedItem]:
        results = self._em.retrieve(query, top_k=top_k)
        return [
            RetrievedItem(id=m.id, text=m.content, score=m.tag.core_affect.valence)
            for m in results
        ]

    def reset(self) -> None:
        self._em = EmotionalMemory(store=InMemoryStore(), embedder=_HashEmbedder())
