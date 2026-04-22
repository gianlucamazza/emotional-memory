"""Adapter contract for replayable realistic benchmarks."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class TokenHashEmbedder:
    """Deterministic token-hash embedder with light lexical semantics."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in self._tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self._dim
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return ["".join(ch for ch in token.lower() if ch.isalnum()) for token in text.split()]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


@dataclass
class ReplayRetrievedItem:
    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplaySessionStart:
    state_loaded: bool
    memory_count_start: int
    mood_start: dict[str, float] | None = None


@dataclass
class ReplaySessionEnd:
    memory_count_end: int
    mood_end: dict[str, float] | None = None


class ReplayAdapter(ABC):
    """Session-aware adapter for realistic replay benchmarks."""

    name: str = "unnamed"
    supports_explanations: bool = False
    supports_persisted_state: bool = False

    @abstractmethod
    def reset(self) -> None:
        """Clear adapter state at the beginning of a benchmark run."""

    @abstractmethod
    def begin_session(self, session_id: str) -> ReplaySessionStart:
        """Prepare the adapter for a new session boundary."""

    @abstractmethod
    def encode(
        self,
        *,
        memory_alias: str,
        content: str,
        valence: float,
        arousal: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a memory event for the current session."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        valence: float | None = None,
        arousal: float | None = None,
    ) -> list[ReplayRetrievedItem]:
        """Return top-k replay items for the current session."""

    @abstractmethod
    def end_session(self) -> ReplaySessionEnd:
        """Persist session state and release resources if needed."""

    @abstractmethod
    def close(self) -> None:
        """Optional final cleanup hook."""
