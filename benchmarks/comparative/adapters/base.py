"""Abstract adapter interface for comparative benchmark systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedItem:
    """One retrieved item from any memory system."""

    id: str
    text: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)  # type: ignore[type-arg]


class MemoryAdapter(ABC):
    """Minimal protocol every comparative system must implement."""

    name: str = "unnamed"

    @abstractmethod
    def encode(self, text: str, valence: float = 0.0, arousal: float = 0.5) -> str:
        """Store *text* with optional affective hint. Returns the assigned id."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedItem]:
        """Return the top-k most relevant items for *query*."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored items (called between benchmark runs)."""
