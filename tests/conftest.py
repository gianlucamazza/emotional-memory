"""Shared test helpers for emotional_memory test suite."""

from datetime import datetime, timedelta, timezone

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.models import Memory, make_emotional_tag
from emotional_memory.stimmung import StimmungField


def make_test_memory(
    content: str = "test",
    valence: float = 0.0,
    arousal: float = 0.5,
    embedding: list[float] | None = None,
    offset_seconds: float = 0.0,
) -> Memory:
    """Create a Memory with a minimal EmotionalTag for use in tests.

    Args:
        content: Memory content string.
        valence: Core affect valence [-1.0, 1.0].
        arousal: Core affect arousal [0.0, 1.0].
        embedding: Optional embedding vector.
        offset_seconds: Seconds to subtract from now for the tag timestamp.
    """
    ts = datetime.now(tz=timezone.utc) - timedelta(seconds=offset_seconds)
    tag = make_emotional_tag(
        core_affect=CoreAffect(valence=valence, arousal=arousal),
        momentum=AffectiveMomentum.zero(),
        stimmung=StimmungField.neutral(),
        consolidation_strength=0.7,
    )
    tag = tag.model_copy(update={"timestamp": ts})
    return Memory.create(content=content, tag=tag, embedding=embedding)


class FixedEmbedder:
    """Always returns the same embedding regardless of input."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, text: str) -> list[float]:
        return list(self._vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class DeterministicEmbedder:
    """Returns distinct embeddings per content string, deterministically (hash-based)."""

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}
        self._dim = 8

    def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            h = hash(text) & 0xFFFFFFFF
            vec = [(((h >> i) & 0xFF) / 255.0) for i in range(self._dim)]
            total = sum(vec) or 1.0
            self._cache[text] = [v / total for v in vec]
        return self._cache[text]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class PolarEmbedder:
    """Two distinct embeddings: 'positive' and 'negative' content families."""

    def embed(self, text: str) -> list[float]:
        if any(w in text.lower() for w in ("joy", "happy", "success", "good")):
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
