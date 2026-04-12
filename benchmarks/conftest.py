"""Shared fixtures and helpers for emotional_memory benchmarks."""

from __future__ import annotations

from typing import ClassVar

import pytest

from emotional_memory import (
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    ResonanceConfig,
    RetrievalConfig,
)


class ScalableEmbedder:
    """Hash-based embedder with configurable dimensionality."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        h = hash(text) & 0xFFFFFFFF
        vec = [((h >> (i % 32)) & 0xFF) / 255.0 for i in range(self._dim)]
        total = sum(vec) or 1.0
        return [v / total for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class ConstantEmbedder:
    """Returns identical embedding for all inputs.

    Neutralises the semantic signal so that emotional/mood signals
    dominate retrieval — useful for psychological fidelity tests.
    """

    _VEC: ClassVar[list[float]] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def embed(self, text: str) -> list[float]:
        return list(self._VEC)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def make_engine(
    resonance_threshold: float = 0.3,
    dim: int = 64,
    **config_overrides: object,
) -> EmotionalMemory:
    """Performance-oriented engine with ScalableEmbedder."""
    config = EmotionalMemoryConfig(
        resonance=ResonanceConfig(threshold=resonance_threshold),
        **config_overrides,  # type: ignore[arg-type]
    )
    return EmotionalMemory(
        store=InMemoryStore(),
        embedder=ScalableEmbedder(dim=dim),
        config=config,
    )


def make_fidelity_engine(
    mood_alpha: float = 0.3,
    ape_threshold: float = 0.01,
    emotional_bias: bool = True,
) -> EmotionalMemory:
    """Psychological fidelity engine with ConstantEmbedder.

    Uses strong emotional weights and low APE threshold so that
    reconsolidation and mood-congruence effects are clearly observable.
    """
    base_weights = (
        [0.10, 0.40, 0.30, 0.10, 0.05, 0.05]  # strong emotional bias
        if emotional_bias
        else [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
    )
    return EmotionalMemory(
        store=InMemoryStore(),
        embedder=ConstantEmbedder(),
        config=EmotionalMemoryConfig(
            retrieval=RetrievalConfig(
                base_weights=base_weights,
                ape_threshold=ape_threshold,
                reconsolidation_learning_rate=0.4,
            ),
            resonance=ResonanceConfig(threshold=0.9),  # disable resonance links
            mood_alpha=mood_alpha,
        ),
    )


def populate_store(engine: EmotionalMemory, n: int) -> None:
    """Encode n memories alternating positive/negative affect.

    Resonance links are disabled during population (threshold=2.0) to keep
    setup O(n) instead of O(n²). The engine's resonance config is restored
    afterward so benchmarked operations use the original threshold.
    """
    original_config = engine._config
    fast_config = original_config.model_copy(update={"resonance": ResonanceConfig(threshold=2.0)})
    engine._config = fast_config
    for i in range(n):
        valence = 0.8 if i % 2 == 0 else -0.7
        arousal = 0.3 + (i % 5) * 0.1
        engine.set_affect(CoreAffect(valence=valence, arousal=arousal))
        engine.encode(f"Memory content number {i} with various contextual details about the event")
    engine._config = original_config


@pytest.fixture
def fidelity_engine() -> EmotionalMemory:
    return make_fidelity_engine()


@pytest.fixture
def perf_engine_small() -> EmotionalMemory:
    engine = make_engine()
    populate_store(engine, 50)
    return engine


@pytest.fixture
def perf_engine_large() -> EmotionalMemory:
    engine = make_engine()
    populate_store(engine, 1_000)
    return engine
