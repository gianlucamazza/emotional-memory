"""Fidelity benchmark: Hebbian co-retrieval strengthening (Hebb 1949).

Hypothesis
----------
Memories that are repeatedly retrieved together should have their associative
link strengths increased after each co-retrieval, following the Hebbian
learning rule: "neurons that fire together, wire together."

After N rounds of co-retrieval, the link strength between co-retrieved
memories must be strictly greater than the initial strength, and must
increase monotonically across rounds.

Theory
------
Hebb, D. O. (1949). The Organisation of Behavior.
  "When an axon of cell A is near enough to excite a cell B and repeatedly
   or persistently takes part in firing it, some growth process or metabolic
   change takes place in one or both cells such that A's efficiency, as one
   of the cells firing B, is increased."

Design
------
Two memories with identical embeddings (ConstantEmbedder) are encoded, then
retrieved together N times.  After each retrieval the link strength between
them is recorded.  We verify that the strength grows monotonically.
Spreading activation is enabled to ensure both are always co-retrieved.
"""

from __future__ import annotations

import pytest

from benchmarks.conftest import ConstantEmbedder
from emotional_memory import (
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    ResonanceConfig,
    RetrievalConfig,
)

pytestmark = pytest.mark.fidelity


def _hebbian_engine(increment: float = 0.15) -> EmotionalMemory:
    return EmotionalMemory(
        store=InMemoryStore(),
        embedder=ConstantEmbedder(),
        config=EmotionalMemoryConfig(
            retrieval=RetrievalConfig(
                ape_threshold=10.0,  # disable reconsolidation
            ),
            resonance=ResonanceConfig(
                threshold=0.0,
                temporal_half_life_seconds=1e9,
                propagation_hops=2,
                hebbian_increment=increment,
            ),
        ),
    )


def _link_strength(em: EmotionalMemory, source_id: str, target_id: str) -> float | None:
    mem = em.get(source_id)
    if mem is None:
        return None
    for lnk in mem.tag.resonance_links:
        if lnk.target_id == target_id:
            return lnk.strength
    return None


class TestHebbianStrengtheningFidelity:
    def test_co_retrieval_increases_link_strength(self):
        """After co-retrieval, the link between two memories must be stronger."""
        em = _hebbian_engine()
        em.set_affect(CoreAffect(valence=0.5, arousal=0.5))
        a = em.encode("memory alpha — repeatedly co-retrieved")
        b = em.encode("memory beta — repeatedly co-retrieved")

        # Capture strength before any retrieval
        strength_before = _link_strength(em, b.id, a.id) or 0.0

        # Co-retrieve once
        em.set_affect(CoreAffect(valence=0.5, arousal=0.5))
        em.retrieve("co-retrieved", top_k=2)

        strength_after = _link_strength(em, b.id, a.id) or 0.0
        assert strength_after > strength_before, (
            f"Link strength must increase after co-retrieval: "
            f"before={strength_before:.4f}, after={strength_after:.4f}"
        )

    def test_repeated_co_retrieval_monotonically_strengthens(self):
        """Link strength must grow (weakly) monotonically across N co-retrieval rounds."""
        em = _hebbian_engine(increment=0.1)
        em.set_affect(CoreAffect(valence=0.6, arousal=0.6))
        a = em.encode("memory A — paired")
        b = em.encode("memory B — paired")

        strengths: list[float] = []
        for _ in range(5):
            em.set_affect(CoreAffect(valence=0.6, arousal=0.6))
            em.retrieve("paired", top_k=2)
            s = _link_strength(em, b.id, a.id)
            if s is not None:
                strengths.append(s)

        assert len(strengths) >= 2, "Need at least 2 strength readings"
        for i in range(1, len(strengths)):
            assert strengths[i] >= strengths[i - 1] - 1e-9, (
                f"Strength decreased at step {i}: {strengths[i - 1]:.4f} → {strengths[i]:.4f}"
            )

    def test_zero_increment_leaves_strength_unchanged(self):
        """With hebbian_increment=0.0, link strengths must not change after co-retrieval."""
        em = _hebbian_engine(increment=0.0)
        em.set_affect(CoreAffect(valence=0.5, arousal=0.5))
        a = em.encode("alpha")
        b = em.encode("beta")

        strength_before = _link_strength(em, b.id, a.id)

        em.set_affect(CoreAffect(valence=0.5, arousal=0.5))
        em.retrieve("memory", top_k=2)

        strength_after = _link_strength(em, b.id, a.id)
        assert strength_before == pytest.approx(strength_after, abs=1e-9), (
            f"Strength must not change when increment=0: "
            f"before={strength_before}, after={strength_after}"
        )

    def test_strength_capped_at_one(self):
        """Link strength must not exceed 1.0 regardless of how many co-retrievals occur."""
        em = _hebbian_engine(increment=0.5)  # large increment to hit the cap quickly
        em.set_affect(CoreAffect(valence=0.5, arousal=0.5))
        a = em.encode("first")
        b = em.encode("second")

        for _ in range(20):
            em.set_affect(CoreAffect(valence=0.5, arousal=0.5))
            em.retrieve("memory", top_k=2)

        s = _link_strength(em, b.id, a.id)
        assert s is not None
        assert s <= 1.0 + 1e-9, f"Strength must be capped at 1.0, got {s:.6f}"
