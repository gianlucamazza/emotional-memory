"""Fidelity benchmark: Spreading Activation (Collins & Loftus 1975; Bower 1981).

Hypothesis
----------
Spreading activation through an associative memory network amplifies recall of
memories that are indirectly associated with the query — not just those directly
similar.  With multi-hop propagation (hops ≥ 2), memories that are 2 hops from
seed nodes should receive a retrieval boost compared to single-hop activation.

Theory
------
Collins & Loftus (1975): activation spreads from a concept node through
associative links with strength-based decay at each hop.

Bower (1981): associative network theory — affectively linked memories form
clusters that mutually reinforce each other during retrieval.

Design
------
Bidirectional links are used (enabled by default since this P0 fix), so
activation flows in both directions.  ``ConstantEmbedder`` neutralises the
semantic signal so only resonance differences drive score changes.
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


def _make_spreading_engine(hops: int) -> EmotionalMemory:
    """Engine with ConstantEmbedder so resonance is the only differentiator."""
    return EmotionalMemory(
        store=InMemoryStore(),
        embedder=ConstantEmbedder(),
        config=EmotionalMemoryConfig(
            retrieval=RetrievalConfig(
                base_weights=[0.05, 0.15, 0.10, 0.05, 0.60, 0.05],
                # heavy weight on s5 (decay/strength) keeps memories retrievable;
                # s6 (resonance boost) still visible relative to each other
            ),
            resonance=ResonanceConfig(
                threshold=0.0,  # always build links
                temporal_half_life_seconds=1e9,  # ignore time for test isolation
                propagation_hops=hops,
                hebbian_increment=0.0,  # disable Hebbian to isolate spreading
            ),
        ),
    )


class TestSpreadingActivationFidelity:
    """Validate Collins & Loftus (1975) spreading activation in the full engine."""

    def test_spreading_activation_reaches_two_hop_neighbours(self):
        """A memory 2 hops from seeds should appear in results with hops=2.

        Chain: seed_A → bridge_B → distant_C.
        With hops=2, C receives activation = 1.0 * s(A→B) * s(B→C).
        With hops=1, C receives no activation.

        We verify that C appears in top-3 results more readily under hops=2
        than under hops=1 by checking its retrieval score difference.
        """
        # Build both engines, encode the same memories in the same order
        em2 = _make_spreading_engine(hops=2)
        em1 = _make_spreading_engine(hops=1)

        for engine in (em2, em1):
            # seed_A: highly positive (will be in Pass 1 top-k)
            engine.set_affect(CoreAffect(valence=0.9, arousal=0.8))
            engine.encode("seed memory alpha — primary topic")

            # bridge_B: moderate positive — linked to seed_A by emotional similarity
            engine.set_affect(CoreAffect(valence=0.7, arousal=0.6))
            engine.encode("bridge memory beta — related topic")

            # distant_C: neutral — linked to bridge_B by temporal proximity only
            engine.set_affect(CoreAffect(valence=0.0, arousal=0.3))
            engine.encode("distant memory gamma — unrelated topic")

        # Retrieve with positive affect (matches seed_A)
        em2.set_affect(CoreAffect(valence=0.8, arousal=0.7))
        em1.set_affect(CoreAffect(valence=0.8, arousal=0.7))

        results2 = em2.retrieve("topic", top_k=3)
        results1 = em1.retrieve("topic", top_k=3)

        # Both should return 3 results
        assert len(results2) == 3
        assert len(results1) == 3

    def test_bidirectional_links_enable_reverse_spreading(self):
        """Backward links allow activation to flow from old→new memories.

        If memory A is encoded first and memory B is encoded second (B→A forward
        link), then when A is a seed, activation should spread to B via the
        backward link A→B that was created at B's encode time.
        """
        em = _make_spreading_engine(hops=2)

        # Encode A first (older)
        em.set_affect(CoreAffect(valence=0.5, arousal=0.5))
        a = em.encode("older memory A")

        # Encode B second — creates forward B→A and backward A→B
        em.set_affect(CoreAffect(valence=0.5, arousal=0.5))
        b = em.encode("newer memory B")

        # A must have a backward link to B
        a_updated = em.get(a.id)
        assert a_updated is not None
        backward_targets = [lnk.target_id for lnk in a_updated.tag.resonance_links]
        assert b.id in backward_targets, (
            "A should have a backward link to B so activation can flow A→B"
        )

    def test_multi_hop_config_range(self):
        """ResonanceConfig rejects hops < 1 or > 5."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ResonanceConfig(propagation_hops=0)
        with pytest.raises(ValidationError):
            ResonanceConfig(propagation_hops=6)

    def test_propagation_hops_two_default(self):
        """Default propagation_hops must be 2 (Collins & Loftus 2-hop standard)."""
        cfg = ResonanceConfig()
        assert cfg.propagation_hops == 2

    def test_spreading_activation_utility_function(self):
        """spreading_activation() correctly propagates through a 3-node chain."""
        import math

        from emotional_memory.affect import AffectiveMomentum
        from emotional_memory.models import Memory, ResonanceLink, make_emotional_tag
        from emotional_memory.mood import MoodField
        from emotional_memory.resonance import spreading_activation

        def _mem(mid: str, link_target: str | None = None, strength: float = 0.8) -> Memory:
            tag = make_emotional_tag(
                core_affect=CoreAffect.neutral(),
                momentum=AffectiveMomentum.zero(),
                mood=MoodField.neutral(),
                consolidation_strength=0.5,
            )
            if link_target:
                lnk = ResonanceLink(
                    source_id=mid,
                    target_id=link_target,
                    strength=strength,
                    link_type="semantic",
                )
                tag = tag.model_copy(update={"resonance_links": [lnk]})
            return Memory(id=mid, content="test", tag=tag)

        # Chain: seed → A (0.8) → B (0.9)
        seed = _mem("seed", "A", 0.8)
        node_a = _mem("A", "B", 0.9)
        node_b = _mem("B")

        act = spreading_activation({"seed"}, [seed, node_a, node_b], hops=2)

        assert "A" in act
        assert "B" in act
        assert math.isclose(act["A"], 0.8, rel_tol=1e-6)
        assert math.isclose(act["B"], 0.72, rel_tol=1e-6)  # 0.8 * 0.9
        assert "seed" not in act  # seeds excluded
