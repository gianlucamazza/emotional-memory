"""Design gap regression tests — formerly xfail, now passing.

These tests verify previously-unimplemented behaviours that have since
been implemented. They serve as regression guards.
"""

import pytest

pytestmark = pytest.mark.fidelity


def test_causal_link_type_reachable():
    """_classify_link_type returns 'causal' for temporally ordered, semantically similar
    memories."""
    from emotional_memory.affect import CoreAffect
    from emotional_memory.resonance import _classify_link_type  # type: ignore[attr-defined]

    affect = CoreAffect(valence=0.5, arousal=0.5)
    result = _classify_link_type(
        semantic_sim=0.8,
        emotional_sim=0.3,
        temporal_prox=0.9,
        source_affect=affect,
        target_affect=affect,
        target_precedes_source=True,
    )
    assert result == "causal"


def test_contrastive_link_type_reachable():
    """_classify_link_type returns 'contrastive' for temporally close, opposing-valence
    memories."""
    from emotional_memory.affect import CoreAffect
    from emotional_memory.resonance import _classify_link_type  # type: ignore[attr-defined]

    result = _classify_link_type(
        semantic_sim=0.2,
        emotional_sim=0.2,
        temporal_prox=0.8,
        source_affect=CoreAffect(valence=1.0, arousal=0.5),
        target_affect=CoreAffect(valence=-0.5, arousal=0.5),
    )
    assert result == "contrastive"


def test_decay_config_exposes_power_exponent():
    """DecayConfig has a 'power' field for scaling the power-law exponent."""
    from emotional_memory.decay import DecayConfig

    config = DecayConfig()
    assert config.power == 1.0
