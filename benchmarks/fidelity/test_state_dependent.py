"""Fidelity benchmark: State-dependent retrieval.

Hypothesis: memories are more accessible when the retrieval state matches
the encoding state (Godden & Baddeley 1975). In AFT terms: the closer the
current Stimmung/CoreAffect to the encoding Stimmung/CoreAffect, the higher
the retrieval score.

Reference: Godden, D.R. & Baddeley, A.D. (1975). Context-dependent memory
in two natural environments. British Journal of Psychology.
"""

from datetime import UTC, datetime

import pytest

from emotional_memory import CoreAffect
from emotional_memory.affect import AffectiveMomentum
from emotional_memory.decay import DecayConfig
from emotional_memory.models import Memory, make_emotional_tag
from emotional_memory.retrieval import RetrievalConfig, retrieval_score
from emotional_memory.stimmung import StimmungField

pytestmark = pytest.mark.fidelity


def _stimmung(valence: float, arousal: float) -> StimmungField:
    return StimmungField(
        valence=valence,
        arousal=arousal,
        dominance=0.5,
        inertia=0.5,
        timestamp=datetime.now(tz=UTC),
    )


def _memory_with_stimmung(valence: float, arousal: float) -> Memory:
    """Create a Memory whose EmotionalTag records a specific Stimmung."""
    stimmung = _stimmung(valence, arousal)
    tag = make_emotional_tag(
        core_affect=CoreAffect(valence=valence, arousal=arousal),
        momentum=AffectiveMomentum.zero(),
        stimmung=stimmung,
        consolidation_strength=0.8,
    )
    tag = tag.model_copy(update={"stimmung_snapshot": stimmung})
    return Memory.create(
        content="state-dependent memory",
        tag=tag,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )


def _score(
    mem: Memory,
    retrieval_valence: float,
    retrieval_arousal: float,
) -> float:
    now = datetime.now(tz=UTC)
    current_stimmung = _stimmung(retrieval_valence, retrieval_arousal)
    config = RetrievalConfig(
        base_weights=[0.10, 0.45, 0.35, 0.05, 0.04, 0.01],  # heavy Stimmung+affect weight
        ape_threshold=10.0,
    )
    return retrieval_score(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        query_affect=CoreAffect(valence=retrieval_valence, arousal=retrieval_arousal),
        current_stimmung=current_stimmung,
        current_momentum=AffectiveMomentum.zero(),
        memory=mem,
        active_memory_ids=[],
        now=now,
        decay_config=DecayConfig(base_decay=0.01),
        retrieval_config=config,
    )


def test_congruent_state_scores_higher_than_opposite():
    """Memory scores higher when retrieved in its encoding state vs opposite."""
    # Memory encoded under positive high-arousal state
    mem = _memory_with_stimmung(valence=0.8, arousal=0.8)

    score_congruent = _score(mem, retrieval_valence=0.8, retrieval_arousal=0.8)
    score_opposite = _score(mem, retrieval_valence=-0.8, retrieval_arousal=0.2)

    assert score_congruent > score_opposite, (
        f"Congruent-state score ({score_congruent:.3f}) should exceed "
        f"opposite-state score ({score_opposite:.3f})"
    )


def test_negative_state_dependency():
    """Negative-state memory scores higher under negative retrieval."""
    mem = _memory_with_stimmung(valence=-0.8, arousal=0.6)

    score_match = _score(mem, retrieval_valence=-0.8, retrieval_arousal=0.6)
    score_mismatch = _score(mem, retrieval_valence=0.8, retrieval_arousal=0.2)

    assert score_match > score_mismatch, (
        f"Match ({score_match:.3f}) should exceed mismatch ({score_mismatch:.3f})"
    )


def test_retrieval_score_monotone_with_stimmung_proximity():
    """Score increases as retrieval Stimmung moves closer to encoding Stimmung."""
    mem = _memory_with_stimmung(valence=0.9, arousal=0.7)

    # Three retrieval states: far, mid, close
    score_far = _score(mem, retrieval_valence=-0.9, retrieval_arousal=0.1)
    score_mid = _score(mem, retrieval_valence=0.3, retrieval_arousal=0.5)
    score_close = _score(mem, retrieval_valence=0.9, retrieval_arousal=0.7)

    assert score_close > score_mid > score_far, (
        f"Expected monotone: close={score_close:.3f} > mid={score_mid:.3f} > far={score_far:.3f}"
    )
