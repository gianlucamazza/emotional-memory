"""Hypothesis strategies for algebraic invariant tests.

Values stay inside the documented domains of the production types. Psychological
claims (mood-congruent recall "works", high arousal always decays slower) are
not generated here — those live in ``benchmarks/fidelity/``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from hypothesis import strategies as st

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.decay import DecayConfig
from emotional_memory.models import EmotionalTag, make_emotional_tag
from emotional_memory.mood import MoodDecayConfig, MoodField

NOW = datetime(2026, 1, 1, tzinfo=UTC)

_unit = st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)
_signed = st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False)


@st.composite
def core_affects(draw: st.DrawFn) -> CoreAffect:
    return CoreAffect(
        valence=draw(_signed),
        arousal=draw(_unit),
        dominance=draw(_unit),
    )


@st.composite
def mood_fields(draw: st.DrawFn) -> MoodField:
    return MoodField(
        valence=draw(_signed),
        arousal=draw(_unit),
        dominance=draw(_unit),
        inertia=draw(_unit),
        timestamp=NOW,
    )


@st.composite
def decay_configs(draw: st.DrawFn) -> DecayConfig:
    return DecayConfig(
        base_decay=draw(st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False)),
        arousal_modulation=draw(_unit),
        retrieval_boost=draw(st.floats(0.0, 2.0, allow_nan=False, allow_infinity=False)),
        floor_arousal_threshold=draw(_unit),
        floor_value=draw(_unit),
        min_seconds=draw(st.floats(1e-3, 10.0, allow_nan=False, allow_infinity=False)),
        power=draw(st.floats(0.0, 4.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def mood_decay_configs(draw: st.DrawFn) -> MoodDecayConfig:
    return MoodDecayConfig(
        base_half_life_seconds=draw(st.floats(1.0, 1e6, allow_nan=False, allow_infinity=False)),
        inertia_scale=draw(st.floats(0.0, 10.0, allow_nan=False, allow_infinity=False)),
        baseline_valence=draw(_signed),
        baseline_arousal=draw(_unit),
        baseline_dominance=draw(_unit),
    )


@st.composite
def emotional_tags(draw: st.DrawFn) -> EmotionalTag:
    tag = make_emotional_tag(
        core_affect=draw(core_affects()),
        momentum=AffectiveMomentum.zero(),
        mood=draw(mood_fields()),
        consolidation_strength=draw(_unit),
    )
    return tag.model_copy(
        update={
            "timestamp": NOW,
            "retrieval_count": draw(st.integers(0, 100)),
        }
    )


finite_components = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)
vectors = st.lists(finite_components, min_size=1, max_size=16)
weight_bases = st.lists(
    st.floats(-1.0, 2.0, allow_nan=False, allow_infinity=False),
    min_size=6,
    max_size=6,
)
elapsed_seconds = st.floats(0.0, 1e8, allow_nan=False, allow_infinity=False)
