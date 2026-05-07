"""Deterministic affect-conditioned query generator for DailyDialog personas.

All queries are generated programmatically from emotion labels.
No LLM is used, guaranteeing determinism and eliminating ground-truth bias.

Four query types are produced per persona (when constraints are satisfiable):
    1. emotion_state_recall       — which conversation felt a specific emotion?
    2. affect_conditioned_content — when feeling X, what were you talking about?
    3. affective_trajectory       — which conversation shifted from one valence to another?
    4. cross_session_control      — which conversation had the most calm/animated atmosphere?
"""

from __future__ import annotations

import random

from benchmarks.dailydialog.dataset import EKMAN_PAD_MAP, Persona, PersonaQuery, PersonaSession

# Query type constants (used as dict keys throughout the benchmark)
TYPE_EMOTION_STATE_RECALL = "emotion_state_recall"
TYPE_AFFECT_CONDITIONED_CONTENT = "affect_conditioned_content"
TYPE_AFFECTIVE_TRAJECTORY = "affective_trajectory"
TYPE_CROSS_SESSION_CONTROL = "cross_session_control"

ALL_QUERY_TYPES = [
    TYPE_EMOTION_STATE_RECALL,
    TYPE_AFFECT_CONDITIONED_CONTENT,
    TYPE_AFFECTIVE_TRAJECTORY,
    TYPE_CROSS_SESSION_CONTROL,
]

# Valence-to-tone words for trajectory queries
_LOW_VALENCE_WORD = "heavy"
_HIGH_VALENCE_WORD = "warm"


def _valence_tone(valence: float) -> str:
    return _HIGH_VALENCE_WORD if valence >= 0.0 else _LOW_VALENCE_WORD


def _arousal_word(arousal: float) -> str:
    return "animated" if arousal >= 0.5 else "calm"


# ---------------------------------------------------------------------------
# Individual query builders (return None when constraints not satisfiable)
# ---------------------------------------------------------------------------


def _try_emotion_state_recall(
    persona: Persona,
    *,
    rng: random.Random,
    query_idx: int,
) -> PersonaQuery | None:
    """Type 1: "Which conversation felt most {emotion}?"

    Target is a session with a non-zero dominant emotion; distractors are
    sessions with a different dominant emotion.  Tests whether the adapter
    can retrieve by affective state rather than pure semantic similarity.
    """
    if len(persona.sessions) < 2:
        return None

    # Prefer sessions with a distinct non-zero emotion
    emotion_sessions = [s for s in persona.sessions if s.dominant_emotion != 0]
    if not emotion_sessions:
        return None

    target = rng.choice(emotion_sessions)
    # Prefer distractors with a different dominant emotion
    diff = [
        s
        for s in persona.sessions
        if s.session_id != target.session_id and s.dominant_emotion != target.dominant_emotion
    ]
    others = [s for s in persona.sessions if s.session_id != target.session_id]
    distractor = rng.choice(diff if diff else others)

    emotion_name = target.dominant_emotion_name
    text = f"Which conversation felt most {emotion_name}?"

    return PersonaQuery(
        query_id=f"{persona.persona_id}_q{query_idx}",
        query_type=TYPE_EMOTION_STATE_RECALL,
        text=text,
        target_session_id=target.session_id,
        distractor_session_ids=(distractor.session_id,),
        top_k=2,
    )


def _try_affect_conditioned_content(
    persona: Persona,
    *,
    rng: random.Random,
    query_idx: int,
) -> PersonaQuery | None:
    """Type 2: "When the {emotion} feeling came up, what was being discussed?"

    Requires ≥2 sessions with different dominant emotions.
    The target is selected by its emotion; distractors have different emotions.
    """
    if len(persona.sessions) < 2:
        return None

    emotion_sessions = [s for s in persona.sessions if s.dominant_emotion != 0]
    if not emotion_sessions:
        return None

    target = rng.choice(emotion_sessions)
    others = [s for s in persona.sessions if s.session_id != target.session_id]
    if not others:
        return None

    diff_emotion = [s for s in others if s.dominant_emotion != target.dominant_emotion]
    distractor = rng.choice(diff_emotion if diff_emotion else others)

    emotion_name = target.dominant_emotion_name
    text = f"When the {emotion_name} feeling came up, what was being discussed?"

    return PersonaQuery(
        query_id=f"{persona.persona_id}_q{query_idx}",
        query_type=TYPE_AFFECT_CONDITIONED_CONTENT,
        text=text,
        target_session_id=target.session_id,
        distractor_session_ids=(distractor.session_id,),
        top_k=2,
    )


def _try_affective_trajectory(
    persona: Persona,
    *,
    rng: random.Random,
    query_idx: int,
) -> PersonaQuery | None:
    """Type 3: "Which conversation shifted from a {from_tone} to a {to_tone} mood?"

    Requires a session where the valence of the first non-neutral turn differs
    from the valence of the last non-neutral turn in direction (one positive,
    one negative).  Tests AFT's momentum signal vs. static cosine similarity.
    """
    trajectory_sessions: list[tuple[PersonaSession, str, str]] = []

    for s in persona.sessions:
        non_neutral = [t for t in s.turns if t.emotion != 0]
        if len(non_neutral) < 2:
            continue
        first_turn = non_neutral[0]
        last_turn = non_neutral[-1]
        first_valence = EKMAN_PAD_MAP[first_turn.emotion][0]
        last_valence = EKMAN_PAD_MAP[last_turn.emotion][0]
        # Require a sign change (positive→negative or negative→positive)
        if (first_valence >= 0.0) != (last_valence >= 0.0):
            from_tone = _valence_tone(first_valence)
            to_tone = _valence_tone(last_valence)
            trajectory_sessions.append((s, from_tone, to_tone))

    if not trajectory_sessions:
        return None

    target, from_tone, to_tone = rng.choice(trajectory_sessions)
    others = [s for s in persona.sessions if s.session_id != target.session_id]
    if not others:
        return None
    distractor = rng.choice(others)

    text = f"Which conversation shifted from a {from_tone} to a {to_tone} mood?"

    return PersonaQuery(
        query_id=f"{persona.persona_id}_q{query_idx}",
        query_type=TYPE_AFFECTIVE_TRAJECTORY,
        text=text,
        target_session_id=target.session_id,
        distractor_session_ids=(distractor.session_id,),
        top_k=2,
    )


def _try_cross_session_control(
    persona: Persona,
    *,
    rng: random.Random,
    query_idx: int,
) -> PersonaQuery | None:
    """Type 4: "Which conversation had the most {calm/animated} atmosphere?"

    Requires ≥2 sessions with different arousal values.
    The target is the session with the lowest (calm) or highest (animated) arousal.
    Tests whether AFT's arousal-based scoring discriminates low vs high arousal memory.
    """
    if len(persona.sessions) < 2:
        return None

    unique_arousals = {s.arousal for s in persona.sessions}
    if len(unique_arousals) < 2:
        return None

    sorted_sessions = sorted(persona.sessions, key=lambda s: s.arousal)
    if rng.random() < 0.5:
        target = sorted_sessions[0]  # lowest arousal → most calm
        atmosphere = "calm"
    else:
        target = sorted_sessions[-1]  # highest arousal → most animated
        atmosphere = "animated"

    others = [s for s in persona.sessions if s.session_id != target.session_id]
    distractor = rng.choice(others)
    text = f"Which conversation had the most {atmosphere} atmosphere?"

    return PersonaQuery(
        query_id=f"{persona.persona_id}_q{query_idx}",
        query_type=TYPE_CROSS_SESSION_CONTROL,
        text=text,
        target_session_id=target.session_id,
        distractor_session_ids=(distractor.session_id,),
        top_k=2,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_queries(persona: Persona, *, rng: random.Random) -> list[PersonaQuery]:
    """Generate up to 4 queries for *persona*, one per type.

    Returns an empty list if fewer than 2 query types are satisfiable
    (persona will be rejected and regenerated by persona_builder).
    """
    builders = [
        _try_emotion_state_recall,
        _try_affect_conditioned_content,
        _try_affective_trajectory,
        _try_cross_session_control,
    ]

    queries: list[PersonaQuery] = []
    for query_idx, builder in enumerate(builders):
        q = builder(persona, rng=rng, query_idx=query_idx)
        if q is not None:
            queries.append(q)

    return queries if len(queries) >= 2 else []
