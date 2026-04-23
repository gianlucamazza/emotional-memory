"""Retrieval Scoring Engine.

Multi-signal retrieval scoring combining six components:
  s1 — semantic similarity         (cosine, query embedding vs memory)
  s2 — mood congruence             (mood-congruent recall, Bower 1981)
  s3 — core affect proximity       (current affect vs affect at encoding)
  s4 — momentum alignment          (trajectory of affective change)
  s5 — recency / decay score       (ACT-R power-law)
  s6 — resonance boost             (spreading activation, Bower 1981)

Weights are adaptive: modulated by current mood state.
  Negative mood  → weight shifts toward emotional signals
  High arousal   → weight shifts toward momentum
  Neutral/calm   → weight shifts toward semantic

Reconsolidation (Nader & Schiller 2000): retrieval computes Affective
Prediction Error (APE) and updates the tag's core_affect if APE exceeds
a threshold.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator

from emotional_memory._math import cosine_similarity
from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.decay import DecayConfig, compute_effective_strength
from emotional_memory.models import EmotionalTag, Memory
from emotional_memory.mood import MoodField

# Max Euclidean distance in the 3-D PAD space used by MoodField:
# valence [-1,1] (range 2), arousal [0,1] (range 1), dominance [0,1] (range 1)
# => sqrt(4 + 1 + 1) = sqrt(6)
_MAX_MOOD_DIST: float = math.sqrt(6.0)
# Max Euclidean distance in the 2-D CoreAffect space:
# valence [-1,1] (range 2), arousal [0,1] (range 1) => sqrt(4 + 1) = sqrt(5)
_MAX_AFFECT_DIST: float = math.sqrt(5.0)
# Threshold below which momentum magnitudes are treated as zero (float stability)
_MOMENTUM_ZERO_THRESHOLD: float = 1e-12


class AdaptiveWeightsConfig(BaseModel):
    """Parameters for smooth mood-modulated retrieval weight adjustment.

    Each of the three psychological conditions (negative mood, high arousal,
    calm/neutral) is modelled as a continuous sigmoid or Gaussian gate rather
    than a hard threshold, eliminating discontinuities in retrieval behaviour.

    ``sharpness`` controls how step-like the transition is: a value of ~50
    approximates the old hard-threshold behaviour; the default of 5.0 produces
    a smooth transition centred at the given point.
    """

    # Negative mood: weight shift from semantic → emotional signals
    negative_mood_strength: float = 0.15
    negative_mood_center: float = -0.5
    negative_mood_sharpness: float = 5.0

    # High arousal: weight shift from semantic → momentum alignment
    high_arousal_strength: float = 0.10
    high_arousal_center: float = 0.7
    high_arousal_sharpness: float = 5.0

    # Calm/neutral: weight shift from emotional → semantic
    calm_strength: float = 0.15
    calm_valence_width: float = 0.2
    calm_arousal_center: float = 0.3
    calm_sharpness: float = 5.0


class RetrievalConfig(BaseModel):
    """Parameters for multi-signal retrieval scoring."""

    base_weights: list[float] = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
    """Weights for [semantic, mood_congruence, affect_proximity,
    momentum_alignment, recency, resonance_boost]. Must have exactly 6 elements."""

    @field_validator("base_weights")
    @classmethod
    def _check_weights_length(cls, v: list[float]) -> list[float]:
        if len(v) != 6:
            raise ValueError(f"base_weights must have exactly 6 elements, got {len(v)}")
        return v

    ape_threshold: float = 0.3
    """Affective Prediction Error threshold to trigger reconsolidation."""

    reconsolidation_learning_rate: float = 0.2
    """Alpha for tag update on reconsolidation. Max shift = 50%."""

    reconsolidation_window_seconds: float = 300.0
    """Seconds after retrieval during which the tag remains labile."""

    candidate_multiplier: int = 3
    """Pre-filter multiplier: when the store has > top_k * candidate_multiplier
    entries, use search_by_embedding to fetch top_k * candidate_multiplier
    candidates before full multi-signal scoring."""

    adaptive_weights_config: AdaptiveWeightsConfig = AdaptiveWeightsConfig()
    """Parameters for smooth mood-modulated weight adjustment."""


class RetrievalSignals(BaseModel):
    """Named retrieval signal values for explainability.

    The six fields map directly to the composite scorer:
      semantic_similarity
      mood_congruence
      affect_proximity
      momentum_alignment
      recency
      resonance
    """

    semantic_similarity: float = 0.0
    mood_congruence: float = 0.0
    affect_proximity: float = 0.0
    momentum_alignment: float = 0.0
    recency: float = 0.0
    resonance: float = 0.0

    def total(self) -> float:
        """Return the sum of all six signals."""
        return (
            self.semantic_similarity
            + self.mood_congruence
            + self.affect_proximity
            + self.momentum_alignment
            + self.recency
            + self.resonance
        )


class RetrievalBreakdown(BaseModel):
    """Structured decomposition of one retrieval score."""

    weights: RetrievalSignals
    raw_signals: RetrievalSignals
    weighted_signals: RetrievalSignals
    total_score: float


class RetrievalExplanation(BaseModel):
    """One retrieved memory plus the score decomposition that ranked it.

    ``score`` and ``breakdown`` describe the ranking-time score before any
    side effects of retrieval (for example reconsolidation or Hebbian
    strengthening) are applied. ``memory`` is the post-retrieval memory state.
    """

    memory: Memory
    score: float
    breakdown: RetrievalBreakdown
    activation_level: float = 0.0
    pass1_rank: int | None = None
    pass2_rank: int | None = None
    selected_as_seed: bool = False
    candidate_count: int = 0


class RankedMemory(BaseModel):
    """A memory plus its retrieval-time score and breakdown."""

    memory: Memory
    score: float
    breakdown: RetrievalBreakdown


class RetrievalPlan(BaseModel):
    """Pure retrieval ranking plan, independent from engine/store side effects.

    This object captures the two-pass ranking result so sync and async engines
    can share one retrieval pipeline and apply their own persistence logic
    afterward.
    """

    weights: RetrievalSignals
    pass1: list[RankedMemory]
    pass2: list[RankedMemory]
    activation_map: dict[str, float]
    seed_ids: list[str]
    candidate_count: int


# ---------------------------------------------------------------------------
# Adaptive weights
# ---------------------------------------------------------------------------


def _smooth_gate(x: float, center: float, sharpness: float) -> float:
    """Smooth sigmoid gate via tanh, centred at ``center``.

    Positive sharpness → activates *above* center (approaches 1 as x → +∞).
    Negative sharpness → activates *below* center (approaches 1 as x → -∞).
    Returns values in [0, 1].
    """
    return 0.5 * (1.0 + math.tanh(sharpness * (x - center)))


def adaptive_weights(
    mood: MoodField,
    base: list[float],
    config: AdaptiveWeightsConfig | None = None,
) -> NDArray[np.float64]:
    """Return retrieval weights modulated by current mood state.

    Mood acts as a weight modulator — not a hard filter but a continuous
    shaping of what signals dominate retrieval (inspired by mood-congruent
    recall research, Bower 1981).

    Three conditions are applied as smooth continuous functions (tanh /
    Gaussian) rather than hard thresholds, eliminating discontinuities at
    boundary values while preserving the same directional intent:
      - Negative mood  → emotional signals dominate (s2, s3 ↑, s1 ↓)
      - High arousal   → momentum alignment dominates (s4 ↑, s1 ↓)
      - Calm/neutral   → semantic retrieval dominates (s1 ↑, s2, s3 ↓)
    """
    cfg = config or AdaptiveWeightsConfig()
    w = np.array(base, dtype=float)

    # Negative mood: sigmoid activates below negative_mood_center
    neg = _smooth_gate(mood.valence, cfg.negative_mood_center, -cfg.negative_mood_sharpness)
    w[1] += cfg.negative_mood_strength * neg * (2.0 / 3.0)  # mood congruence
    w[2] += cfg.negative_mood_strength * neg * (1.0 / 3.0)  # affect proximity
    w[0] -= cfg.negative_mood_strength * neg  # semantic

    # High arousal: sigmoid activates above high_arousal_center
    aro = _smooth_gate(mood.arousal, cfg.high_arousal_center, cfg.high_arousal_sharpness)
    w[3] += cfg.high_arousal_strength * aro  # momentum alignment
    w[0] -= cfg.high_arousal_strength * aro  # semantic

    # Calm/neutral: Gaussian over valence x inverted sigmoid over arousal
    # Peaks at (valence≈0, arousal≈0); falls off smoothly in all directions
    valence_calm = math.exp(-((mood.valence / cfg.calm_valence_width) ** 2))
    arousal_calm = _smooth_gate(mood.arousal, cfg.calm_arousal_center, -cfg.calm_sharpness)
    calm = valence_calm * arousal_calm
    w[0] += cfg.calm_strength * calm  # semantic
    w[1] -= cfg.calm_strength * calm * (2.0 / 3.0)  # mood congruence
    w[2] -= cfg.calm_strength * calm * (1.0 / 3.0)  # affect proximity

    # Clamp negatives and normalise; fall back to uniform weights when all
    # signals cancel out (extreme mood state) to avoid zero-score retrieval.
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    w = w / total if total > 0 else np.full(6, 1.0 / 6.0)
    return np.asarray(w, dtype=np.float64)


# ---------------------------------------------------------------------------
# Sub-signal helpers
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    return cosine_similarity(a, b)


def _mood_congruence(current: MoodField, snapshot: MoodField) -> float:
    """Similarity between current mood and the mood at encoding.

    Implements mood-congruent recall (Bower, 1981): memories are more
    accessible when the current mood matches the encoding mood.

    Maps PAD distance [0, max_dist] → similarity [1, 0].
    max_dist = sqrt(6) ≈ 2.449 for the 3-D PAD space
    (valence [-1,1], arousal [0,1], dominance [0,1]).
    """
    dist = current.distance(snapshot)
    return max(0.0, 1.0 - dist / _MAX_MOOD_DIST)


def _affect_proximity(current: CoreAffect, encoded: CoreAffect) -> float:
    """Similarity in core affect space. Distance <= sqrt(5) ~= 2.236."""
    dist = current.distance(encoded)
    return max(0.0, 1.0 - dist / _MAX_AFFECT_DIST)


def _momentum_alignment(current: AffectiveMomentum, encoded: AffectiveMomentum) -> float:
    """Cosine similarity of velocity vectors (direction alignment).

    Returns 0.5 when one or both are zero (neutral — no information).
    """
    mag_c = current.magnitude()
    mag_e = encoded.magnitude()
    if mag_c < _MOMENTUM_ZERO_THRESHOLD or mag_e < _MOMENTUM_ZERO_THRESHOLD:
        return 0.5
    dot = current.d_valence * encoded.d_valence + current.d_arousal * encoded.d_arousal
    cos = dot / (mag_c * mag_e)
    # Map [-1, +1] → [0, 1]
    return (cos + 1.0) / 2.0


def _resonance_boost(memory_id: str, activation_map: dict[str, float]) -> float:
    """Boost from spreading activation (Collins & Loftus, 1975).

    Returns the pre-computed activation level for this memory from the
    spreading activation map, or 0.0 if the memory was not reached.
    The activation map is built by ``spreading_activation()`` in the
    engine before Pass 2 scoring.
    """
    return activation_map.get(memory_id, 0.0)


# ---------------------------------------------------------------------------
# Explainability helpers
# ---------------------------------------------------------------------------


def _signals_from_values(
    values: Sequence[float],
) -> RetrievalSignals:
    return RetrievalSignals(
        semantic_similarity=values[0],
        mood_congruence=values[1],
        affect_proximity=values[2],
        momentum_alignment=values[3],
        recency=values[4],
        resonance=values[5],
    )


def _component_values(
    query_embedding: list[float],
    query_affect: CoreAffect,
    current_mood: MoodField,
    current_momentum: AffectiveMomentum,
    memory: Memory,
    activation_map: dict[str, float],
    now: datetime,
    decay_config: DecayConfig,
) -> tuple[float, float, float, float, float, float]:
    emb = memory.embedding or []
    s1 = _cosine(query_embedding, emb) if (query_embedding and emb) else 0.0
    s2 = _mood_congruence(current_mood, memory.tag.mood_snapshot)
    s3 = _affect_proximity(query_affect, memory.tag.core_affect)
    s4 = _momentum_alignment(current_momentum, memory.tag.momentum)
    s5 = compute_effective_strength(memory.tag, now, decay_config)
    s6 = _resonance_boost(memory.id, activation_map)
    return (s1, s2, s3, s4, s5, s6)


def retrieval_breakdown(
    query_embedding: list[float],
    query_affect: CoreAffect,
    current_mood: MoodField,
    current_momentum: AffectiveMomentum,
    memory: Memory,
    activation_map: dict[str, float],
    now: datetime,
    decay_config: DecayConfig,
    retrieval_config: RetrievalConfig,
    precomputed_weights: NDArray[np.float64] | None = None,
) -> RetrievalBreakdown:
    """Return the full score decomposition for one memory.

    This is the explainable counterpart to ``retrieval_score()`` and uses the
    exact same scoring path.
    """
    weight_arr = (
        precomputed_weights
        if precomputed_weights is not None
        else adaptive_weights(
            current_mood,
            retrieval_config.base_weights,
            retrieval_config.adaptive_weights_config,
        )
    )
    raw_values = _component_values(
        query_embedding=query_embedding,
        query_affect=query_affect,
        current_mood=current_mood,
        current_momentum=current_momentum,
        memory=memory,
        activation_map=activation_map,
        now=now,
        decay_config=decay_config,
    )
    weight_values = tuple(float(v) for v in weight_arr.tolist())
    weighted_values = tuple(w * s for w, s in zip(weight_values, raw_values, strict=True))

    weighted = _signals_from_values(weighted_values)
    return RetrievalBreakdown(
        weights=_signals_from_values(weight_values),
        raw_signals=_signals_from_values(raw_values),
        weighted_signals=weighted,
        total_score=weighted.total(),
    )


def build_retrieval_plan(
    query_embedding: list[float],
    query_affect: CoreAffect,
    current_mood: MoodField,
    current_momentum: AffectiveMomentum,
    candidates: list[Memory],
    top_k: int,
    now: datetime,
    decay_config: DecayConfig,
    retrieval_config: RetrievalConfig,
    propagation_hops: int,
    spreading_activation_fn: Callable[[set[str], list[Memory], int], dict[str, float]],
    precomputed_weights: NDArray[np.float64] | None = None,
) -> RetrievalPlan:
    """Build the two-pass retrieval ranking plan for a fixed candidate set.

    This function is intentionally pure: it performs no store writes and no
    state mutation. Engines remain responsible for side effects such as
    reconsolidation, retrieval counters, and Hebbian strengthening.

    Args:
        precomputed_weights: When provided, use these weights instead of calling
            ``adaptive_weights()`` — allows the engine to apply an ablation mask
            or any other pre-processing before passing weights to the planner.
    """
    weight_arr = (
        precomputed_weights
        if precomputed_weights is not None
        else adaptive_weights(
            current_mood,
            retrieval_config.base_weights,
            retrieval_config.adaptive_weights_config,
        )
    )

    def _score_all(activation_map: dict[str, float]) -> list[RankedMemory]:
        scored: list[RankedMemory] = []
        for mem in candidates:
            breakdown = retrieval_breakdown(
                query_embedding=query_embedding,
                query_affect=query_affect,
                current_mood=current_mood,
                current_momentum=current_momentum,
                memory=mem,
                activation_map=activation_map,
                now=now,
                decay_config=decay_config,
                retrieval_config=retrieval_config,
                precomputed_weights=weight_arr,
            )
            scored.append(
                RankedMemory(
                    memory=mem,
                    score=breakdown.total_score,
                    breakdown=breakdown,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored

    pass1 = _score_all({})
    seed_ids = [item.memory.id for item in pass1[:top_k]]
    activation_map = spreading_activation_fn(set(seed_ids), candidates, propagation_hops)
    pass2 = _score_all(activation_map) if activation_map else pass1
    return RetrievalPlan(
        weights=_signals_from_values(tuple(float(v) for v in weight_arr.tolist())),
        pass1=pass1,
        pass2=pass2,
        activation_map=activation_map,
        seed_ids=seed_ids,
        candidate_count=len(candidates),
    )


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------


def retrieval_score(
    query_embedding: list[float],
    query_affect: CoreAffect,
    current_mood: MoodField,
    current_momentum: AffectiveMomentum,
    memory: Memory,
    activation_map: dict[str, float],
    now: datetime,
    decay_config: DecayConfig,
    retrieval_config: RetrievalConfig,
    precomputed_weights: NDArray[np.float64] | None = None,
) -> float:
    """Compute the composite retrieval score for a single memory.

    Args:
        activation_map: Pre-computed spreading activation levels from
            ``spreading_activation()``.  Pass an empty dict for Pass 1
            (no resonance boost).  Pass the result of ``spreading_activation()``
            for Pass 2 to enable Collins & Loftus (1975) multi-hop boosting.
        precomputed_weights: Pre-computed adaptive weights from ``adaptive_weights()``.
            Pass this when scoring many memories with the same mood/config to avoid
            redundant computation. If None, weights are computed inline.
    """
    return retrieval_breakdown(
        query_embedding=query_embedding,
        query_affect=query_affect,
        current_mood=current_mood,
        current_momentum=current_momentum,
        memory=memory,
        activation_map=activation_map,
        now=now,
        decay_config=decay_config,
        retrieval_config=retrieval_config,
        precomputed_weights=precomputed_weights,
    ).total_score


# ---------------------------------------------------------------------------
# Reconsolidation
# ---------------------------------------------------------------------------


def affective_prediction_error(expected: CoreAffect, observed: CoreAffect) -> float:
    """Euclidean distance in affect space — the APE signal.

    A large APE indicates the current emotional context significantly
    differs from what the memory was encoded under, triggering reconsolidation.
    """
    return expected.distance(observed)


def reconsolidate(
    tag: EmotionalTag,
    current_affect: CoreAffect,
    ape: float,
    learning_rate: float,
) -> EmotionalTag:
    """Update the tag's core_affect proportionally to the APE.

    alpha = min(ape * learning_rate, 0.5) — linearly scaled, capped at 50% per retrieval.
    Pearce-Hall associability (large errors increase the learning rate) is handled separately
    by update_prediction().
    Only core_affect is updated; all other fields remain unchanged.
    reconsolidation_count is incremented.
    """
    alpha = min(ape * learning_rate, 0.5)
    new_affect = tag.core_affect.lerp(current_affect, alpha)
    return tag.model_copy(
        update={
            "core_affect": new_affect,
            "reconsolidation_count": tag.reconsolidation_count + 1,
        }
    )


# ---------------------------------------------------------------------------
# Pearce-Hall predictive learning
# ---------------------------------------------------------------------------


def compute_ape(tag: EmotionalTag, observed: CoreAffect) -> float:
    """Compute Affective Prediction Error for a memory tag.

    Uses ``tag.expected_affect`` as the reference point when set; otherwise
    falls back to ``tag.core_affect`` (i.e. the encoding-time affect).

    The APE is the Euclidean distance in affect space between the expected
    and observed affect (Pearce & Hall, 1980).
    """
    reference = tag.expected_affect if tag.expected_affect is not None else tag.core_affect
    return affective_prediction_error(reference, observed)


def update_prediction(
    tag: EmotionalTag,
    observed: CoreAffect,
    ape: float,
) -> EmotionalTag:
    """Update the tag's expected_affect via Pearce-Hall associability learning.

    The learning rate is adaptive: large APE increases the rate (fast relearning
    after surprise), small APE decreases it (stable predictions need less updating).

    Learning rate update (EMA of APE toward target LR):
      new_lr = lr + lr_step * ape — current_lr_step  (simplified)
    Concretely:
      target_lr = base_lr * (1 + ape)   (larger APE → higher LR)
      new_lr = lerp(current_lr, target_lr, 0.2)  (smooth adaptation)
    Clamped to [0.05, 0.80].

    The new expected_affect is an EMA blend:
      new_expected = lerp(current_expected, observed, new_lr)

    Args:
        tag:      The tag whose prediction should be updated.
        observed: The affect observed at retrieval time.
        ape:      The pre-computed APE (from compute_ape).

    Returns:
        Updated tag with new ``expected_affect`` and ``prediction_learning_rate``.
        All other fields are unchanged.

    References
    ----------
    Pearce, J. M., & Hall, G. (1980). A model for Pavlovian learning: Variations
    in the effectiveness of conditioned but not of unconditioned stimuli.
    Psychological Review, 87(6), 532-552.
    """
    _LR_MIN: float = 0.05
    _LR_MAX: float = 0.80
    _LR_ADAPT_RATE: float = 0.2  # meta-learning rate for lr update

    current_lr = tag.prediction_learning_rate
    # Pearce-Hall: associability tracks prediction error.
    # APE (clamped to [LR_MIN, LR_MAX]) is the attractor — large surprise
    # drives LR up, accurate predictions drive LR down toward LR_MIN.
    target_lr = max(_LR_MIN, min(_LR_MAX, ape))
    # Smooth adaptation of current_lr toward target_lr
    new_lr = current_lr + _LR_ADAPT_RATE * (target_lr - current_lr)
    new_lr = max(_LR_MIN, min(_LR_MAX, new_lr))

    # Update expected_affect: EMA blend toward observed
    base = tag.expected_affect if tag.expected_affect is not None else tag.core_affect
    new_expected = base.lerp(observed, new_lr)

    return tag.model_copy(
        update={
            "expected_affect": new_expected,
            "prediction_learning_rate": new_lr,
        }
    )
