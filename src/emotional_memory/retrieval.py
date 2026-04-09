"""Step 7: Retrieval Scoring Engine.

Multi-signal retrieval scoring combining six components:
  s1 — semantic similarity         (cosine, query embedding vs memory)
  s2 — Stimmung congruence         (mood-congruent memory, Bower 1981)
  s3 — core affect proximity       (current affect vs affect at encoding)
  s4 — momentum alignment          (Spinoza: trajectory matters)
  s5 — recency / decay score       (ACT-R power-law)
  s6 — resonance boost             (spreading activation, Bower 1981)

Weights are adaptive: modulated by current Stimmung (Heidegger).
  Negative Stimmung → weight shifts toward emotional signals
  High arousal      → weight shifts toward momentum
  Neutral/calm      → weight shifts toward semantic

Reconsolidation (Nader & Schiller 2000): retrieval computes Affective
Prediction Error (APE) and updates the tag's core_affect if APE exceeds
a threshold.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from emotional_memory.affect import AffectiveMomentum, CoreAffect
from emotional_memory.decay import DecayConfig, compute_effective_strength
from emotional_memory.models import EmotionalTag, Memory, ResonanceLink
from emotional_memory.stimmung import StimmungField


class AdaptiveWeightsConfig(BaseModel):
    """Parameters for smooth Stimmung-modulated retrieval weight adjustment.

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
    """Weights for [semantic, stimmung_congruence, affect_proximity,
    momentum_alignment, recency, resonance_boost]."""

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
    """Parameters for smooth Stimmung-modulated weight adjustment."""


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
    stimmung: StimmungField,
    base: list[float],
    config: AdaptiveWeightsConfig | None = None,
) -> NDArray[np.float64]:
    """Return retrieval weights modulated by current Stimmung.

    Heidegger: mood is not a filter on cognition but its ground — the
    Stimmung shapes what stands out and what recedes.

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
    neg = _smooth_gate(stimmung.valence, cfg.negative_mood_center, -cfg.negative_mood_sharpness)
    w[1] += cfg.negative_mood_strength * neg * (2.0 / 3.0)  # stimmung congruence
    w[2] += cfg.negative_mood_strength * neg * (1.0 / 3.0)  # affect proximity
    w[0] -= cfg.negative_mood_strength * neg  # semantic

    # High arousal: sigmoid activates above high_arousal_center
    aro = _smooth_gate(stimmung.arousal, cfg.high_arousal_center, cfg.high_arousal_sharpness)
    w[3] += cfg.high_arousal_strength * aro  # momentum alignment
    w[0] -= cfg.high_arousal_strength * aro  # semantic

    # Calm/neutral: Gaussian over valence x inverted sigmoid over arousal
    # Peaks at (valence≈0, arousal≈0); falls off smoothly in all directions
    valence_calm = math.exp(-((stimmung.valence / cfg.calm_valence_width) ** 2))
    arousal_calm = _smooth_gate(stimmung.arousal, cfg.calm_arousal_center, -cfg.calm_sharpness)
    calm = valence_calm * arousal_calm
    w[0] += cfg.calm_strength * calm  # semantic
    w[1] -= cfg.calm_strength * calm * (2.0 / 3.0)  # stimmung congruence
    w[2] -= cfg.calm_strength * calm * (1.0 / 3.0)  # affect proximity

    # Clamp negatives and normalise
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total > 0:
        w = w / total
    return np.asarray(w, dtype=np.float64)


# ---------------------------------------------------------------------------
# Sub-signal helpers
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    from emotional_memory._math import cosine_similarity

    return cosine_similarity(a, b)


def _stimmung_congruence(current: StimmungField, snapshot: StimmungField) -> float:
    """Similarity between current Stimmung and the Stimmung at encoding.

    Maps distance [0, max_dist] → similarity [1, 0].
    max_dist ≈ sqrt(4 + 1 + 1) ≈ 2.45 for PAD space.
    """
    dist = current.distance(snapshot)
    return max(0.0, 1.0 - dist / 2.45)


def _affect_proximity(current: CoreAffect, encoded: CoreAffect) -> float:
    """Similarity in core affect space. Distance ≤ sqrt(4+1)≈2.24."""
    dist = current.distance(encoded)
    return max(0.0, 1.0 - dist / 2.24)


def _momentum_alignment(current: AffectiveMomentum, encoded: AffectiveMomentum) -> float:
    """Cosine similarity of velocity vectors (direction alignment).

    Returns 0.5 when one or both are zero (neutral — no information).
    """
    mag_c = current.magnitude()
    mag_e = encoded.magnitude()
    if mag_c == 0.0 or mag_e == 0.0:
        return 0.5
    dot = current.d_valence * encoded.d_valence + current.d_arousal * encoded.d_arousal
    cos = dot / (mag_c * mag_e)
    # Map [-1, +1] → [0, 1]
    return (cos + 1.0) / 2.0


def _resonance_boost(active_ids: list[str], links: list[ResonanceLink]) -> float:
    """Boost from active memories connected via resonance links."""
    if not links or not active_ids:
        return 0.0
    active_set = set(active_ids)
    strengths = [lnk.strength for lnk in links if lnk.target_id in active_set]
    return min(1.0, sum(strengths)) if strengths else 0.0


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------


def retrieval_score(
    query_embedding: list[float],
    query_affect: CoreAffect,
    current_stimmung: StimmungField,
    current_momentum: AffectiveMomentum,
    memory: Memory,
    active_memory_ids: list[str],
    now: datetime,
    decay_config: DecayConfig,
    retrieval_config: RetrievalConfig,
) -> float:
    """Compute the composite retrieval score for a single memory."""
    w = adaptive_weights(
        current_stimmung,
        retrieval_config.base_weights,
        retrieval_config.adaptive_weights_config,
    )

    emb = memory.embedding or []
    s1 = _cosine(query_embedding, emb) if (query_embedding and emb) else 0.0
    s2 = _stimmung_congruence(current_stimmung, memory.tag.stimmung_snapshot)
    s3 = _affect_proximity(query_affect, memory.tag.core_affect)
    s4 = _momentum_alignment(current_momentum, memory.tag.momentum)
    s5 = compute_effective_strength(memory.tag, now, decay_config)
    s6 = _resonance_boost(active_memory_ids, memory.tag.resonance_links)

    return float(w[0] * s1 + w[1] * s2 + w[2] * s3 + w[3] * s4 + w[4] * s5 + w[5] * s6)


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

    alpha = min(ape * learning_rate, 0.5) — max 50% shift per retrieval.
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
