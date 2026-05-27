"""Decay Engine.

Memory consolidation strength decays as a power law (ACT-R, Anderson 1983),
modulated by:
  - arousal at encoding: high arousal → slower decay (McGaugh 2004)
  - retrieval count: spacing effect → each retrieval slows future decay
  - floor: memories above arousal threshold never fully fade
    (Merleau-Ponty's body memory — habitual emotional patterns persist)
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from emotional_memory.models import EmotionalTag


class DecayConfig(BaseModel):
    """Parameters controlling how consolidation strength decays over time."""

    model_config = ConfigDict(frozen=True)

    base_decay: float = 0.5
    """Power-law decay exponent baseline. Higher = faster decay."""

    arousal_modulation: float = 0.5
    """How much high arousal slows decay. 0 = no effect, 1 = fully suppressed at max arousal."""

    retrieval_boost: float = 0.1
    """Reduction in decay exponent per retrieval (spacing effect)."""

    floor_arousal_threshold: float = 0.7
    """Memories encoded above this arousal level have a minimum strength floor."""

    floor_value: float = 0.1
    """Minimum consolidation_strength for high-arousal memories."""

    min_seconds: float = 1.0
    """Avoid division by zero: treat elapsed < min_seconds as min_seconds."""

    power: float = 1.0
    """Scaling exponent applied to the decay formula.
    Values > 1 accelerate decay; values < 1 slow it.
    1.0 = standard ACT-R power-law."""


def compute_effective_strength(
    tag: EmotionalTag,
    now: datetime,
    config: DecayConfig,
) -> float:
    """Compute current consolidation strength given elapsed time.

    Formula:
        effective_decay = base_decay
                          * (1 - arousal_modulation * arousal_at_encoding)
                          * (1 / (1 + retrieval_boost * retrieval_count))

        strength(t) = initial_strength * elapsed_seconds ^ (-effective_decay)
        strength(t) = max(strength(t), floor)   # if arousal > threshold

    At t=0 (elapsed < min_seconds), returns initial consolidation_strength.
    """
    elapsed = (now - tag.timestamp).total_seconds()
    elapsed = max(elapsed, config.min_seconds)

    arousal = tag.core_affect.arousal
    effective_decay = (
        config.base_decay
        * (1.0 - config.arousal_modulation * arousal)
        * (1.0 / (1.0 + config.retrieval_boost * tag.retrieval_count))
    )
    effective_decay = max(effective_decay, 0.0)

    strength = tag.consolidation_strength * (elapsed ** (-effective_decay * config.power))
    # Decay must never boost strength above the initial consolidation value.
    # (sub-min_seconds elapsed times can produce elapsed^(-x) > 1 when min_seconds < 1)
    strength = min(strength, tag.consolidation_strength)
    strength = max(0.0, strength)

    if arousal >= config.floor_arousal_threshold:
        strength = max(strength, config.floor_value)

    return float(strength)


def compute_effective_strength_batch(
    tags: Sequence[EmotionalTag],
    now: datetime,
    config: DecayConfig,
) -> NDArray[np.float64]:
    """Vectorized version of ``compute_effective_strength`` over a sequence of tags.

    Identical semantics to the scalar form; uses numpy broadcasting for 10-50x
    speedup on stores larger than ~10k memories.

    Returns an array of length ``len(tags)`` in the same order as *tags*.
    Returns an empty array for empty input.
    """
    if not tags:
        return np.empty(0, dtype=np.float64)

    elapsed = np.array([(now - t.timestamp).total_seconds() for t in tags], dtype=np.float64)
    elapsed = np.maximum(elapsed, config.min_seconds)

    arousal = np.array([t.core_affect.arousal for t in tags], dtype=np.float64)
    retrieval = np.array([t.retrieval_count for t in tags], dtype=np.float64)
    consolidation = np.array([t.consolidation_strength for t in tags], dtype=np.float64)

    effective_decay = (
        config.base_decay
        * (1.0 - config.arousal_modulation * arousal)
        * (1.0 / (1.0 + config.retrieval_boost * retrieval))
    )
    effective_decay = np.maximum(effective_decay, 0.0)

    strength = consolidation * np.power(elapsed, -effective_decay * config.power)
    strength = np.minimum(strength, consolidation)
    strength = np.maximum(strength, 0.0)

    above_floor = arousal >= config.floor_arousal_threshold
    strength = np.where(above_floor, np.maximum(strength, config.floor_value), strength)

    return strength.astype(np.float64)
