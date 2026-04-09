"""Step 8: Associative Resonance Builder.

When a memory is encoded, it resonates with existing memories via:
  - semantic similarity   (Aristotle: resemblance)
  - emotional congruence  (Hume's sympathy; Bower's affective network)
  - temporal proximity    (Aristotle: contiguity)

Resonance links form affective clusters that amplify each other during
retrieval (spreading activation, Bower 1981).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from emotional_memory._math import cosine_similarity
from emotional_memory.affect import CoreAffect
from emotional_memory.models import Memory, ResonanceLink


class ResonanceConfig(BaseModel):
    """Parameters for resonance link construction."""

    threshold: float = 0.3
    """Minimum composite resonance score to create a link."""

    max_links: int = 5
    """Maximum resonance links per memory."""

    semantic_weight: float = 0.5
    emotional_weight: float = 0.3
    temporal_weight: float = 0.2

    temporal_half_life_seconds: float = 3600.0
    """Half-life for temporal proximity decay."""


def temporal_proximity(t1: datetime, t2: datetime, half_life_seconds: float = 3600.0) -> float:
    """Exponential decay of temporal proximity.

    proximity = exp(-|delta_t| * ln(2) / half_life)
    Returns 1.0 for simultaneous events, approaches 0 for distant events.
    """
    delta = abs((t1 - t2).total_seconds())
    return math.exp(-delta * math.log(2) / half_life_seconds)


def _emotional_similarity(a: CoreAffect, b: CoreAffect) -> float:
    """Maps affect distance [0, 2.24] → similarity [1, 0]."""
    dist = a.distance(b)
    return max(0.0, 1.0 - dist / 2.24)


def _classify_link_type(
    semantic_sim: float,
    emotional_sim: float,
    temporal_prox: float,
) -> Literal["semantic", "emotional", "temporal", "causal", "contrastive"]:
    """Return the dominant associative principle for a link."""
    if semantic_sim >= emotional_sim and semantic_sim >= temporal_prox:
        return "semantic"
    if emotional_sim >= temporal_prox:
        return "emotional"
    return "temporal"


def build_resonance_links(
    new_memory: Memory,
    candidates: list[Memory],
    config: ResonanceConfig,
) -> list[ResonanceLink]:
    """Build resonance links from new_memory to the most resonant candidates.

    Excludes self-links. Returns up to config.max_links links above threshold.
    """
    links: list[tuple[float, ResonanceLink]] = []

    for mem in candidates:
        if mem.id == new_memory.id:
            continue

        # Semantic similarity (requires embeddings on both)
        if new_memory.embedding and mem.embedding:
            sem_sim = cosine_similarity(new_memory.embedding, mem.embedding)
        else:
            sem_sim = 0.0

        emo_sim = _emotional_similarity(new_memory.tag.core_affect, mem.tag.core_affect)
        temp_prox = temporal_proximity(
            new_memory.tag.timestamp,
            mem.tag.timestamp,
            config.temporal_half_life_seconds,
        )

        score = (
            config.semantic_weight * sem_sim
            + config.emotional_weight * emo_sim
            + config.temporal_weight * temp_prox
        )

        if score < config.threshold:
            continue

        link_type = _classify_link_type(sem_sim, emo_sim, temp_prox)
        link = ResonanceLink(
            source_id=new_memory.id,
            target_id=mem.id,
            strength=score,
            link_type=link_type,
        )
        links.append((score, link))

    links.sort(key=lambda t: t[0], reverse=True)
    return [lnk for _, lnk in links[: config.max_links]]
