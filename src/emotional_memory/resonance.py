"""Associative Resonance Builder.

When a memory is encoded, it resonates with existing memories via:
  - semantic similarity   (Aristotle: resemblance)
  - emotional congruence  (Hume's sympathy; Bower's affective network)
  - temporal proximity    (Aristotle: contiguity)

Resonance links form affective clusters that amplify each other during
retrieval (spreading activation, Collins & Loftus 1975; Bower 1981).

Bidirectional links ensure that activation flows in both directions
through the associative network. Hebbian co-retrieval strengthening
(Hebb, 1949) progressively reinforces frequently co-activated links.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from emotional_memory._math import cosine_similarity
from emotional_memory.affect import CoreAffect
from emotional_memory.models import Memory, ResonanceLink


class ResonanceConfig(BaseModel):
    """Parameters for resonance link construction and spreading activation."""

    threshold: float = 0.3
    """Minimum composite resonance score to create a link."""

    max_links: int = 5
    """Maximum resonance links per memory (both forward and backward)."""

    semantic_weight: float = 0.5
    emotional_weight: float = 0.3
    temporal_weight: float = 0.2

    temporal_half_life_seconds: float = Field(default=3600.0, gt=0)
    """Half-life for temporal proximity decay. Must be positive."""

    candidate_multiplier: int = 3
    """Pre-filter multiplier for encode-side resonance building.

    When the store has more than ``max_links * candidate_multiplier`` entries,
    ``search_by_embedding`` is used to narrow candidates before scoring, keeping
    resonance-link construction sub-linear in store size."""

    # -----------------------------------------------------------------------
    # Spreading activation (Collins & Loftus, 1975)
    # -----------------------------------------------------------------------

    propagation_hops: int = Field(default=2, ge=1, le=5)
    """Number of hops for spreading activation (Collins & Loftus, 1975).

    1 = direct neighbours only (one-hop check, minimal overhead).
    2 = two hops, capturing indirect associations with strength decay.
    Higher values reach more distant associations but with diminishing returns
    since each hop multiplies by link strength (< 1.0).
    """

    # -----------------------------------------------------------------------
    # Hebbian co-retrieval strengthening (Hebb, 1949)
    # -----------------------------------------------------------------------

    hebbian_increment: float = Field(default=0.05, ge=0.0, le=0.5)
    """Strength increment applied to links between co-retrieved memories.

    "Neurons that fire together wire together" (Hebb, 1949).  Set to 0.0
    to disable Hebbian strengthening.
    """

    # -----------------------------------------------------------------------
    # Link classification thresholds (configurable, formerly hardcoded)
    # -----------------------------------------------------------------------

    contrastive_temporal_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Minimum temporal proximity for contrastive link classification."""

    contrastive_valence_threshold: float = Field(default=1.0, ge=0.0, le=2.0)
    """Minimum absolute valence difference for contrastive link classification."""

    causal_temporal_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    """Minimum temporal proximity for causal link classification."""

    causal_semantic_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    """Minimum semantic similarity for causal link classification."""


def temporal_proximity(t1: datetime, t2: datetime, half_life_seconds: float = 3600.0) -> float:
    """Exponential decay of temporal proximity.

    proximity = exp(-|delta_t| * ln(2) / half_life)
    Returns 1.0 for simultaneous events, approaches 0 for distant events.
    """
    if half_life_seconds <= 0:
        return 0.0
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
    source_affect: CoreAffect | None = None,
    target_affect: CoreAffect | None = None,
    target_precedes_source: bool = False,
    config: ResonanceConfig | None = None,
) -> Literal["semantic", "emotional", "temporal", "causal", "contrastive"]:
    """Return the dominant associative principle for a link.

    Causal: target memory preceded source in time and they are semantically
    similar — the target may have influenced/caused the source experience.

    Contrastive: temporally close memories with opposing valence (A ↔ B).

    Args:
        target_precedes_source: True when the target memory's timestamp is
            earlier than the source memory's timestamp, enabling causal
            classification.
    """
    cfg = config or ResonanceConfig()

    if (
        source_affect is not None
        and target_affect is not None
        and temporal_prox > cfg.contrastive_temporal_threshold
        and abs(source_affect.valence - target_affect.valence) > cfg.contrastive_valence_threshold
    ):
        return "contrastive"
    if (
        target_precedes_source
        and temporal_prox > cfg.causal_temporal_threshold
        and semantic_sim > cfg.causal_semantic_threshold
    ):
        return "causal"
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

    Excludes self-links. Returns up to config.max_links links above threshold,
    sorted by strength descending.

    Note: This function only builds *forward* links (new_memory → candidate).
    The caller is responsible for creating the corresponding *backward* links
    on each target memory (candidate → new_memory) to maintain a bidirectional
    associative network.
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

        # Causal: target (existing memory) preceded source (new memory) in time.
        # Since new_memory is just being encoded, existing memories almost always
        # satisfy this condition — gated further by temporal_prox and semantic_sim.
        target_precedes_source = mem.tag.timestamp < new_memory.tag.timestamp
        link_type = _classify_link_type(
            sem_sim,
            emo_sim,
            temp_prox,
            source_affect=new_memory.tag.core_affect,
            target_affect=mem.tag.core_affect,
            target_precedes_source=target_precedes_source,
            config=config,
        )
        link = ResonanceLink(
            source_id=new_memory.id,
            target_id=mem.id,
            strength=score,
            link_type=link_type,
        )
        links.append((score, link))

    links.sort(key=lambda t: t[0], reverse=True)
    return [lnk for _, lnk in links[: config.max_links]]


def spreading_activation(
    seed_ids: set[str],
    candidates: list[Memory],
    hops: int = 2,
) -> dict[str, float]:
    """Compute activation levels via spreading activation from seed memories.

    Collins & Loftus (1975): activation spreads from seed nodes through
    associative links with multiplicative strength decay at each hop.
    Because each link has strength < 1.0, activation diminishes naturally
    with distance — no explicit hop-discount factor is needed.

    The adjacency is built from every memory's stored resonance_links,
    which include both forward (new→old) and backward (old→new) links
    when bidirectional link creation is used at encode time.

    Args:
        seed_ids: IDs of initially activated memories (e.g. Pass 1 top-k).
        candidates: All candidate memories in the retrieval pool.
        hops: Number of propagation steps. 1 = direct neighbours only.

    Returns:
        Mapping memory_id → activation level in (0, 1] for non-seed memories
        that were reached by activation spreading. Seeds are excluded from
        the returned map (they are already in the top-k).
    """
    if not seed_ids or not candidates:
        return {}

    # Build adjacency: memory_id → list of (target_id, strength)
    # Only includes memories in candidates (retrieval pool).
    candidate_ids: set[str] = {m.id for m in candidates}
    adj: dict[str, list[tuple[str, float]]] = {}
    for mem in candidates:
        for link in mem.tag.resonance_links:
            if link.target_id in candidate_ids:
                adj.setdefault(mem.id, []).append((link.target_id, link.strength))

    if not adj:
        return {}

    # Spreading activation via BFS with activation levels.
    # Each node accumulates the maximum activation reaching it (not the sum),
    # preventing artificial inflation from multiple converging paths.
    current: dict[str, float] = {sid: 1.0 for sid in seed_ids if sid in candidate_ids}
    accumulated: dict[str, float] = {}

    for _ in range(hops):
        if not current:
            break
        next_wave: dict[str, float] = {}
        for node_id, node_activation in current.items():
            for target_id, strength in adj.get(node_id, []):
                if target_id in seed_ids:
                    continue  # do not feed activation back into seeds
                spread = node_activation * strength
                if spread > next_wave.get(target_id, 0.0):
                    next_wave[target_id] = spread
        # Merge into accumulated, taking maximum
        for tid, val in next_wave.items():
            new_val = min(1.0, val)
            if new_val > accumulated.get(tid, 0.0):
                accumulated[tid] = new_val
        current = next_wave

    return accumulated


def hebbian_strengthen(
    memory: Memory,
    co_retrieved_ids: set[str],
    increment: float,
) -> list[ResonanceLink]:
    """Strengthen links to co-retrieved memories (Hebb, 1949).

    "Neurons that fire together wire together" — when two linked memories
    are both retrieved in the same query, their associative link is
    incrementally strengthened up to a maximum of 1.0.

    Args:
        memory: The memory whose links are being updated.
        co_retrieved_ids: IDs of the other memories retrieved in the same
            query (excluding the memory itself).
        increment: Strength increment per co-retrieval event.

    Returns:
        Updated resonance_links list. Returns the original list unchanged
        if no links were strengthened (avoids unnecessary copies).
    """
    if not co_retrieved_ids or increment == 0.0:
        return list(memory.tag.resonance_links)

    updated: list[ResonanceLink] = []
    changed = False
    for link in memory.tag.resonance_links:
        if link.target_id in co_retrieved_ids:
            new_strength = min(1.0, link.strength + increment)
            if new_strength != link.strength:
                updated.append(link.model_copy(update={"strength": new_strength}))
                changed = True
                continue
        updated.append(link)

    return updated if changed else list(memory.tag.resonance_links)
