"""Step 10: EmotionalMemory — the main facade.

Orchestrates the full AFT pipeline:

  encode():
    event → appraisal → core_affect update → AffectiveState update
    → EmotionalTag (snapshot of all 5 layers) → embed → store → resonance links

  retrieve():
    query → embed → multi-signal score per candidate
    → top-k → reconsolidation check on each result

  get_state() / set_affect():
    read and write the runtime affective state
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import BaseModel

from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal import AppraisalEngine, AppraisalVector, consolidation_strength
from emotional_memory.decay import DecayConfig
from emotional_memory.interfaces import Embedder, MemoryStore
from emotional_memory.models import Memory, ResonanceLink, make_emotional_tag
from emotional_memory.mood import MoodDecayConfig, MoodField
from emotional_memory.resonance import (
    ResonanceConfig,
    build_resonance_links,
    hebbian_strengthen,
    spreading_activation,
)
from emotional_memory.retrieval import (
    RetrievalConfig,
    adaptive_weights,
    affective_prediction_error,
    reconsolidate,
    retrieval_score,
)
from emotional_memory.state import AffectiveState

logger = logging.getLogger(__name__)


class EmotionalMemoryConfig(BaseModel):
    decay: DecayConfig = DecayConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    resonance: ResonanceConfig = ResonanceConfig()
    mood_alpha: float = 0.1
    mood_decay: MoodDecayConfig | None = None
    """Time-based regression of mood toward baseline. None = disabled."""


class EmotionalMemory:
    """Emotional memory system for LLMs based on Affective Field Theory.

    Usage::

        store = InMemoryStore()
        embedder = MyEmbedder()          # implements Embedder protocol
        engine = EmotionalMemory(store, embedder)

        # Encode with automatic appraisal (if engine provided) or manual affect
        engine.set_affect(CoreAffect(valence=0.8, arousal=0.6))
        memory = engine.encode("Just completed a difficult project successfully!")

        # Retrieve — emotionally weighted by current state
        results = engine.retrieve("challenging work accomplishment")
    """

    __slots__ = ("_appraisal_engine", "_config", "_embedder", "_state", "_store")

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        appraisal_engine: AppraisalEngine | None = None,
        config: EmotionalMemoryConfig | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._appraisal_engine = appraisal_engine
        self._config = config or EmotionalMemoryConfig()
        self._state = AffectiveState.initial()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(store={self._store!r}, memories={len(self._store)})"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        content: str,
        appraisal: AppraisalVector | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Encode content into emotional memory.

        Pipeline:
          1. Compute/use appraisal → derive CoreAffect
          2. Update AffectiveState (core_affect, momentum, mood)
          3. Build EmotionalTag (snapshot of all 5 layers)
          4. Embed content
          5. Store memory
          6. Build resonance links against existing memories
          7. Update stored memory with resonance links
        """
        now = datetime.now(tz=UTC)
        logger.debug("encode start: content_len=%d", len(content))

        # Step 1: resolve affect
        if appraisal is None and self._appraisal_engine is not None:
            appraisal = self._appraisal_engine.appraise(content, context=metadata)

        if appraisal is not None:
            new_affect = appraisal.to_core_affect()
        else:
            new_affect = self._state.core_affect

        # Step 2: update affective state
        self._state = self._state.update(
            new_affect,
            now=now,
            mood_alpha=self._config.mood_alpha,
            mood_decay=self._config.mood_decay,
        )

        # Step 3: build EmotionalTag
        cs = consolidation_strength(new_affect.arousal, self._state.mood.arousal)
        tag = make_emotional_tag(
            core_affect=self._state.core_affect,
            momentum=self._state.momentum,
            mood=self._state.mood,
            consolidation_strength=cs,
            appraisal=appraisal,
        )

        # Step 4: embed
        embedding = self._embedder.embed(content)
        if embedding and bool(np.isnan(np.asarray(embedding)).any()):
            warnings.warn(
                f"Embedder returned NaN values for content (len={len(content)}). "
                "Semantic retrieval will be degraded for this memory.",
                stacklevel=2,
            )

        # Step 5: create and store memory (without resonance links yet)
        memory = Memory.create(content=content, tag=tag, embedding=embedding, metadata=metadata)
        self._store.save(memory)
        logger.debug(
            "encode stored: id=%s valence=%.3f arousal=%.3f cs=%.3f",
            memory.id,
            tag.core_affect.valence,
            tag.core_affect.arousal,
            cs,
        )

        # Step 6: build resonance links — pre-filter when store is large (G2)
        resonance_limit = (
            self._config.resonance.max_links * self._config.resonance.candidate_multiplier
        )
        if len(self._store) > resonance_limit and memory.embedding is not None:
            candidates = self._store.search_by_embedding(memory.embedding, resonance_limit)
        else:
            candidates = self._store.list_all()
        links = build_resonance_links(memory, candidates, self._config.resonance)

        if links:
            updated_tag = tag.model_copy(update={"resonance_links": links})
            memory = memory.model_copy(update={"tag": updated_tag})
            self._store.update(memory)
            logger.debug("encode resonance: id=%s links=%d", memory.id, len(links))

            # Bidirectional links: add a backward link on each target memory
            # so that spreading activation can traverse the graph in both
            # directions (new→old and old→new).
            for link in links:
                target_mem = self._store.get(link.target_id)
                if target_mem is None:
                    continue
                backward = ResonanceLink(
                    source_id=link.target_id,
                    target_id=memory.id,
                    strength=link.strength,
                    link_type=link.link_type,
                )
                existing = list(target_mem.tag.resonance_links)
                max_links = self._config.resonance.max_links
                if len(existing) >= max_links:
                    # Replace the weakest link only if backward is stronger
                    weakest_idx = min(range(len(existing)), key=lambda i: existing[i].strength)
                    if backward.strength <= existing[weakest_idx].strength:
                        continue
                    existing[weakest_idx] = backward
                else:
                    existing.append(backward)
                updated_target = target_mem.model_copy(
                    update={"tag": target_mem.tag.model_copy(update={"resonance_links": existing})}
                )
                self._store.update(updated_target)

        return memory

    def retrieve(self, query: str, top_k: int = 5) -> list[Memory]:
        """Retrieve the top-k most relevant memories for the query.

        Scoring uses all 6 AFT signals with mood-adaptive weights.

        Two-pass strategy for spreading activation (G1):
          Pass 1 — score all candidates with empty active_ids to get an initial
                   top-k ranking; their IDs become the active set.
          Pass 2 — re-score with the active set, allowing resonance links to
                   boost memories associated with the initially top-ranked ones.

        Pre-filter (G2): when the store is large, use search_by_embedding to
        narrow candidates to top_k * candidate_multiplier before full scoring.

        Reconsolidation (D1): only triggered if the memory was last retrieved
        within the reconsolidation_window_seconds lability window.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        now = datetime.now(tz=UTC)
        logger.debug(
            "retrieve start: query_len=%d top_k=%d store=%d", len(query), top_k, len(self._store)
        )
        query_embedding = self._embedder.embed(query)

        # G2 — pre-filter candidates via embedding search when store is large
        rc = self._config.retrieval
        candidate_limit = top_k * rc.candidate_multiplier
        if len(self._store) > candidate_limit:
            candidates = self._store.search_by_embedding(query_embedding, candidate_limit)
        else:
            candidates = self._store.list_all()

        if not candidates:
            return []

        # Compute adaptive weights once — invariant across all candidates
        weights = adaptive_weights(self._state.mood, rc.base_weights, rc.adaptive_weights_config)

        def _score_all(activation_map: dict[str, float]) -> list[tuple[float, Memory]]:
            scored = []
            for mem in candidates:
                score = retrieval_score(
                    query_embedding=query_embedding,
                    query_affect=self._state.core_affect,
                    current_mood=self._state.mood,
                    current_momentum=self._state.momentum,
                    memory=mem,
                    activation_map=activation_map,
                    now=now,
                    decay_config=self._config.decay,
                    retrieval_config=rc,
                    precomputed_weights=weights,
                )
                scored.append((score, mem))
            scored.sort(key=lambda t: t[0], reverse=True)
            return scored

        # G1 — spreading activation (Collins & Loftus, 1975)
        # Pass 1: score without resonance to identify seed memories
        pass1 = _score_all({})
        seed_ids = {mem.id for _, mem in pass1[:top_k]}

        # Compute multi-hop activation map from seeds through the link graph
        act_map = spreading_activation(
            seed_ids, candidates, self._config.resonance.propagation_hops
        )

        # Pass 2: re-score with activation map (skip when no activation spread)
        pass2 = _score_all(act_map) if act_map else pass1

        top = [mem for _, mem in pass2[:top_k]]

        # Reconsolidation check and retrieval count update
        window = rc.reconsolidation_window_seconds
        result = []
        for mem in top:
            updated = mem.model_copy(
                update={
                    "tag": mem.tag.model_copy(
                        update={
                            "last_retrieved": now,
                            "retrieval_count": mem.tag.retrieval_count + 1,
                        }
                    )
                }
            )
            ape = affective_prediction_error(mem.tag.core_affect, self._state.core_affect)
            # D1 — only reconsolidate within the lability window
            last = mem.tag.last_retrieved
            in_window = last is not None and (now - last).total_seconds() <= window
            if ape > rc.ape_threshold and in_window:
                updated = updated.model_copy(
                    update={
                        "tag": reconsolidate(
                            updated.tag,
                            self._state.core_affect,
                            ape,
                            rc.reconsolidation_learning_rate,
                        )
                    }
                )
                logger.debug("reconsolidate: id=%s ape=%.3f", mem.id, ape)
            self._store.update(updated)
            result.append(updated)

        # Hebbian co-retrieval strengthening (Hebb, 1949)
        # Strengthen links between memories retrieved together in the same query.
        increment = self._config.resonance.hebbian_increment
        if increment > 0.0 and len(result) > 1:
            co_ids = {m.id for m in result}
            for i, mem in enumerate(result):
                new_links = hebbian_strengthen(mem, co_ids - {mem.id}, increment)
                if new_links != list(mem.tag.resonance_links):
                    strengthened = mem.model_copy(
                        update={"tag": mem.tag.model_copy(update={"resonance_links": new_links})}
                    )
                    self._store.update(strengthened)
                    result[i] = strengthened
                    logger.debug("hebbian: id=%s links_strengthened=%d", mem.id, len(new_links))

        logger.debug("retrieve done: returned=%d candidates=%d", len(result), len(candidates))
        return result

    def encode_batch(
        self,
        contents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[Memory]:
        """Encode multiple contents using a single embed_batch call.

        Uses embed_batch() for efficient batched embedding (G6), then encodes
        each item sequentially so that AffectiveState evolves naturally.
        Resonance links are built against the store state at each step, meaning
        later items in the batch can link to earlier ones.
        """
        if metadata is not None and len(metadata) != len(contents):
            raise ValueError(
                f"metadata length ({len(metadata)}) must match contents length ({len(contents)})"
            )
        embeddings = self._embedder.embed_batch(contents)
        results = []
        for i, (content, embedding) in enumerate(zip(contents, embeddings, strict=True)):
            meta = metadata[i] if metadata else None
            now = datetime.now(tz=UTC)

            # Resolve appraisal per item (mirrors encode())
            appraisal: AppraisalVector | None = None
            if self._appraisal_engine is not None:
                appraisal = self._appraisal_engine.appraise(content, context=meta)

            if appraisal is not None:
                new_affect = appraisal.to_core_affect()
            else:
                new_affect = self._state.core_affect

            self._state = self._state.update(
                new_affect,
                now=now,
                mood_alpha=self._config.mood_alpha,
                mood_decay=self._config.mood_decay,
            )

            cs = consolidation_strength(new_affect.arousal, self._state.mood.arousal)
            tag = make_emotional_tag(
                core_affect=self._state.core_affect,
                momentum=self._state.momentum,
                mood=self._state.mood,
                consolidation_strength=cs,
                appraisal=appraisal,
            )

            if embedding and bool(np.isnan(np.asarray(embedding)).any()):
                warnings.warn(
                    f"Embedder returned NaN values for content[{i}] (len={len(content)}). "
                    "Semantic retrieval will be degraded for this memory.",
                    stacklevel=2,
                )
            memory = Memory.create(content=content, tag=tag, embedding=embedding, metadata=meta)
            self._store.save(memory)

            resonance_limit = (
                self._config.resonance.max_links * self._config.resonance.candidate_multiplier
            )
            if len(self._store) > resonance_limit and memory.embedding is not None:
                candidates = self._store.search_by_embedding(memory.embedding, resonance_limit)
            else:
                candidates = self._store.list_all()
            links = build_resonance_links(memory, candidates, self._config.resonance)
            if links:
                updated_tag = tag.model_copy(update={"resonance_links": links})
                memory = memory.model_copy(update={"tag": updated_tag})
                self._store.update(memory)

                # Bidirectional links: add backward links on target memories
                for link in links:
                    target_mem = self._store.get(link.target_id)
                    if target_mem is None:
                        continue
                    backward = ResonanceLink(
                        source_id=link.target_id,
                        target_id=memory.id,
                        strength=link.strength,
                        link_type=link.link_type,
                    )
                    existing = list(target_mem.tag.resonance_links)
                    max_links = self._config.resonance.max_links
                    if len(existing) >= max_links:
                        weakest_idx = min(range(len(existing)), key=lambda i: existing[i].strength)
                        if backward.strength <= existing[weakest_idx].strength:
                            continue
                        existing[weakest_idx] = backward
                    else:
                        existing.append(backward)
                    updated_target = target_mem.model_copy(
                        update={
                            "tag": target_mem.tag.model_copy(update={"resonance_links": existing})
                        }
                    )
                    self._store.update(updated_target)

            results.append(memory)
        return results

    def delete(self, memory_id: str) -> None:
        """Remove a memory from the store."""
        self._store.delete(memory_id)

    def get(self, memory_id: str) -> Memory | None:
        """Look up a single memory by ID, or None if not found."""
        return self._store.get(memory_id)

    def list_all(self) -> list[Memory]:
        """Return all memories in the store."""
        return self._store.list_all()

    def __len__(self) -> int:
        """Return the number of memories in the store."""
        return len(self._store)

    def get_state(self) -> AffectiveState:
        """Return a copy of the current affective state."""
        return self._state.model_copy()

    def set_affect(self, core_affect: CoreAffect) -> None:
        """Manually inject a CoreAffect (e.g. from external appraisal).

        Updates core_affect, momentum, and mood.
        """
        self._state = self._state.update(
            core_affect,
            mood_alpha=self._config.mood_alpha,
            mood_decay=self._config.mood_decay,
        )

    def save_state(self) -> dict[str, Any]:
        """Serialise the current affective state for persistence.

        Returns a JSON-serialisable dict that can be stored alongside the
        memory store and later restored with ``load_state()``.  The private
        momentum history is included so that momentum computation continues
        correctly after a restart.
        """
        return self._state.snapshot()

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore a previously saved affective state.

        Args:
            data: A dict produced by a previous ``save_state()`` call.
        """
        self._state = AffectiveState.restore(data)

    def prune(self, threshold: float = 0.05) -> int:
        """Remove memories whose effective consolidation strength has fallen below *threshold*.

        Args:
            threshold: Minimum acceptable strength in [0, 1].  Memories whose
                       ``compute_effective_strength()`` is below this value are deleted.
                       Defaults to 0.05 (5%).

        Returns:
            Number of memories removed.
        """
        from emotional_memory.decay import compute_effective_strength

        now = datetime.now(tz=UTC)
        removed = 0
        for memory in self._store.list_all():
            if compute_effective_strength(memory.tag, now, self._config.decay) < threshold:
                self._store.delete(memory.id)
                removed += 1
        return removed

    def export_memories(self) -> list[dict[str, Any]]:
        """Export all memories as a list of JSON-serialisable dicts.

        Suitable for backup or migration between store backends.  The output
        can be restored with ``import_memories()``.
        """
        return [json.loads(m.model_dump_json()) for m in self._store.list_all()]

    def import_memories(self, data: list[dict[str, Any]], *, overwrite: bool = False) -> int:
        """Import memories from a list of dicts produced by ``export_memories()``.

        Args:
            data:      List of memory dicts (as returned by ``export_memories()``).
            overwrite: When True, existing memories with the same ID are replaced.
                       When False (default), duplicates are skipped silently.

        Returns:
            Number of memories actually written.
        """
        written = 0
        for item in data:
            memory = Memory.model_validate(item)
            if not overwrite and self._store.get(memory.id) is not None:
                continue
            self._store.save(memory)
            written += 1
        return written

    def get_current_mood(self, now: datetime | None = None) -> MoodField:
        """Return the mood field regressed to ``now`` without modifying state.

        Useful for read-only mood inspection between encode/retrieve calls.
        If ``mood_decay`` is not configured, returns the frozen mood.
        """
        if now is None:
            now = datetime.now(tz=UTC)
        if self._config.mood_decay is not None:
            return self._state.mood.regress(now, self._config.mood_decay)
        return self._state.mood

    def close(self) -> None:
        """Release resources held by the underlying store, if supported.

        Calls ``store.close()`` when the store exposes that method (e.g.
        ``SQLiteStore``).  Safe to call on stores that do not implement it.
        """
        close = getattr(self._store, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> EmotionalMemory:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
