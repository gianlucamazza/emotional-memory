"""EmotionalMemory — the main facade.

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

import logging
import warnings
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal import AppraisalEngine, AppraisalVector, consolidation_strength
from emotional_memory.categorize import label_tag
from emotional_memory.decay import DecayConfig
from emotional_memory.interfaces import AffectiveStateStore, Embedder, MemoryStore
from emotional_memory.models import EmotionalTag, Memory, ResonanceLink, make_emotional_tag
from emotional_memory.mood import MoodDecayConfig, MoodField
from emotional_memory.resonance import (
    ResonanceConfig,
    build_resonance_links,
    hebbian_strengthen,
    spreading_activation,
)
from emotional_memory.retrieval import (
    RetrievalConfig,
    RetrievalExplanation,
    adaptive_weights,
    build_retrieval_plan,
    compute_ape,
    reconsolidate,
    update_prediction,
)
from emotional_memory.state import AffectiveState
from emotional_memory.telemetry import traced_span

logger = logging.getLogger(__name__)


class EmotionalMemoryConfig(BaseModel):
    decay: DecayConfig = DecayConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    resonance: ResonanceConfig = ResonanceConfig()
    mood_alpha: float = 0.1
    mood_decay: MoodDecayConfig | None = None
    """Time-based regression of mood toward baseline. None = disabled."""

    dual_path_encoding: bool = False
    """When True, fast-path encoding skips appraisal; call elaborate() later
    to run full appraisal and blend core_affect (LeDoux, 1996)."""

    elaboration_learning_rate: float = 0.7
    """Blend ratio used in elaborate(): 70% appraised / 30% raw affect."""

    auto_categorize: bool = False
    """When True, run Plutchik categorization on encode and store EmotionLabel."""

    enable_appraisal: bool = True
    """When False, skip the appraisal engine call in _build_tag (ablation)."""

    enable_mood_signal: bool = True
    """When False, zero the mood-congruence retrieval weight (ablation of Layer 3)."""

    enable_momentum: bool = True
    """When False, zero the momentum-alignment retrieval weight (ablation of Layer 2)."""

    enable_resonance: bool = True
    """When False, skip resonance link building and spreading activation (ablation of Layer 5)."""

    enable_reconsolidation: bool = True
    """When False, skip the APE-gated reconsolidation window at retrieval time (ablation of
    Pearce-Hall 1980 APE gate). Predictive-learning (update_prediction) still runs."""


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

    __slots__ = ("_appraisal_engine", "_config", "_embedder", "_state", "_state_store", "_store")

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        appraisal_engine: AppraisalEngine | None = None,
        config: EmotionalMemoryConfig | None = None,
        state_store: AffectiveStateStore | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._appraisal_engine = appraisal_engine
        self._config = config or EmotionalMemoryConfig()
        self._state_store = state_store
        self._state = self._load_initial_state()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(store={self._store!r}, memories={len(self._store)})"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_tag(
        self,
        content: str,
        appraisal: AppraisalVector | None,
        metadata: dict[str, Any] | None,
        *,
        now: datetime,
        allow_fast_path: bool,
    ) -> tuple[EmotionalTag, bool]:
        """Advance affective state for *content* and return the resulting tag."""
        use_fast_path = (
            allow_fast_path
            and self._config.dual_path_encoding
            and appraisal is None
            and self._appraisal_engine is not None
        )

        if (
            not use_fast_path
            and appraisal is None
            and self._appraisal_engine is not None
            and self._config.enable_appraisal
        ):
            appraisal = self._appraisal_engine.appraise(content, context=metadata)

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
        self._persist_state()

        cs = consolidation_strength(new_affect.arousal, self._state.mood.arousal)
        tag = make_emotional_tag(
            core_affect=self._state.core_affect,
            momentum=self._state.momentum,
            mood=self._state.mood,
            consolidation_strength=cs,
            appraisal=appraisal,
        )
        if use_fast_path:
            tag = tag.model_copy(update={"pending_appraisal": True})
        if self._config.auto_categorize:
            tag = label_tag(tag)
        return tag, use_fast_path

    def _load_initial_state(self) -> AffectiveState:
        if self._state_store is None:
            return AffectiveState.initial()
        persisted = self._state_store.load()
        return AffectiveState.initial() if persisted is None else persisted.model_copy()

    def _persist_state(self) -> None:
        if self._state_store is not None:
            self._state_store.save(self._state)

    def _effective_retrieval_weights(self) -> NDArray[np.float64]:
        """Return adaptive retrieval weights with ablation mask applied.

        Computes mood-modulated weights, then zeroes disabled signal slots and
        renormalises so the active signals absorb the freed mass.
        """
        rc = self._config.retrieval
        weights = adaptive_weights(self._state.mood, rc.base_weights, rc.adaptive_weights_config)
        mask = np.array(
            [
                True,  # s0 semantic — always active
                self._config.enable_mood_signal,  # s1 mood congruence (Bower 1981)
                True,  # s2 affect_proximity — always active
                self._config.enable_momentum,  # s3 momentum alignment
                True,  # s4 recency — always active
                self._config.enable_resonance,  # s5 resonance boost
            ],
            dtype=bool,
        )
        if mask.all():
            return weights
        weights = weights * mask.astype(np.float64)
        total = float(weights.sum())
        if total > 0.0:
            return weights / total
        active = mask.astype(np.float64)
        active_total = float(active.sum())
        if active_total > 0.0:
            return active / active_total
        return np.full(6, 1.0 / 6.0, dtype=np.float64)

    def observe(
        self,
        content: str,
        appraisal: AppraisalVector | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EmotionalTag:
        """Update affective state from content without storing a retrievable memory."""
        with traced_span("emotional_memory.observe", {"content_length": len(content)}):
            now = datetime.now(tz=UTC)
            tag, _ = self._build_tag(
                content,
                appraisal,
                metadata,
                now=now,
                allow_fast_path=False,
            )
        return tag

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
        with traced_span("emotional_memory.encode", {"content_length": len(content)}):
            now = datetime.now(tz=UTC)
            logger.debug("encode start: content_len=%d", len(content))

            # Steps 1-3: resolve affect, update state, and build EmotionalTag.
            tag, _ = self._build_tag(
                content,
                appraisal,
                metadata,
                now=now,
                allow_fast_path=True,
            )

            # Step 4: embed
            with traced_span("emotional_memory.embed", {"content_length": len(content)}):
                embedding = self._embedder.embed(content)
            if embedding and bool(np.isnan(np.asarray(embedding)).any()):
                warnings.warn(
                    f"Embedder returned NaN values for content (len={len(content)}). "
                    "Semantic retrieval will be degraded for this memory.",
                    stacklevel=2,
                )

            # Step 5: create and store memory (without resonance links yet)
            memory = Memory.create(
                content=content, tag=tag, embedding=embedding, metadata=metadata
            )
            self._store.save(memory)
            logger.debug(
                "encode stored: id=%s valence=%.3f arousal=%.3f cs=%.3f",
                memory.id,
                tag.core_affect.valence,
                tag.core_affect.arousal,
                tag.consolidation_strength,
            )

            # Step 6: build resonance links — pre-filter when store is large (G2)
            if self._config.enable_resonance:
                resonance_limit = (
                    self._config.resonance.max_links * self._config.resonance.candidate_multiplier
                )
                if len(self._store) > resonance_limit and memory.embedding is not None:
                    with traced_span("emotional_memory.store.search_by_embedding"):
                        candidates = self._store.search_by_embedding(
                            memory.embedding, resonance_limit
                        )
                else:
                    candidates = self._store.list_all()
                links = build_resonance_links(memory, candidates, self._config.resonance)

                if links:
                    updated_tag = tag.model_copy(update={"resonance_links": links})
                    memory = memory.model_copy(update={"tag": updated_tag})
                    self._store.update(memory)
                    logger.debug("encode resonance: id=%s links=%d", memory.id, len(links))

                    self._add_bidirectional_links(memory, links)

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
        with traced_span(
            "emotional_memory.retrieve",
            {"query_length": len(query), "top_k": top_k, "store_size": len(self._store)},
        ):
            now = datetime.now(tz=UTC)
            logger.debug(
                "retrieve start: query_len=%d top_k=%d store=%d",
                len(query),
                top_k,
                len(self._store),
            )
            with traced_span("emotional_memory.embed", {"content_length": len(query)}):
                query_embedding = self._embedder.embed(query)

            # G2 — pre-filter candidates via embedding search when store is large
            rc = self._config.retrieval
            candidate_limit = top_k * rc.candidate_multiplier
            if len(self._store) > candidate_limit:
                with traced_span(
                    "emotional_memory.store.search_by_embedding", {"limit": candidate_limit}
                ):
                    candidates = self._store.search_by_embedding(query_embedding, candidate_limit)
            else:
                candidates = self._store.list_all()

            if not candidates:
                return []

            _spreading_fn = (
                (lambda *a, **kw: {})
                if not self._config.enable_resonance
                else spreading_activation
            )
            plan = build_retrieval_plan(
                query_embedding=query_embedding,
                query_affect=self._state.core_affect,
                current_mood=self._state.mood,
                current_momentum=self._state.momentum,
                candidates=candidates,
                top_k=top_k,
                now=now,
                decay_config=self._config.decay,
                retrieval_config=rc,
                propagation_hops=self._config.resonance.propagation_hops,
                spreading_activation_fn=_spreading_fn,
                precomputed_weights=self._effective_retrieval_weights(),
            )
            top = [item.memory for item in plan.pass2[:top_k]]
            result = self._apply_retrieval_updates(top, now)
            logger.debug("retrieve done: returned=%d candidates=%d", len(result), len(candidates))
        return result

    def retrieve_with_explanations(self, query: str, top_k: int = 5) -> list[RetrievalExplanation]:
        """Retrieve memories plus a structured score decomposition.

        Ranking is computed by the same two-pass retrieval pipeline used by
        ``retrieve()``. The returned score and breakdown reflect the ranking
        state before retrieval-side updates such as reconsolidation and Hebbian
        strengthening; the returned ``memory`` objects reflect the post-
        retrieval stored state.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        now = datetime.now(tz=UTC)
        logger.debug(
            "retrieve_with_explanations start: query_len=%d top_k=%d store=%d",
            len(query),
            top_k,
            len(self._store),
        )
        query_embedding = self._embedder.embed(query)

        rc = self._config.retrieval
        candidate_limit = top_k * rc.candidate_multiplier
        if len(self._store) > candidate_limit:
            candidates = self._store.search_by_embedding(query_embedding, candidate_limit)
        else:
            candidates = self._store.list_all()

        if not candidates:
            return []

        _spreading_fn = (
            (lambda *a, **kw: {}) if not self._config.enable_resonance else spreading_activation
        )
        plan = build_retrieval_plan(
            query_embedding=query_embedding,
            query_affect=self._state.core_affect,
            current_mood=self._state.mood,
            current_momentum=self._state.momentum,
            candidates=candidates,
            top_k=top_k,
            now=now,
            decay_config=self._config.decay,
            retrieval_config=rc,
            propagation_hops=self._config.resonance.propagation_hops,
            spreading_activation_fn=_spreading_fn,
            precomputed_weights=self._effective_retrieval_weights(),
        )
        top_ranked = plan.pass2[:top_k]
        result = self._apply_retrieval_updates([item.memory for item in top_ranked], now)
        updated_by_id = {mem.id: mem for mem in result}
        pass1_ranks = {item.memory.id: index for index, item in enumerate(plan.pass1, start=1)}

        explanations = [
            RetrievalExplanation(
                memory=updated_by_id[item.memory.id],
                score=item.score,
                breakdown=item.breakdown,
                activation_level=plan.activation_map.get(item.memory.id, 0.0),
                pass1_rank=pass1_ranks.get(item.memory.id),
                pass2_rank=index,
                selected_as_seed=item.memory.id in plan.seed_ids,
                candidate_count=plan.candidate_count,
            )
            for index, item in enumerate(top_ranked, start=1)
        ]
        logger.debug(
            "retrieve_with_explanations done: returned=%d candidates=%d",
            len(explanations),
            len(candidates),
        )
        return explanations

    def _apply_retrieval_updates(self, top: list[Memory], now: datetime) -> list[Memory]:
        """Apply retrieval side effects after ranking has been computed."""
        # Reconsolidation check and retrieval count update
        cfg = self._config.retrieval
        result = []
        for mem in top:
            tag = mem.tag.model_copy(
                update={
                    "last_retrieved": now,
                    "retrieval_count": mem.tag.retrieval_count + 1,
                }
            )
            ape = compute_ape(tag, self._state.core_affect)

            if self._config.enable_reconsolidation:
                # APE-gated reconsolidation window (Nader & Schiller, 2000)
                # HIGH APE: open window + reconsolidate
                if ape > cfg.ape_threshold and tag.window_opened_at is None:
                    tag = reconsolidate(
                        tag, self._state.core_affect, ape, cfg.reconsolidation_learning_rate
                    )
                    tag = tag.model_copy(update={"window_opened_at": now})
                    logger.debug("reconsolidate: id=%s ape=%.3f (window opened)", mem.id, ape)

                # WITHIN OPEN WINDOW: reconsolidate regardless of APE
                elif tag.window_opened_at is not None:
                    elapsed = (now - tag.window_opened_at).total_seconds()
                    if elapsed <= cfg.reconsolidation_window_seconds:
                        tag = reconsolidate(
                            tag, self._state.core_affect, ape, cfg.reconsolidation_learning_rate
                        )
                        logger.debug("reconsolidate: id=%s ape=%.3f (within window)", mem.id, ape)
                    else:
                        # Window expired — close it
                        tag = tag.model_copy(update={"window_opened_at": None})

            # Pearce-Hall predictive learning: always update prediction
            tag = update_prediction(tag, self._state.core_affect, ape)

            updated = mem.model_copy(update={"tag": tag})
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
        with traced_span("emotional_memory.encode_batch", {"batch_size": len(contents)}):
            with traced_span(
                "emotional_memory.embed", {"content_length": sum(len(c) for c in contents)}
            ):
                embeddings = self._embedder.embed_batch(contents)
            results = []
            for i, (content, embedding) in enumerate(zip(contents, embeddings, strict=True)):
                meta = metadata[i] if metadata else None
                now = datetime.now(tz=UTC)

                # Resolve appraisal per item (mirrors encode())
                # Dual-path: skip appraisal on fast path
                use_fast_path = (
                    self._config.dual_path_encoding and self._appraisal_engine is not None
                )
                appraisal: AppraisalVector | None = None
                if not use_fast_path and self._appraisal_engine is not None:
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
                self._persist_state()

                cs = consolidation_strength(new_affect.arousal, self._state.mood.arousal)
                tag = make_emotional_tag(
                    core_affect=self._state.core_affect,
                    momentum=self._state.momentum,
                    mood=self._state.mood,
                    consolidation_strength=cs,
                    appraisal=appraisal,
                )

                # Mark pending_appraisal for fast-path memories
                if use_fast_path:
                    tag = tag.model_copy(update={"pending_appraisal": True})

                # Auto-categorize: attach Plutchik EmotionLabel
                if self._config.auto_categorize:
                    tag = label_tag(tag)

                if embedding and bool(np.isnan(np.asarray(embedding)).any()):
                    warnings.warn(
                        f"Embedder returned NaN values for content[{i}] (len={len(content)}). "
                        "Semantic retrieval will be degraded for this memory.",
                        stacklevel=2,
                    )
                memory = Memory.create(
                    content=content, tag=tag, embedding=embedding, metadata=meta
                )
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

                    self._add_bidirectional_links(memory, links)

                results.append(memory)
        return results

    def _elaborate_with_memory(self, mem: Memory) -> Memory | None:
        """Run the slow-path appraisal on an already-fetched pending memory.

        Internal helper shared by ``elaborate()`` and ``elaborate_pending()``
        to avoid a redundant store.get() when the caller already holds the object.
        """
        if not mem.tag.pending_appraisal:
            return None
        if self._appraisal_engine is None:
            return None

        appraisal = self._appraisal_engine.appraise(mem.content)
        appraised_affect = appraisal.to_core_affect()

        # Blend: elaboration_learning_rate controls how much the appraised
        # affect replaces the raw fast-path affect (30% raw, 70% appraised).
        lr = self._config.elaboration_learning_rate
        blended_affect = mem.tag.core_affect.lerp(appraised_affect, lr)

        # Use appraised arousal for consolidation_strength — the elaboration
        # process re-evaluates significance; the appraised arousal captures
        # how emotionally significant the event truly is (McGaugh, 2004).
        new_cs = consolidation_strength(appraised_affect.arousal, mem.tag.mood_snapshot.arousal)

        now = datetime.now(tz=UTC)
        updated_tag = mem.tag.model_copy(
            update={
                "core_affect": blended_affect,
                "appraisal": appraisal,
                "consolidation_strength": new_cs,
                "pending_appraisal": False,
                "window_opened_at": now,
            }
        )
        updated_mem = mem.model_copy(update={"tag": updated_tag})
        self._store.update(updated_mem)
        logger.debug(
            "elaborate: id=%s blended_valence=%.3f blended_arousal=%.3f",
            mem.id,
            blended_affect.valence,
            blended_affect.arousal,
        )
        return updated_mem

    def elaborate(self, memory_id: str) -> Memory | None:
        """Run full appraisal on a fast-path (pending) memory and blend core_affect.

        Dual-path slow path (LeDoux, 1996): runs the full Scherer appraisal
        on the memory's content, blends the appraised affect with the raw
        fast-path affect, updates consolidation_strength, clears
        pending_appraisal, and opens the reconsolidation window.

        Args:
            memory_id: ID of the memory to elaborate.

        Returns:
            The updated Memory, or None if the memory does not exist,
            is not pending appraisal, or no appraisal engine is configured.
        """
        with traced_span("emotional_memory.elaborate", {"memory_id": memory_id}):
            mem = self._store.get(memory_id)
            if mem is None:
                return None
            result = self._elaborate_with_memory(mem)
        return result

    def elaborate_pending(self) -> list[Memory]:
        """Elaborate all memories with pending_appraisal=True.

        Convenience method to run the slow appraisal path on all fast-path
        memories in the store. Each memory is elaborated in sequence.

        Returns:
            List of updated Memory objects (only those that were elaborated).
        """
        results = []
        for mem in self._store.list_all():
            if mem.tag.pending_appraisal:
                updated = self._elaborate_with_memory(mem)
                if updated is not None:
                    results.append(updated)
        return results

    def delete(self, memory_id: str) -> None:
        """Remove a memory from the store."""
        self._store.delete(memory_id)

    def get(self, memory_id: str) -> Memory | None:
        """Look up a single memory by ID, or None if not found."""
        return self._store.get(memory_id)

    def _add_bidirectional_links(self, memory: Memory, links: list[ResonanceLink]) -> None:
        """Persist backward resonance links on each target memory.

        Called after forward links have already been saved on *memory*.  Each
        link in *links* gets a symmetric backward entry on its target so that
        spreading activation can traverse the graph in both directions.
        """
        max_links = self._config.resonance.max_links
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
            if len(existing) >= max_links:
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
        self._persist_state()

    def reset_state(self) -> None:
        """Reset the runtime affective state to its initial baseline."""
        self._state = AffectiveState.initial()
        self._persist_state()

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
        self._persist_state()

    def persist_state(self) -> dict[str, Any]:
        """Return and, when configured, persist the current affective state."""
        self._persist_state()
        return self.save_state()

    def restore_persisted_state(self) -> bool:
        """Load the last persisted affective state from the configured state store."""
        if self._state_store is None:
            return False
        persisted = self._state_store.load()
        if persisted is None:
            return False
        self._state = persisted.model_copy()
        return True

    def clear_persisted_state(self) -> None:
        """Remove the persisted affective-state snapshot, if configured."""
        if self._state_store is not None:
            self._state_store.clear()

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

        with traced_span("emotional_memory.prune", {"threshold": threshold}):
            now = datetime.now(tz=UTC)
            to_delete = [
                m.id
                for m in self._store.list_all()
                if compute_effective_strength(m.tag, now, self._config.decay) < threshold
            ]
            for memory_id in to_delete:
                self._store.delete(memory_id)
        return len(to_delete)

    def export_memories(self) -> list[dict[str, Any]]:
        """Export all memories as a list of JSON-serialisable dicts.

        Suitable for backup or migration between store backends.  The output
        can be restored with ``import_memories()``.
        """
        return [m.model_dump(mode="json") for m in self._store.list_all()]

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
            exists = self._store.get(memory.id) is not None
            if exists and not overwrite:
                continue
            if exists:
                self._store.update(memory)
            else:
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
        state_close = getattr(self._state_store, "close", None)
        if callable(state_close):
            state_close()

    def __enter__(self) -> EmotionalMemory:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
