"""AsyncEmotionalMemory — async-native facade for the AFT pipeline.

Mirrors ``EmotionalMemory`` exactly but awaits all I/O boundaries
(embed, store operations, appraise).  The CPU-bound scoring logic
(adaptive_weights, retrieval_score, build_resonance_links, decay) runs
synchronously inline — no threading overhead for pure computation.

Design follows the httpx / SQLAlchemy pattern: separate ``EmotionalMemory``
(sync) and ``AsyncEmotionalMemory`` (async) classes with no shared base class,
so each API surface is unambiguous and type-checker friendly.

Typical usage::

    from emotional_memory.async_engine import AsyncEmotionalMemory
    from emotional_memory.stores.in_memory import InMemoryStore
    from emotional_memory.async_adapters import SyncToAsyncStore, SyncToAsyncEmbedder

    engine = AsyncEmotionalMemory(
        store=SyncToAsyncStore(InMemoryStore()),
        embedder=SyncToAsyncEmbedder(my_sync_embedder),
    )

    memory = await engine.encode("Something happened")
    results = await engine.retrieve("what happened?")

Or wrap an existing sync engine::

    from emotional_memory.async_adapters import as_async
    async_engine = as_async(sync_engine)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import warnings
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal import AppraisalVector, consolidation_strength
from emotional_memory.categorize import label_tag
from emotional_memory.engine import EmotionalMemoryConfig
from emotional_memory.interfaces import AffectiveStateStore
from emotional_memory.interfaces_async import AsyncAppraisalEngine, AsyncEmbedder, AsyncMemoryStore
from emotional_memory.models import EmotionalTag, Memory, ResonanceLink, make_emotional_tag
from emotional_memory.mood import MoodField
from emotional_memory.resonance import (
    build_resonance_links,
    hebbian_strengthen,
    spreading_activation,
)
from emotional_memory.retrieval import (
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


class AsyncEmotionalMemory:
    """Async-native EmotionalMemory using the AFT pipeline.

    All public methods are coroutines.  An internal ``asyncio.Lock`` protects
    the affective state from interleaved mutations across concurrent
    ``encode`` / ``encode_batch`` calls.  Synchronous helpers
    (``set_affect``, ``reset_state``, ``load_state``) are intended for setup
    **before** concurrent work begins and are **not** lock-protected — do not
    call them while ``encode`` or ``observe`` coroutines may be running.

    When using a ``state_store`` backed by network I/O (e.g.
    ``RedisAffectiveStateStore``), constructing the engine will briefly block
    the event loop to load the persisted snapshot.  For fully non-blocking
    initialisation, pass ``state_store=None`` and explicitly call
    ``await engine.restore_persisted_state()`` after construction.
    """

    __slots__ = (
        "_appraisal_engine",
        "_config",
        "_embedder",
        "_state",
        "_state_lock",
        "_state_store",
        "_store",
    )

    def __init__(
        self,
        store: AsyncMemoryStore,
        embedder: AsyncEmbedder,
        appraisal_engine: AsyncAppraisalEngine | None = None,
        config: EmotionalMemoryConfig | None = None,
        state_store: AffectiveStateStore | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._appraisal_engine = appraisal_engine
        self._config = config or EmotionalMemoryConfig()
        self._state_store = state_store
        self._state = self._load_initial_state()
        self._state_lock: asyncio.Lock = asyncio.Lock()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(store={self._store!r})"

    def _load_initial_state(self) -> AffectiveState:
        if self._state_store is None:
            return AffectiveState.initial()
        persisted = self._state_store.load()
        return AffectiveState.initial() if persisted is None else persisted.model_copy()

    def _effective_retrieval_weights(self) -> NDArray[np.float64]:
        """Return adaptive retrieval weights with ablation mask applied."""
        rc = self._config.retrieval
        weights = adaptive_weights(self._state.mood, rc.base_weights, rc.adaptive_weights_config)
        mask = np.array(
            [
                True,
                self._config.enable_mood_signal,
                True,
                self._config.enable_momentum,
                True,
                self._config.enable_resonance,
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

    def _persist_state_sync(self) -> None:
        if self._state_store is not None:
            self._state_store.save(self._state)

    async def _persist_state_async(self) -> None:
        if self._state_store is not None:
            await asyncio.to_thread(self._state_store.save, self._state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def _build_tag(
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
            appraisal = await self._appraisal_engine.appraise(content, context=metadata)

        new_affect = (
            appraisal.to_core_affect() if appraisal is not None else self._state.core_affect
        )
        async with self._state_lock:
            self._state = self._state.update(
                new_affect,
                now=now,
                mood_alpha=self._config.mood_alpha,
                mood_decay=self._config.mood_decay,
            )
            state_snapshot = self._state
        await self._persist_state_async()

        cs = consolidation_strength(new_affect.arousal, state_snapshot.mood.arousal)
        tag = make_emotional_tag(
            core_affect=state_snapshot.core_affect,
            momentum=state_snapshot.momentum,
            mood=state_snapshot.mood,
            consolidation_strength=cs,
            appraisal=appraisal,
        )
        if use_fast_path:
            tag = tag.model_copy(update={"pending_appraisal": True})
        if self._config.auto_categorize:
            tag = label_tag(tag)
        return tag, use_fast_path

    async def observe(
        self,
        content: str,
        appraisal: AppraisalVector | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EmotionalTag:
        """Update affective state from content without storing a retrievable memory."""
        with traced_span("emotional_memory.observe", {"content_length": len(content)}):
            tag, _ = await self._build_tag(
                content,
                appraisal,
                metadata,
                now=datetime.now(tz=UTC),
                allow_fast_path=False,
            )
        return tag

    async def encode(
        self,
        content: str,
        appraisal: AppraisalVector | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Encode content into emotional memory (async).

        Mirrors ``EmotionalMemory.encode`` with awaited I/O calls.
        """
        with traced_span("emotional_memory.encode", {"content_length": len(content)}):
            now = datetime.now(tz=UTC)
            logger.debug("encode start: content_len=%d", len(content))

            # Steps 1-3: resolve affect, update state, and build EmotionalTag.
            tag, _ = await self._build_tag(
                content,
                appraisal,
                metadata,
                now=now,
                allow_fast_path=True,
            )

            # Step 4: embed (async I/O)
            with traced_span("emotional_memory.embed", {"content_length": len(content)}):
                embedding = await self._embedder.embed(content)
            if embedding and bool(np.isnan(np.asarray(embedding)).any()):
                warnings.warn(
                    f"Embedder returned NaN values for content (len={len(content)}). "
                    "Semantic retrieval will be degraded for this memory.",
                    stacklevel=2,
                )

            # Step 5: store (async I/O)
            memory = Memory.create(
                content=content, tag=tag, embedding=embedding, metadata=metadata
            )
            await self._store.save(memory)

            # Step 6: resonance links — pre-filter when store is large (async I/O)
            if self._config.enable_resonance:
                resonance_limit = (
                    self._config.resonance.max_links * self._config.resonance.candidate_multiplier
                )
                store_size = await self._store.count()
                if store_size > resonance_limit and memory.embedding is not None:
                    with traced_span("emotional_memory.store.search_by_embedding"):
                        candidates = await self._store.search_by_embedding(
                            memory.embedding, resonance_limit
                        )
                else:
                    candidates = await self._store.list_all()

                links = build_resonance_links(memory, candidates, self._config.resonance)
                if links:
                    updated_tag = tag.model_copy(update={"resonance_links": links})
                    memory = memory.model_copy(update={"tag": updated_tag})
                    await self._store.update(memory)
                    logger.debug("encode resonance: id=%s links=%d", memory.id, len(links))

                    await self._add_bidirectional_links(memory, links)

            logger.debug(
                "encode stored: id=%s valence=%.3f arousal=%.3f cs=%.3f",
                memory.id,
                tag.core_affect.valence,
                tag.core_affect.arousal,
                tag.consolidation_strength,
            )
        return memory

    async def retrieve(self, query: str, top_k: int = 5) -> list[Memory]:
        """Retrieve the top-k most relevant memories for the query (async).

        Mirrors ``EmotionalMemory.retrieve`` with awaited I/O calls.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        with traced_span(
            "emotional_memory.retrieve",
            {"query_length": len(query), "top_k": top_k},
        ):
            now = datetime.now(tz=UTC)
            store_count = await self._store.count()
            logger.debug(
                "retrieve start: query_len=%d top_k=%d store=%d", len(query), top_k, store_count
            )
            with traced_span("emotional_memory.embed", {"content_length": len(query)}):
                query_embedding = await self._embedder.embed(query)

            # G2 pre-filter (async)
            rc = self._config.retrieval
            candidate_limit = top_k * rc.candidate_multiplier
            if store_count > candidate_limit:
                with traced_span(
                    "emotional_memory.store.search_by_embedding", {"limit": candidate_limit}
                ):
                    candidates = await self._store.search_by_embedding(
                        query_embedding, candidate_limit
                    )
            else:
                candidates = await self._store.list_all()

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
            result = await self._apply_retrieval_updates(top, now)
            logger.debug("retrieve done: returned=%d candidates=%d", len(result), len(candidates))
        return result

    async def retrieve_with_explanations(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalExplanation]:
        """Async retrieval variant that exposes the ranking breakdown."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        now = datetime.now(tz=UTC)
        store_count = await self._store.count()
        logger.debug(
            "retrieve_with_explanations start: query_len=%d top_k=%d store=%d",
            len(query),
            top_k,
            store_count,
        )
        query_embedding = await self._embedder.embed(query)

        rc = self._config.retrieval
        candidate_limit = top_k * rc.candidate_multiplier
        if store_count > candidate_limit:
            candidates = await self._store.search_by_embedding(query_embedding, candidate_limit)
        else:
            candidates = await self._store.list_all()

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
        result = await self._apply_retrieval_updates([item.memory for item in top_ranked], now)
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

    async def _apply_retrieval_updates(self, top: list[Memory], now: datetime) -> list[Memory]:
        """Apply retrieval side effects after ranking has been computed."""
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
            await self._store.update(updated)
            result.append(updated)

        # Hebbian co-retrieval strengthening (Hebb, 1949)
        increment = self._config.resonance.hebbian_increment
        if increment > 0.0 and len(result) > 1:
            co_ids = {m.id for m in result}
            for i, mem in enumerate(result):
                new_links = hebbian_strengthen(mem, co_ids - {mem.id}, increment)
                if new_links != list(mem.tag.resonance_links):
                    strengthened = mem.model_copy(
                        update={"tag": mem.tag.model_copy(update={"resonance_links": new_links})}
                    )
                    await self._store.update(strengthened)
                    result[i] = strengthened
                    logger.debug("hebbian: id=%s links_strengthened=%d", mem.id, len(new_links))
        return result

    async def encode_batch(
        self,
        contents: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[Memory]:
        """Encode multiple contents using a single embed_batch call (async)."""
        if metadata is not None and len(metadata) != len(contents):
            raise ValueError(
                f"metadata length ({len(metadata)}) must match contents length ({len(contents)})"
            )
        with traced_span("emotional_memory.encode_batch", {"batch_size": len(contents)}):
            with traced_span(
                "emotional_memory.embed", {"content_length": sum(len(c) for c in contents)}
            ):
                embeddings = await self._embedder.embed_batch(contents)
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
                    appraisal = await self._appraisal_engine.appraise(content, context=meta)

                new_affect = (
                    appraisal.to_core_affect()
                    if appraisal is not None
                    else self._state.core_affect
                )
                async with self._state_lock:
                    self._state = self._state.update(
                        new_affect,
                        now=now,
                        mood_alpha=self._config.mood_alpha,
                        mood_decay=self._config.mood_decay,
                    )
                    _state_snap = self._state
                await self._persist_state_async()

                cs = consolidation_strength(new_affect.arousal, _state_snap.mood.arousal)
                tag = make_emotional_tag(
                    core_affect=_state_snap.core_affect,
                    momentum=_state_snap.momentum,
                    mood=_state_snap.mood,
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
                await self._store.save(memory)

                resonance_limit = (
                    self._config.resonance.max_links * self._config.resonance.candidate_multiplier
                )
                store_size = await self._store.count()
                if store_size > resonance_limit and memory.embedding is not None:
                    candidates = await self._store.search_by_embedding(
                        memory.embedding, resonance_limit
                    )
                else:
                    candidates = await self._store.list_all()

                links = build_resonance_links(memory, candidates, self._config.resonance)
                if links:
                    updated_tag = tag.model_copy(update={"resonance_links": links})
                    memory = memory.model_copy(update={"tag": updated_tag})
                    await self._store.update(memory)

                    await self._add_bidirectional_links(memory, links)

                results.append(memory)
        return results

    async def _add_bidirectional_links(self, memory: Memory, links: list[ResonanceLink]) -> None:
        """Persist backward resonance links on each target memory (async).

        Mirrors ``EmotionalMemory._add_bidirectional_links`` with awaited I/O.
        """
        max_links = self._config.resonance.max_links
        for link in links:
            target_mem = await self._store.get(link.target_id)
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
            await self._store.update(updated_target)

    async def _elaborate_with_memory(self, mem: Memory) -> Memory | None:
        """Run the slow-path appraisal on an already-fetched pending memory (async).

        Internal helper shared by ``elaborate()`` and ``elaborate_pending()``
        to avoid a redundant store.get() when the caller already holds the object.
        """
        if not mem.tag.pending_appraisal:
            return None
        if self._appraisal_engine is None:
            return None

        appraisal = await self._appraisal_engine.appraise(mem.content)
        appraised_affect = appraisal.to_core_affect()

        lr = self._config.elaboration_learning_rate
        blended_affect = mem.tag.core_affect.lerp(appraised_affect, lr)

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
        await self._store.update(updated_mem)
        logger.debug(
            "elaborate: id=%s blended_valence=%.3f blended_arousal=%.3f",
            mem.id,
            blended_affect.valence,
            blended_affect.arousal,
        )
        return updated_mem

    async def elaborate(self, memory_id: str) -> Memory | None:
        """Run full appraisal on a fast-path (pending) memory and blend core_affect.

        Async version of ``EmotionalMemory.elaborate()``.

        Args:
            memory_id: ID of the memory to elaborate.

        Returns:
            The updated Memory, or None if the memory does not exist,
            is not pending appraisal, or no appraisal engine is configured.
        """
        with traced_span("emotional_memory.elaborate", {"memory_id": memory_id}):
            mem = await self._store.get(memory_id)
            if mem is None:
                return None
            result = await self._elaborate_with_memory(mem)
        return result

    async def elaborate_pending(self) -> list[Memory]:
        """Elaborate all memories with pending_appraisal=True (async).

        Returns:
            List of updated Memory objects (only those that were elaborated).
        """
        results = []
        for mem in await self._store.list_all():
            if mem.tag.pending_appraisal:
                updated = await self._elaborate_with_memory(mem)
                if updated is not None:
                    results.append(updated)
        return results

    async def delete(self, memory_id: str) -> None:
        """Remove a memory from the store."""
        await self._store.delete(memory_id)

    async def get(self, memory_id: str) -> Memory | None:
        """Look up a single memory by ID, or None if not found."""
        return await self._store.get(memory_id)

    async def list_all(self) -> list[Memory]:
        """Return all memories in the store."""
        return await self._store.list_all()

    async def count(self) -> int:
        """Return the number of memories in the store."""
        return await self._store.count()

    def get_state(self) -> AffectiveState:
        """Return a copy of the current affective state."""
        return self._state.model_copy()

    def set_affect(self, core_affect: CoreAffect) -> None:
        """Manually inject a CoreAffect, updating state synchronously."""
        self._state = self._state.update(
            core_affect,
            mood_alpha=self._config.mood_alpha,
            mood_decay=self._config.mood_decay,
        )
        self._persist_state_sync()

    def reset_state(self) -> None:
        """Reset the runtime affective state to its initial baseline."""
        self._state = AffectiveState.initial()
        self._persist_state_sync()

    def save_state(self) -> dict[str, Any]:
        """Serialise the current affective state for persistence."""
        return self._state.snapshot()

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore a previously saved affective state."""
        self._state = AffectiveState.restore(data)
        self._persist_state_sync()

    async def persist_state(self) -> dict[str, Any]:
        """Return and, when configured, persist the current affective state."""
        await self._persist_state_async()
        return self.save_state()

    async def restore_persisted_state(self) -> bool:
        """Load the last persisted affective state from the configured state store."""
        if self._state_store is None:
            return False
        persisted = await asyncio.to_thread(self._state_store.load)
        if persisted is None:
            return False
        self._state = persisted.model_copy()
        return True

    async def clear_persisted_state(self) -> None:
        """Remove the persisted affective-state snapshot, if configured."""
        if self._state_store is not None:
            await asyncio.to_thread(self._state_store.clear)

    async def prune(self, threshold: float = 0.05) -> int:
        """Remove memories whose effective strength has fallen below *threshold* (async).

        Args:
            threshold: Minimum acceptable strength in [0, 1].  Defaults to 0.05 (5%).

        Returns:
            Number of memories removed.
        """
        from emotional_memory.decay import compute_effective_strength

        with traced_span("emotional_memory.prune", {"threshold": threshold}):
            now = datetime.now(tz=UTC)
            to_delete = [
                m.id
                for m in await self._store.list_all()
                if compute_effective_strength(m.tag, now, self._config.decay) < threshold
            ]
            for memory_id in to_delete:
                await self._store.delete(memory_id)
        return len(to_delete)

    async def export_memories(self) -> list[dict[str, Any]]:
        """Export all memories as a list of JSON-serialisable dicts (async)."""
        memories = await self._store.list_all()
        return [m.model_dump(mode="json") for m in memories]

    async def import_memories(self, data: list[dict[str, Any]], *, overwrite: bool = False) -> int:
        """Import memories from a list of dicts produced by ``export_memories()`` (async).

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
            existing = await self._store.get(memory.id)
            if existing is not None:
                if not overwrite:
                    continue
                await self._store.update(memory)
            else:
                await self._store.save(memory)
            written += 1
        return written

    def get_current_mood(self, now: datetime | None = None) -> MoodField:
        """Return the mood field regressed to ``now`` without modifying state."""
        if now is None:
            now = datetime.now(tz=UTC)
        if self._config.mood_decay is not None:
            return self._state.mood.regress(now, self._config.mood_decay)
        return self._state.mood

    async def close(self) -> None:
        """Release resources held by the underlying store, if supported.

        Awaits ``store.close()`` when available; falls back to sync ``close()``
        via ``asyncio.to_thread``; silently skips stores without cleanup.
        """
        close = getattr(self._store, "close", None)
        if close is not None:
            if inspect.iscoroutinefunction(close):
                await close()
            else:
                await asyncio.to_thread(close)
        state_close = getattr(self._state_store, "close", None)
        if callable(state_close):
            await asyncio.to_thread(state_close)

    async def __aenter__(self) -> AsyncEmotionalMemory:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
