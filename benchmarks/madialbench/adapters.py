"""Adapters for the MADial-Bench retrieval benchmark (Addendum X).

Both arms share the same embedder and the same document text (the memory
``event`` field only — third-party emotion labels never enter the semantic
channel; see pre-registration §Protocol). Ingest happens once per arm; queries
run sequentially in file order (production behavior: Hebbian strengthening and
APE-gated reconsolidation stay active in the AFT arm).
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod

from benchmarks.common.similarity import cosine
from benchmarks.madialbench.dataset import MadialMemory
from emotional_memory import (
    DIRECT_VAD_SCHEMA,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
)
from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal import AppraisalEngine
from emotional_memory.appraisal_llm import (
    KeywordAppraisalEngine,
    LLMAppraisalConfig,
    LLMAppraisalEngine,
)
from emotional_memory.decay import DecayConfig
from emotional_memory.embedders import SentenceTransformerEmbedder
from emotional_memory.llm_http import OpenAICompatibleLLMConfig, make_httpx_llm


class MadialAdapter(ABC):
    """Ingest the full memory bank once, then answer ranked-retrieval queries."""

    name: str = "unnamed"

    @abstractmethod
    def ingest(self, memories: list[MadialMemory]) -> None: ...

    @abstractmethod
    def retrieve(self, query_text: str, *, top_k: int) -> list[int]:
        """Return ranked memory ids (length <= top_k)."""

    def close(self) -> None:
        """Release adapter resources; no-op by default."""
        return None


def _make_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder.make_bge_small()


class NaiveCosineMadialAdapter(MadialAdapter):
    """Pure semantic baseline: cosine over ``event``-text embeddings."""

    name = "naive_cosine"

    def __init__(self, *, embedder: SentenceTransformerEmbedder | None = None) -> None:
        self._embedder = embedder if embedder is not None else _make_embedder()
        self._store: list[tuple[int, list[float]]] = []

    def ingest(self, memories: list[MadialMemory]) -> None:
        self._store = [(m.memory_id, self._embedder.embed(m.event)) for m in memories]

    def retrieve(self, query_text: str, *, top_k: int) -> list[int]:
        qvec = self._embedder.embed(query_text)
        ranked = sorted(self._store, key=lambda e: cosine(qvec, e[1]), reverse=True)
        return [mem_id for mem_id, _ in ranked[:top_k]]


class AFTQueryAppraisedMadialAdapter(MadialAdapter):
    """Primary arm: AFT, oracle-free end to end.

    Encode-time affect: direct-VAD appraisal of the memory ``event`` text.
    Retrieve-time affect: direct-VAD appraisal of the query text, passed via the
    public ``query_affect`` API (s3 override, no state mutation) — the
    production-reachable mechanism of Addendum T.

    Amendment A1: ``DecayConfig(base_decay=0, arousal_modulation=0,
    retrieval_boost=0)`` makes stored strength time-invariant, removing the
    time/file-order confound while keeping the arousal-gated consolidation
    strength (affect channel) active.
    """

    name = "aft_query_appraised"

    def __init__(
        self,
        *,
        dry_run: bool = False,
        embedder: SentenceTransformerEmbedder | None = None,
    ) -> None:
        self._appraiser = _make_appraiser(dry_run=dry_run)
        self._engine = EmotionalMemory(
            store=InMemoryStore(),
            embedder=embedder if embedder is not None else _make_embedder(),
            appraisal_engine=self._appraiser,
            config=EmotionalMemoryConfig(
                decay=DecayConfig(base_decay=0.0, arousal_modulation=0.0, retrieval_boost=0.0),
            ),
        )
        self._memory_id_map: dict[str, int] = {}
        # memory_id -> appraised (valence, arousal), for diagnostic D1.
        self.encoded_affect: dict[int, tuple[float, float]] = {}
        # query_id order is the caller's concern; keyed by query text like T2A.
        self.appraised_query_affect: dict[str, CoreAffect] = {}

    def ingest(self, memories: list[MadialMemory]) -> None:
        for m in memories:
            stored = self._engine.encode(m.event, metadata={"memory_id": m.memory_id})
            self._memory_id_map[stored.id] = m.memory_id
            ca = stored.tag.core_affect
            self.encoded_affect[m.memory_id] = (ca.valence, ca.arousal)
        # Encoding 160 memories drags mood/momentum; return to baseline so no
        # ingest-order state leaks into retrieval (pre-registration §Protocol).
        self._engine.reset_state()

    def retrieve(self, query_text: str, *, top_k: int) -> list[int]:
        self._engine.reset_state()
        ca = self._appraiser.appraise(query_text).to_core_affect()
        self.appraised_query_affect[query_text] = ca
        memories = self._engine.retrieve(query_text, top_k=top_k, query_affect=ca)
        return [self._memory_id_map[m.id] for m in memories if m.id in self._memory_id_map]

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._engine.close()


def _make_appraiser(*, dry_run: bool) -> AppraisalEngine:
    if dry_run:
        # Smoke mode: no LLM, no API key — keyword appraisal stands in so the
        # pipeline is exercised end to end. Never used for scored runs.
        return KeywordAppraisalEngine()
    cfg = OpenAICompatibleLLMConfig.from_env()
    if cfg is None:
        raise RuntimeError(
            "EMOTIONAL_MEMORY_LLM_API_KEY not set — required for the scored run "
            "(use --dry-run for the no-LLM smoke test)."
        )
    return LLMAppraisalEngine(
        llm=make_httpx_llm(cfg),
        config=LLMAppraisalConfig(
            cache_size=4096, fallback_on_error=True, appraisal_schema=DIRECT_VAD_SCHEMA
        ),
    )
