"""Shim: wraps a comparative MemoryAdapter as a ReplayAdapter.

Enables Mem0 and LangMem to participate in the realistic replay benchmark
(benchmarks/realistic/runner.py) without re-implementing the multi-session
scenario protocol.

Methodological note: comparative adapters (Mem0, LangMem) ignore the
``valence``/``arousal`` arguments on encode and retrieve.  AFT in contrast
receives preset oracle-affect values from the dataset.  This asymmetry must
be acknowledged in any reporting that uses this shim.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except ImportError:
    pass

# Bridge project-specific keys → standard OpenAI env vars expected by mem0/langmem.
if "OPENAI_API_KEY" not in os.environ and "EMOTIONAL_MEMORY_LLM_API_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["EMOTIONAL_MEMORY_LLM_API_KEY"]
if "OPENAI_BASE_URL" not in os.environ and "EMOTIONAL_MEMORY_LLM_BASE_URL" in os.environ:
    os.environ["OPENAI_BASE_URL"] = os.environ["EMOTIONAL_MEMORY_LLM_BASE_URL"]

from benchmarks.comparative.adapters.base import MemoryAdapter
from benchmarks.realistic.adapters.base import (
    ReplayAdapter,
    ReplayRetrievedItem,
    ReplaySessionEnd,
    ReplaySessionStart,
)


class ComparativeReplayShim(ReplayAdapter):
    """Wraps a comparative MemoryAdapter in the ReplayAdapter interface.

    Affect signals are silently ignored — the underlying adapter does not
    consume valence/arousal.  The shim tracks encode count for lightweight
    session-boundary reporting only; it does not implement cross-session
    state persistence (``supports_persisted_state = False``).
    """

    supports_explanations: bool = False
    supports_persisted_state: bool = False

    def __init__(self, inner: MemoryAdapter, *, adapter_name: str) -> None:
        self._inner = inner
        self.name = adapter_name
        self._encode_count: int = 0

    def reset(self) -> None:
        self._inner.reset()
        self._encode_count = 0

    def begin_session(self, session_id: str) -> ReplaySessionStart:
        return ReplaySessionStart(
            state_loaded=False,
            memory_count_start=self._encode_count,
        )

    def encode(
        self,
        *,
        memory_alias: str,
        content: str,
        valence: float,
        arousal: float,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        mid = self._inner.encode(content)
        self._encode_count += 1
        return mid

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        valence: float | None = None,
        arousal: float | None = None,
    ) -> list[ReplayRetrievedItem]:
        items = self._inner.retrieve(query, top_k=top_k)
        return [
            ReplayRetrievedItem(id=item.id, text=item.text, score=item.score) for item in items
        ]

    def end_session(self) -> ReplaySessionEnd:
        return ReplaySessionEnd(memory_count_end=self._encode_count)

    def close(self) -> None:
        close_fn = getattr(self._inner, "close", None)
        if callable(close_fn):
            close_fn()


def make_mem0_replay_shim(workdir: Path) -> ComparativeReplayShim:
    """Factory: Mem0-backed ReplayAdapter.  Raises RuntimeError if mem0ai is unavailable."""
    from benchmarks.comparative.adapters.mem0_adapter import Mem0Adapter

    adapter = Mem0Adapter()
    if not adapter.available:
        raise RuntimeError(f"Mem0 adapter unavailable: {adapter.not_available_reason}")
    return ComparativeReplayShim(adapter, adapter_name="mem0")


def make_langmem_replay_shim(workdir: Path) -> ComparativeReplayShim:
    """Factory: LangMem-backed ReplayAdapter.  Raises RuntimeError if langmem is unavailable."""
    from benchmarks.comparative.adapters.langmem_adapter import LangMemAdapter

    adapter = LangMemAdapter()
    if not adapter.available:
        raise RuntimeError(f"LangMem adapter unavailable: {adapter.not_available_reason}")
    return ComparativeReplayShim(adapter, adapter_name="langmem")
