"""H13 structural overhead: dual-path encode skips appraisal on the hot path.

Mirrors ``benchmarks/perf/bench_profile_breakdown.run_h13_sim`` so ``make check``
covers the dual-path latency contract without a live LLM.
"""

from __future__ import annotations

import time
from typing import Any

from conftest import DeterministicEmbedder

from emotional_memory import EmotionalMemory, EmotionalMemoryConfig, InMemoryStore
from emotional_memory.appraisal_llm import KeywordAppraisalEngine


class _CountingDelayedAppraisal:
    def __init__(self, delay_s: float = 0.02) -> None:
        self._inner = KeywordAppraisalEngine()
        self._delay_s = delay_s
        self.calls = 0

    def appraise(self, event_text: str, context: dict[str, Any] | None = None) -> Any:
        self.calls += 1
        time.sleep(self._delay_s)
        return self._inner.appraise(event_text, context=context)


def test_h13_dual_path_hot_path_skips_appraisal_and_is_faster() -> None:
    delay = 0.03
    n = 4
    texts = [f"H13 unit item {i}: proud of the release." for i in range(n)]

    sync_app = _CountingDelayedAppraisal(delay)
    dual_app = _CountingDelayedAppraisal(delay)
    embedder = DeterministicEmbedder()

    em_sync = EmotionalMemory(
        store=InMemoryStore(),
        embedder=embedder,
        appraisal_engine=sync_app,
        config=EmotionalMemoryConfig(dual_path_encoding=False, enable_resonance=False),
    )
    t0 = time.perf_counter()
    for t in texts:
        em_sync.encode(t)
    sync_ms = (time.perf_counter() - t0) * 1000.0

    em_dual = EmotionalMemory(
        store=InMemoryStore(),
        embedder=embedder,
        appraisal_engine=dual_app,
        config=EmotionalMemoryConfig(dual_path_encoding=True, enable_resonance=False),
    )
    t0 = time.perf_counter()
    for t in texts:
        mem = em_dual.encode(t)
        assert mem.tag.pending_appraisal is True
    dual_ms = (time.perf_counter() - t0) * 1000.0

    assert sync_app.calls == n
    assert dual_app.calls == 0
    # Dual hot path must be far cheaper than n * delay
    assert dual_ms < sync_ms * 0.5
    assert dual_ms < delay * n * 1000 * 0.5

    t0 = time.perf_counter()
    done = em_dual.elaborate_pending()
    elab_ms = (time.perf_counter() - t0) * 1000.0
    assert len(done) == n
    assert dual_app.calls == n
    # Elaborate pays the deferred appraisal cost
    assert elab_ms >= delay * n * 1000 * 0.7
