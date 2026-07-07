"""encode_batch appraises the batch in parallel (E1), preserving order/results.

Appraisal depends only on content/context, so a batch's appraisals run
concurrently (async: asyncio.gather + Semaphore; sync: ThreadPoolExecutor) and
are then consumed by the order-preserving state-evolution loop. These tests
assert (a) real concurrency occurs, (b) ``appraisal_max_concurrency=1`` forces
sequential, and (c) results are identical regardless of concurrency.
"""

import asyncio
import threading
import time

import pytest
from conftest import FixedEmbedder

from emotional_memory.appraisal import AppraisalVector
from emotional_memory.async_adapters import SyncToAsyncEmbedder, SyncToAsyncStore
from emotional_memory.async_engine import AsyncEmotionalMemory
from emotional_memory.engine import EmotionalMemory, EmotionalMemoryConfig
from emotional_memory.stores.in_memory import InMemoryStore

_CONTENTS = [f"event number {i}" for i in range(8)]


def _vector_for(text: str) -> AppraisalVector:
    # Distinct, deterministic per content so order is checkable downstream.
    gr = (len(text) % 5) / 5.0
    return AppraisalVector(
        novelty=0.0,
        goal_relevance=gr,
        coping_potential=0.5,
        norm_congruence=0.0,
        self_relevance=0.0,
    )


class SlowSyncAppraiser:
    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.calls: list[str] = []

    def appraise(self, event_text: str, context=None) -> AppraisalVector:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        time.sleep(self.delay)
        with self._lock:
            self._active -= 1
            self.calls.append(event_text)
        return _vector_for(event_text)


class SlowAsyncAppraiser:
    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self._active = 0
        self.max_active = 0
        self.calls: list[str] = []

    async def appraise(self, event_text: str, context=None) -> AppraisalVector:
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        await asyncio.sleep(self.delay)
        self._active -= 1
        self.calls.append(event_text)
        return _vector_for(event_text)


def _sync_engine(appraiser, concurrency: int) -> EmotionalMemory:
    return EmotionalMemory(
        store=InMemoryStore(),
        embedder=FixedEmbedder([1.0, 0.0]),
        appraisal_engine=appraiser,
        config=EmotionalMemoryConfig(appraisal_max_concurrency=concurrency),
    )


def _async_engine(appraiser, concurrency: int) -> AsyncEmotionalMemory:
    return AsyncEmotionalMemory(
        store=SyncToAsyncStore(InMemoryStore()),
        embedder=SyncToAsyncEmbedder(FixedEmbedder([1.0, 0.0])),
        appraisal_engine=appraiser,
        config=EmotionalMemoryConfig(appraisal_max_concurrency=concurrency),
    )


class TestSyncBatchConcurrency:
    def test_parallel_by_default(self):
        appraiser = SlowSyncAppraiser()
        engine = _sync_engine(appraiser, concurrency=8)
        results = engine.encode_batch(_CONTENTS)
        assert [m.content for m in results] == _CONTENTS  # order preserved
        assert sorted(appraiser.calls) == sorted(_CONTENTS)  # one call per item
        assert appraiser.max_active > 1  # genuinely concurrent

    def test_concurrency_one_is_sequential(self):
        appraiser = SlowSyncAppraiser()
        engine = _sync_engine(appraiser, concurrency=1)
        engine.encode_batch(_CONTENTS)
        assert appraiser.max_active == 1

    def test_results_identical_regardless_of_concurrency(self):
        seq = _sync_engine(SlowSyncAppraiser(delay=0.0), concurrency=1).encode_batch(_CONTENTS)
        par = _sync_engine(SlowSyncAppraiser(delay=0.0), concurrency=8).encode_batch(_CONTENTS)
        assert [m.content for m in seq] == [m.content for m in par]
        assert [m.tag.core_affect for m in seq] == [m.tag.core_affect for m in par]


@pytest.mark.asyncio
class TestAsyncBatchConcurrency:
    async def test_parallel_by_default(self):
        appraiser = SlowAsyncAppraiser()
        engine = _async_engine(appraiser, concurrency=8)
        results = await engine.encode_batch(_CONTENTS)
        assert [m.content for m in results] == _CONTENTS
        assert sorted(appraiser.calls) == sorted(_CONTENTS)
        assert appraiser.max_active > 1

    async def test_concurrency_one_is_sequential(self):
        appraiser = SlowAsyncAppraiser()
        engine = _async_engine(appraiser, concurrency=1)
        await engine.encode_batch(_CONTENTS)
        assert appraiser.max_active == 1

    async def test_results_identical_regardless_of_concurrency(self):
        seq = await _async_engine(SlowAsyncAppraiser(delay=0.0), concurrency=1).encode_batch(
            _CONTENTS
        )
        par = await _async_engine(SlowAsyncAppraiser(delay=0.0), concurrency=8).encode_batch(
            _CONTENTS
        )
        assert [m.tag.core_affect for m in seq] == [m.tag.core_affect for m in par]
