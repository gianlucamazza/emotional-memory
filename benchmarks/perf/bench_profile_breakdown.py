"""Wave-2 overhead profile: isolate embed vs prefilter vs AFT plan vs e2e retrieve.

Answers H12 (does SBERT dominate retrieve wall-time?) and optionally H13
(dual-path encode vs sync LLM appraisal).

Not part of ``make check``. Prefer::

    # H12 — hash + SBERT (requires [sentence-transformers] for SBERT arm)
    uv run python -m benchmarks.perf.bench_profile_breakdown

    # H13 — only when EMOTIONAL_MEMORY_LLM_API_KEY is set
    uv run python -m benchmarks.perf.bench_profile_breakdown --llm-encode

Or via pytest (component timers, not pytest-benchmark)::

    uv run python -m pytest benchmarks/perf/bench_profile_breakdown.py -v -s
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from benchmarks.conftest import ScalableEmbedder, populate_store
from emotional_memory import (
    CoreAffect,
    EmotionalMemory,
    EmotionalMemoryConfig,
    InMemoryStore,
    ResonanceConfig,
)
from emotional_memory.resonance import spreading_activation
from emotional_memory.retrieval import build_retrieval_plan

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

_QUERY = "project work accomplishment"


def _median_ms(fn: Callable[[], Any], *, warmup: int = 3, rounds: int = 25) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(samples))


@dataclass(frozen=True)
class RetrieveBreakdown:
    embedder: str
    n: int
    embed_ms: float
    prefilter_ms: float
    plan_ms: float
    e2e_ms: float

    @property
    def plan_share_of_e2e(self) -> float:
        return 0.0 if self.e2e_ms <= 0 else self.plan_ms / self.e2e_ms

    @property
    def embed_share_of_e2e(self) -> float:
        return 0.0 if self.e2e_ms <= 0 else self.embed_ms / self.e2e_ms


def _measure_retrieve_breakdown(
    *,
    embedder: Any,
    embedder_name: str,
    n: int,
    top_k: int = 5,
    resonance_threshold: float = 0.3,
    rounds: int = 25,
) -> RetrieveBreakdown:
    """Populate store with *embedder*, then time retrieve components."""
    config = EmotionalMemoryConfig(
        resonance=ResonanceConfig(threshold=resonance_threshold),
    )
    engine = EmotionalMemory(store=InMemoryStore(), embedder=embedder, config=config)
    populate_store(engine, n)
    engine.set_affect(CoreAffect(valence=0.5, arousal=0.6))

    store = engine._store
    candidate_limit = top_k * engine._config.retrieval.candidate_multiplier
    # Warm matrix cache + model
    q0 = embedder.embed(_QUERY)
    store.search_by_embedding(q0, candidate_limit)
    engine.retrieve(_QUERY, top_k)

    def do_embed() -> list[float]:
        return embedder.embed(_QUERY)

    embed_ms = _median_ms(do_embed, rounds=rounds)
    query_embedding = embedder.embed(_QUERY)

    def do_prefilter() -> list:
        return store.search_by_embedding(query_embedding, candidate_limit)

    prefilter_ms = _median_ms(do_prefilter, rounds=rounds)
    candidates = store.search_by_embedding(query_embedding, candidate_limit)
    if len(store) <= candidate_limit:
        candidates = store.list_all()

    now = datetime.now(tz=UTC)
    state = engine.get_state()
    query_affect = state.core_affect
    mood = state.mood
    momentum = state.momentum
    decay = engine._config.decay
    retrieval = engine._config.retrieval
    hops = engine._config.resonance.propagation_hops

    def do_plan() -> None:
        build_retrieval_plan(
            query_embedding=query_embedding,
            query_affect=query_affect,
            current_mood=mood,
            current_momentum=momentum,
            candidates=candidates,
            top_k=top_k,
            now=now,
            decay_config=decay,
            retrieval_config=retrieval,
            propagation_hops=hops,
            spreading_activation_fn=spreading_activation,
        )

    plan_ms = _median_ms(do_plan, rounds=rounds)

    def do_e2e() -> None:
        engine.retrieve(_QUERY, top_k)

    e2e_ms = _median_ms(do_e2e, rounds=rounds)
    return RetrieveBreakdown(
        embedder=embedder_name,
        n=n,
        embed_ms=embed_ms,
        prefilter_ms=prefilter_ms,
        plan_ms=plan_ms,
        e2e_ms=e2e_ms,
    )


def _try_sbert() -> Any | None:
    try:
        from emotional_memory.embedders import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder()
    except Exception as exc:
        print(f"[skip SBERT] {exc}")
        return None


def run_h12(*, sizes: tuple[int, ...] = (100, 1_000), rounds: int = 25) -> list[RetrieveBreakdown]:
    rows = [
        _measure_retrieve_breakdown(
            embedder=ScalableEmbedder(dim=64),
            embedder_name="hash-64",
            n=n,
            rounds=rounds,
        )
        for n in sizes
    ]
    sbert = _try_sbert()
    if sbert is not None:
        _ = sbert.embed("warmup")
        rows.extend(
            _measure_retrieve_breakdown(
                embedder=sbert,
                embedder_name="sbert-minilm",
                n=n,
                rounds=max(5, rounds // 2),
            )
            for n in sizes
        )
    return rows


def format_breakdown_table(rows: list[RetrieveBreakdown]) -> str:
    header = (
        "| Embedder | N | embed ms | prefilter ms | AFT plan ms "
        "| e2e retrieve ms | plan/e2e | embed/e2e |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    body = [
        (
            f"| {r.embedder} | {r.n} | {r.embed_ms:.3f} | {r.prefilter_ms:.3f} | "
            f"{r.plan_ms:.3f} | {r.e2e_ms:.3f} | {100 * r.plan_share_of_e2e:.1f}% | "
            f"{100 * r.embed_share_of_e2e:.1f}% |"
        )
        for r in rows
    ]
    return "\n".join([header, sep, *body])


def h5_gate_from_rows(rows: list[RetrieveBreakdown], *, threshold: float = 0.15) -> str:
    """Return PASS (open H5 PR) or FAIL (decline H5) from SBERT e2e shares."""
    sbert_rows = [r for r in rows if r.embedder.startswith("sbert")]
    if not sbert_rows:
        return "INCONCLUSIVE (no SBERT rows)"
    max_share = max(r.plan_share_of_e2e for r in sbert_rows)
    if max_share < threshold:
        return (
            f"DECLINE H5 (max AFT plan/e2e under SBERT = {100 * max_share:.1f}% "
            f"< {100 * threshold:.0f}% gate)"
        )
    return (
        f"CONSIDER H5 (max AFT plan/e2e under SBERT = {100 * max_share:.1f}% "
        f">= {100 * threshold:.0f}% gate)"
    )


# ---------------------------------------------------------------------------
# H13 — dual-path encode (LLM, opt-in)
# ---------------------------------------------------------------------------


def run_h13(*, n_items: int = 8) -> str:
    key = os.environ.get("EMOTIONAL_MEMORY_LLM_API_KEY", "").strip()
    if not key:
        return "SKIP H13: EMOTIONAL_MEMORY_LLM_API_KEY not set"

    from emotional_memory.appraisal_llm import LLMAppraisalEngine
    from emotional_memory.llm_http import OpenAICompatibleLLMConfig, make_httpx_llm

    base = os.environ.get("EMOTIONAL_MEMORY_LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("EMOTIONAL_MEMORY_LLM_MODEL", "gpt-4o-mini")
    llm = make_httpx_llm(OpenAICompatibleLLMConfig(api_key=key, base_url=base, model=model))
    appraisal = LLMAppraisalEngine(llm)
    embedder = ScalableEmbedder(dim=64)
    texts = [f"Encode latency sample item {i}: mixed emotional content." for i in range(n_items)]

    def _encode_all(dual: bool) -> float:
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=embedder,
            appraisal_engine=appraisal,
            config=EmotionalMemoryConfig(dual_path_encoding=dual),
        )
        t0 = time.perf_counter()
        for t in texts:
            em.encode(t)
        return (time.perf_counter() - t0) * 1000.0

    sync_ms = _encode_all(False)
    dual_ms = _encode_all(True)

    em_dual = EmotionalMemory(
        store=InMemoryStore(),
        embedder=embedder,
        appraisal_engine=appraisal,
        config=EmotionalMemoryConfig(dual_path_encoding=True),
    )
    for t in texts:
        em_dual.encode(t)
    t0 = time.perf_counter()
    em_dual.elaborate_pending()
    elaborate_ms = (time.perf_counter() - t0) * 1000.0

    per_sync = sync_ms / n_items
    per_dual = dual_ms / n_items
    per_elab = elaborate_ms / n_items
    ratio = dual_ms / sync_ms if sync_ms > 0 else float("nan")
    return (
        f"H13 n={n_items} model={model}\n"
        f"  sync encode total:     {sync_ms:.0f} ms ({per_sync:.0f} ms/item)\n"
        f"  dual-path encode total:{dual_ms:.0f} ms ({per_dual:.0f} ms/item)  "
        f"hot-path ratio vs sync: {ratio:.2f}\n"
        f"  elaborate_pending:     {elaborate_ms:.0f} ms ({per_elab:.0f} ms/item)\n"
        f"  dual hot+elaborate:    {dual_ms + elaborate_ms:.0f} ms "
        f"(≈ full work deferred, not free)"
    )


# ---------------------------------------------------------------------------
# pytest entry (runs H12; H13 only with key)
# ---------------------------------------------------------------------------


def test_h12_profile_breakdown() -> None:
    """Measure-only: print H12 table and assert structure (not latency gates)."""
    rows = run_h12(sizes=(100, 1_000), rounds=15)
    table = format_breakdown_table(rows)
    gate = h5_gate_from_rows(rows)
    print("\n" + table)
    print(gate)
    assert any(r.embedder == "hash-64" for r in rows)
    assert all(r.e2e_ms > 0 for r in rows)
    hash_1k = next(r for r in rows if r.embedder == "hash-64" and r.n == 1_000)
    assert hash_1k.plan_ms > 0


@pytest.mark.skipif(
    not os.environ.get("EMOTIONAL_MEMORY_LLM_API_KEY", "").strip(),
    reason="EMOTIONAL_MEMORY_LLM_API_KEY not set",
)
def test_h13_dual_path_encode_latency() -> None:
    report = run_h13(n_items=5)
    print("\n" + report)
    assert "sync encode" in report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm-encode",
        action="store_true",
        help="Also run H13 dual-path vs sync encode (requires LLM API key)",
    )
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000])
    args = parser.parse_args(argv)

    print("=== H12 retrieve breakdown (median ms) ===")
    rows = run_h12(sizes=tuple(args.sizes), rounds=args.rounds)
    print(format_breakdown_table(rows))
    print(h5_gate_from_rows(rows))

    if args.llm_encode:
        print("\n=== H13 dual-path encode ===")
        print(run_h13())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
