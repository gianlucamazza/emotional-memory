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
# H13 — dual-path encode (LLM live or simulated appraisal cost)
# ---------------------------------------------------------------------------


class _DelayedKeywordAppraisal:
    """AppraisalEngine that sleeps then delegates to KeywordAppraisalEngine.

    Used for offline H13 structural measurement without a live LLM key:
    proves dual-path skips ``appraise()`` on the encode hot path.
    """

    def __init__(self, delay_s: float = 0.15) -> None:
        from emotional_memory.appraisal_llm import KeywordAppraisalEngine

        self._inner = KeywordAppraisalEngine()
        self._delay_s = delay_s
        self.calls = 0

    def appraise(self, event_text: str, context: dict[str, Any] | None = None) -> Any:
        self.calls += 1
        time.sleep(self._delay_s)
        return self._inner.appraise(event_text, context=context)


def _h13_texts(prefix: str, n_items: int) -> list[str]:
    return [
        f"{prefix} item {i}: I felt proud after shipping the feature early."
        for i in range(n_items)
    ]


def _h13_format_report(
    *,
    label: str,
    n_items: int,
    sync_ms: float,
    dual_ms: float,
    elaborate_ms: float,
    dual_only_ms: float,
    n_elaborated: int,
    sync_calls: int | None = None,
    dual_calls: int | None = None,
    elab_calls: int | None = None,
    extra: str = "",
) -> str:
    per_sync = sync_ms / n_items
    per_dual = dual_ms / n_items
    per_elab = elaborate_ms / max(n_elaborated, 1)
    ratio = dual_ms / sync_ms if sync_ms > 0 else float("nan")
    full_dual = dual_only_ms + elaborate_ms
    combined_ratio = full_dual / sync_ms if sync_ms > 0 else float("nan")

    def _calls(n: int | None) -> str:
        return f" [appraise calls={n}]" if n is not None else ""

    return (
        f"H13 {label} n={n_items}\n"
        f"  sync encode total:      {sync_ms:.0f} ms ({per_sync:.0f} ms/item)"
        f"{_calls(sync_calls)}\n"
        f"  dual-path encode total: {dual_ms:.0f} ms ({per_dual:.0f} ms/item)"
        f"{_calls(dual_calls)}  hot-path ratio vs sync: {ratio:.2f}\n"
        f"  elaborate_pending:      {elaborate_ms:.0f} ms "
        f"({per_elab:.0f} ms/item, n={n_elaborated})"
        f"{_calls(elab_calls)}\n"
        f"  dual hot+elaborate:     {full_dual:.0f} ms "
        f"({full_dual / n_items:.0f} ms/item combined; combined/sync={combined_ratio:.2f})\n"
        f"  verdict: dual hot-path is {ratio:.2f}x sync; work is deferred not free"
        f"{extra}"
    )


def run_h13_sim(*, n_items: int = 5, delay_s: float = 0.15) -> str:
    """Offline H13: artificial appraisal delay, no network.

    Acceptance (structural):
    - dual encode makes 0 ``appraise`` calls
    - sync encode makes ``n_items`` calls
    - dual hot-path wall time << sync
    - elaborate_pending pays ~n_items * delay
    """
    embedder = ScalableEmbedder(dim=64)

    def _run(dual: bool, texts: list[str], appraisal: _DelayedKeywordAppraisal) -> float:
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=embedder,
            appraisal_engine=appraisal,
            config=EmotionalMemoryConfig(dual_path_encoding=dual, enable_resonance=False),
        )
        t0 = time.perf_counter()
        for t in texts:
            em.encode(t)
        return (time.perf_counter() - t0) * 1000.0

    sync_app = _DelayedKeywordAppraisal(delay_s)
    dual_app = _DelayedKeywordAppraisal(delay_s)
    elab_app = _DelayedKeywordAppraisal(delay_s)

    sync_ms = _run(False, _h13_texts("H13-sim-sync", n_items), sync_app)
    dual_ms = _run(True, _h13_texts("H13-sim-dual", n_items), dual_app)

    em_dual = EmotionalMemory(
        store=InMemoryStore(),
        embedder=embedder,
        appraisal_engine=elab_app,
        config=EmotionalMemoryConfig(dual_path_encoding=True, enable_resonance=False),
    )
    elab_texts = _h13_texts("H13-sim-elab", n_items)
    t0 = time.perf_counter()
    for t in elab_texts:
        em_dual.encode(t)
    dual_only_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    elaborated = em_dual.elaborate_pending()
    elaborate_ms = (time.perf_counter() - t0) * 1000.0

    return _h13_format_report(
        label=f"sim delay={delay_s * 1000:.0f}ms/appraise",
        n_items=n_items,
        sync_ms=sync_ms,
        dual_ms=dual_ms,
        elaborate_ms=elaborate_ms,
        dual_only_ms=dual_only_ms,
        n_elaborated=len(elaborated),
        sync_calls=sync_app.calls,
        dual_calls=dual_app.calls,
        elab_calls=elab_app.calls,
        extra=(
            f"\n  structural OK: dual_calls={dual_app.calls} "
            f"(expect 0), sync_calls={sync_app.calls} (expect {n_items})"
        ),
    )


def run_h13(*, n_items: int = 5) -> str:
    """Measure dual-path vs sync encode with a real LLM appraisal engine.

    Fairness rules:
    - ``cache_size=0`` so LRU does not make a later arm free
    - unique text per arm
    - separate ``LLMAppraisalEngine`` per arm
    - **INCONCLUSIVE** if any arm records ``fallback_count > 0`` (auth/network/parse)
    """
    key = os.environ.get("EMOTIONAL_MEMORY_LLM_API_KEY", "").strip()
    if not key:
        return "SKIP H13 live: EMOTIONAL_MEMORY_LLM_API_KEY not set (use --llm-encode-sim)"

    from emotional_memory.appraisal_llm import LLMAppraisalConfig, LLMAppraisalEngine
    from emotional_memory.llm_http import OpenAICompatibleLLMConfig, make_httpx_llm

    cfg = OpenAICompatibleLLMConfig.from_env()
    if cfg is None:
        return "SKIP H13 live: OpenAICompatibleLLMConfig.from_env() returned None"
    model = cfg.model
    embedder = ScalableEmbedder(dim=64)

    def _fresh() -> LLMAppraisalEngine:
        return LLMAppraisalEngine(
            make_httpx_llm(cfg),
            config=LLMAppraisalConfig(cache_size=0),
        )

    def _encode_all(dual: bool, texts: list[str], appraisal: LLMAppraisalEngine) -> float:
        em = EmotionalMemory(
            store=InMemoryStore(),
            embedder=embedder,
            appraisal_engine=appraisal,
            config=EmotionalMemoryConfig(dual_path_encoding=dual, enable_resonance=False),
        )
        t0 = time.perf_counter()
        for t in texts:
            em.encode(t)
        return (time.perf_counter() - t0) * 1000.0

    sync_eng = _fresh()
    dual_eng = _fresh()
    elab_eng = _fresh()

    sync_ms = _encode_all(False, _h13_texts("H13-sync", n_items), sync_eng)
    dual_ms = _encode_all(True, _h13_texts("H13-dual", n_items), dual_eng)

    em_dual = EmotionalMemory(
        store=InMemoryStore(),
        embedder=embedder,
        appraisal_engine=elab_eng,
        config=EmotionalMemoryConfig(dual_path_encoding=True, enable_resonance=False),
    )
    elab_texts = _h13_texts("H13-elab", n_items)
    t0 = time.perf_counter()
    for t in elab_texts:
        em_dual.encode(t)
    dual_only_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    elaborated = em_dual.elaborate_pending()
    elaborate_ms = (time.perf_counter() - t0) * 1000.0

    fb = sync_eng.fallback_count + dual_eng.fallback_count + elab_eng.fallback_count
    report = _h13_format_report(
        label=f"live model={model} cache_size=0",
        n_items=n_items,
        sync_ms=sync_ms,
        dual_ms=dual_ms,
        elaborate_ms=elaborate_ms,
        dual_only_ms=dual_only_ms,
        n_elaborated=len(elaborated),
        extra=f"\n  fallback_count total={fb}",
    )
    if fb > 0:
        return (
            f"H13 live INCONCLUSIVE: {fb} appraisal fallback(s) "
            f"(invalid key, network, or parse errors). Do not treat ms as LLM cost.\n"
            f"{report}\n"
            f"  tip: fix EMOTIONAL_MEMORY_LLM_API_KEY or use --llm-encode-sim"
        )
    return report


# ---------------------------------------------------------------------------
# pytest entry
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


def test_h13_sim_dual_path_skips_appraisal() -> None:
    """Structural H13: dual encode does not call appraise; sync does."""
    report = run_h13_sim(n_items=3, delay_s=0.05)
    print("\n" + report)
    assert "dual_calls=0" in report
    assert "sync_calls=3" in report
    # Parse hot-path ratio from "hot-path ratio vs sync: 0.XX"
    assert "hot-path ratio vs sync:" in report


@pytest.mark.skipif(
    not os.environ.get("EMOTIONAL_MEMORY_LLM_API_KEY", "").strip(),
    reason="EMOTIONAL_MEMORY_LLM_API_KEY not set",
)
def test_h13_dual_path_encode_latency() -> None:
    report = run_h13(n_items=3)
    print("\n" + report)
    # May be INCONCLUSIVE if key is bad — still a valid harness outcome
    assert "H13" in report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm-encode",
        action="store_true",
        help="H13 live dual-path vs sync (requires valid LLM API key)",
    )
    parser.add_argument(
        "--llm-encode-sim",
        action="store_true",
        help="H13 offline with delayed KeywordAppraisal (no network)",
    )
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--h13-n", type=int, default=5, help="Items per H13 arm")
    args = parser.parse_args(argv)

    print("=== H12 retrieve breakdown (median ms) ===")
    rows = run_h12(sizes=tuple(args.sizes), rounds=args.rounds)
    print(format_breakdown_table(rows))
    print(h5_gate_from_rows(rows))

    if args.llm_encode_sim:
        print("\n=== H13 dual-path encode (simulated appraisal delay) ===")
        print(run_h13_sim(n_items=args.h13_n))
    if args.llm_encode:
        print("\n=== H13 dual-path encode (live LLM) ===")
        print(run_h13(n_items=args.h13_n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
