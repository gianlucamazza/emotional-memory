"""Comparative benchmark runner.

Measures recall@k and latency for AFT vs baselines on affect_reference_v1.jsonl.

Usage::

    python -m benchmarks.comparative.runner
    python -m benchmarks.comparative.runner --systems aft,naive_cosine,recency
    python -m benchmarks.comparative.runner --embedder sbert --top-k 5 --out results.csv

The benchmark encodes all 258 examples, then issues mood-congruent queries
(one per Russell quadrant) and measures recall@k: how many of the top-k
results belong to the correct quadrant (same valence sign x arousal level).

For affect-aware adapters (AFT) the query's affective centroid is passed to
``retrieve()`` so the mood-congruent scorer operates on the correct quadrant.

This is a controlled synthetic retrieval probe. It is useful for testing
affect-aware ranking behavior, but it does not establish general downstream
superiority across production memory systems.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

# Bridge project-specific key → standard OpenAI env var expected by mem0/langmem.
if "OPENAI_API_KEY" not in os.environ and "EMOTIONAL_MEMORY_LLM_API_KEY" in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["EMOTIONAL_MEMORY_LLM_API_KEY"]

from benchmarks.common.statistics import (
    DEFAULT_N_BOOTSTRAP,
    bootstrap_ci,
    ci_payload,
    format_point_ci,
    mcnemar_exact,
    paired_bootstrap_diff,
)
from benchmarks.comparative.adapters.aft import AFTAdapter
from benchmarks.comparative.adapters.base import MemoryAdapter
from benchmarks.comparative.adapters.naive_cosine import NaiveCosineAdapter
from benchmarks.comparative.adapters.recency import RecencyAdapter

ROOT = Path(__file__).parent.parent.parent
DATASET = ROOT / "benchmarks" / "datasets" / "affect_reference_v1.jsonl"
DEFAULT_OUT = ROOT / "benchmarks" / "comparative" / "results.csv"

# ---------------------------------------------------------------------------
# Mood-congruent query set — one representative query per quadrant.
# query_valence / query_arousal are the centroid of each Russell quadrant,
# passed to affect-aware adapters so mood-congruent scoring is active.
# ---------------------------------------------------------------------------

QUERIES: list[dict[str, Any]] = [
    {
        "query": "feeling joyful and enthusiastic about new opportunities",
        "valence_min": 0.0,
        "arousal_min": 0.5,
        "query_valence": 0.6,
        "query_arousal": 0.75,
        "label": "Q1_joy",
    },
    {
        "query": "anxious and tense, something bad might happen",
        "valence_max": 0.0,
        "arousal_min": 0.5,
        "query_valence": -0.6,
        "query_arousal": 0.75,
        "label": "Q2_fear",
    },
    {
        "query": "sad, hopeless, and exhausted",
        "valence_max": 0.0,
        "arousal_max": 0.5,
        "query_valence": -0.6,
        "query_arousal": 0.25,
        "label": "Q3_sadness",
    },
    {
        "query": "calm, peaceful, and contentedly at rest",
        "valence_min": 0.0,
        "arousal_max": 0.5,
        "query_valence": 0.5,
        "query_arousal": 0.25,
        "label": "Q4_calm",
    },
]


def build_protocol_metadata(
    *,
    system_names: list[str],
    top_k: int,
    embedder_type: str,
    n_examples: int,
) -> dict[str, Any]:
    return {
        "benchmark": "comparative_affect_reference_v1",
        "question": (
            "Does the system surface memories from the intended affective region "
            "under a controlled query-state setup?"
        ),
        "dataset": {
            "name": "affect_reference_v1",
            "path": str(DATASET.relative_to(ROOT)),
            "examples": n_examples,
            "type": "synthetic affect-labeled retrieval probe",
        },
        "query_labels": [q["label"] for q in QUERIES],
        "top_k": top_k,
        "embedder": embedder_type,
        "systems": system_names,
        "primary_metric": "recall@k quadrant-level affect congruence",
        "secondary_metrics": ["encode_ms_avg", "retrieve_p50_ms", "retrieve_p95_ms"],
        "affect_aware_adapters_receive_query_affect": ["aft"],
        "interpretation_guardrails": [
            "This is a controlled synthetic benchmark, not a general downstream evaluation.",
            "AFT receives explicit query affect (valence, arousal) in this protocol.",
            "General-purpose systems may ignore query-affect fields entirely.",
            "The results do not establish production superiority or human-like emotional memory.",
        ],
    }


def _is_congruent(ex: dict, q: dict) -> bool:  # type: ignore[type-arg]
    v, a = ex["valence"], ex["arousal"]
    if "valence_min" in q and v < q["valence_min"]:
        return False
    if "valence_max" in q and v >= q["valence_max"]:
        return False
    if "arousal_min" in q and a < q["arousal_min"]:
        return False
    return not ("arousal_max" in q and a >= q["arousal_max"])


def _load_dataset() -> list[dict]:  # type: ignore[type-arg]
    return [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]


def _build_id_map(examples: list[dict], adapter: MemoryAdapter) -> dict[str, dict]:  # type: ignore[type-arg]
    """Encode all examples and return {assigned_id: example}."""
    id_map: dict[str, dict] = {}  # type: ignore[type-arg]
    for ex in examples:
        assigned = adapter.encode(ex["text"], valence=ex["valence"], arousal=ex["arousal"])
        id_map[assigned] = ex
    return id_map


def _recall_at_k(
    retrieved: list,  # type: ignore[type-arg]
    id_map: dict,  # type: ignore[type-arg]
    query_spec: dict,  # type: ignore[type-arg]
    k: int,
) -> list[bool]:
    """Per-item congruence flags for the top-k results (True = mood-congruent)."""
    if not retrieved:
        return []
    top = retrieved[:k]
    return [item.id in id_map and _is_congruent(id_map[item.id], query_spec) for item in top]


def run_adapter(
    adapter: MemoryAdapter,
    examples: list[dict],  # type: ignore[type-arg]
    top_k: int = 5,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
) -> dict:  # type: ignore[type-arg]
    """Run one full benchmark pass for *adapter*. Returns metrics dict.

    ``_item_flags`` (list[bool]) is included for pairwise comparison in the
    caller; strip it before writing to CSV.
    """
    adapter.reset()

    # Encode
    t0 = time.perf_counter()
    id_map = _build_id_map(examples, adapter)
    encode_total = time.perf_counter() - t0
    encode_ms = (encode_total / len(examples)) * 1000

    # Retrieve — one query per quadrant, pass affective centroid for mood-aware adapters
    all_item_flags: list[bool] = []
    per_query_scores: list[float] = []
    latencies_ms: list[float] = []

    for q in QUERIES:
        t1 = time.perf_counter()
        retrieved = adapter.retrieve(
            q["query"],
            top_k=top_k,
            valence=q["query_valence"],
            arousal=q["query_arousal"],
        )
        latency = (time.perf_counter() - t1) * 1000
        latencies_ms.append(latency)
        flags = _recall_at_k(retrieved, id_map, q, top_k)
        all_item_flags.extend(flags)
        per_query_scores.append(sum(flags) / max(len(flags), 1) if flags else 0.0)

    lat_sorted = sorted(latencies_ms)
    p50 = statistics.median(lat_sorted)
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)] if len(lat_sorted) > 1 else lat_sorted[0]

    item_floats = [float(f) for f in all_item_flags]
    ci = ci_payload(
        *bootstrap_ci(item_floats, n_bootstrap=n_bootstrap, seed=seed),
        n_bootstrap=n_bootstrap,
    )
    return {
        "system": adapter.name,
        f"recall@{top_k}": round(statistics.mean(per_query_scores), 4),
        "encode_ms_avg": round(encode_ms, 2),
        "retrieve_p50_ms": round(p50, 2),
        "retrieve_p95_ms": round(p95, 2),
        "n_examples": len(examples),
        "top_k": top_k,
        "status": "ok",
        "reason": "",
        "ci": ci,
        "ci_note": f"Item-level CI over top_k * n_queries items (N={len(QUERIES)} queries)",
        "_item_flags": all_item_flags,
    }


def _build_embedder(embedder_type: str) -> Any:
    if embedder_type == "sbert":
        try:
            from emotional_memory.embedders import SentenceTransformerEmbedder

            print("  embedder: SentenceTransformer (all-MiniLM-L6-v2)")
            return SentenceTransformerEmbedder()
        except ImportError as exc:
            print(f"[WARN] sentence-transformers not installed ({exc}); falling back to hash")
    return None  # triggers _HashEmbedder inside adapters


def _make_adapters(system_names: list[str], embedder: Any = None) -> list[MemoryAdapter]:
    registry: dict[str, Any] = {
        "aft": lambda: AFTAdapter(embedder=embedder),
        "naive_cosine": lambda: NaiveCosineAdapter(embedder=embedder),
        "recency": RecencyAdapter,
    }
    # Attempt optional adapters
    try:
        from benchmarks.comparative.adapters.mem0_adapter import Mem0Adapter

        registry["mem0"] = Mem0Adapter
    except ImportError:
        pass

    try:
        from benchmarks.comparative.adapters.letta_adapter import LettaAdapter

        registry["letta"] = LettaAdapter
    except ImportError:
        pass

    try:
        from benchmarks.comparative.adapters.langmem_adapter import LangMemAdapter

        registry["langmem"] = LangMemAdapter
    except ImportError:
        pass

    adapters: list[MemoryAdapter] = []
    for name in system_names:
        factory = registry.get(name)
        if factory is None:
            print(f"[WARN] Unknown system '{name}' — skipping")
            continue
        adapters.append(factory())
    return adapters


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparative memory benchmark")
    parser.add_argument(
        "--systems",
        default="aft,naive_cosine,recency",
        help="Comma-separated list of systems (default: aft,naive_cosine,recency)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--embedder",
        default="hash",
        choices=["hash", "sbert"],
        help=(
            "Embedding backend: 'hash' (deterministic SHA-256) or 'sbert' (sentence-transformers)"
        ),
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for CI bootstrap.")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help="Number of bootstrap resamples for CI computation.",
    )
    args = parser.parse_args()

    examples = _load_dataset()
    print(f"Dataset: {len(examples)} examples")
    print(f"Embedder: {args.embedder}")

    embedder = _build_embedder(args.embedder)
    system_names = [s.strip() for s in args.systems.split(",")]
    adapters = _make_adapters(system_names, embedder=embedder)

    rows: list[dict] = []  # type: ignore[type-arg]

    for adapter in adapters:
        # Check if adapter has an availability flag (optional adapters)
        if hasattr(adapter, "available") and not adapter.available:
            reason = getattr(adapter, "not_available_reason", "not installed")
            print(f"[SKIP] {adapter.name}: {reason}")
            rows.append(
                {
                    "system": adapter.name,
                    f"recall@{args.top_k}": "—",
                    "encode_ms_avg": "—",
                    "retrieve_p50_ms": "—",
                    "retrieve_p95_ms": "—",
                    "n_examples": 0,
                    "top_k": args.top_k,
                    "status": "not_evaluated",
                    "reason": reason,
                }
            )
            continue

        print(f"Running {adapter.name}...")
        try:
            metrics = run_adapter(
                adapter, examples, top_k=args.top_k, n_bootstrap=args.n_bootstrap, seed=args.seed
            )
            rows.append(metrics)
            print(
                f"  recall@{args.top_k}={metrics[f'recall@{args.top_k}']}"
                f"  p50={metrics['retrieve_p50_ms']}ms"
                f"  p95={metrics['retrieve_p95_ms']}ms"
            )
        except Exception as exc:
            print(f"[ERROR] {adapter.name}: {exc}")
            rows.append(
                {
                    "system": adapter.name,
                    f"recall@{args.top_k}": "—",
                    "encode_ms_avg": "—",
                    "retrieve_p50_ms": "—",
                    "retrieve_p95_ms": "—",
                    "n_examples": 0,
                    "top_k": args.top_k,
                    "status": "error",
                    "reason": str(exc),
                }
            )

    # Compute pairwise comparisons vs naive_cosine (only rows with _item_flags present)
    pairwise_rows: list[dict[str, Any]] = []
    baseline_flags: list[bool] | None = None
    for r in rows:
        if r.get("system") == "naive_cosine" and "_item_flags" in r:
            baseline_flags = r["_item_flags"]
            break
    if baseline_flags is not None:
        for r in rows:
            if r.get("system") == "naive_cosine" or "_item_flags" not in r:
                continue
            sys_flags = r["_item_flags"]
            n_pad = max(0, len(baseline_flags) - len(sys_flags))
            a_padded = sys_flags + [False] * n_pad
            b_padded = baseline_flags + [False] * max(0, len(sys_flags) - len(baseline_flags))
            n = min(len(a_padded), len(b_padded))
            a_f = [float(v) for v in a_padded[:n]]
            b_f = [float(v) for v in b_padded[:n]]
            diff, lo, hi, p_boot = paired_bootstrap_diff(
                a_f, b_f, n_bootstrap=args.n_bootstrap, seed=args.seed
            )
            only_a = sum(1 for av, bv in zip(a_f, b_f, strict=True) if av > bv)
            only_b = sum(1 for av, bv in zip(a_f, b_f, strict=True) if bv > av)
            p_mc = mcnemar_exact(only_a, only_b)
            pairwise_rows.append(
                {
                    "system": r["system"],
                    "baseline": "naive_cosine",
                    "diff": round(diff, 4),
                    "ci_lower": round(lo, 4),
                    "ci_upper": round(hi, 4),
                    "p_bootstrap": round(p_boot, 4),
                    "p_mcnemar": round(p_mc, 4),
                    "n_items": n,
                    "n_discordant": only_a + only_b,
                    "n_padded": n_pad,
                }
            )

    # Strip internal keys before CSV
    _internal_keys = {"ci", "ci_note", "_item_flags"}
    csv_rows = [{k: v for k, v in r.items() if k not in _internal_keys} for r in rows]

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        protocol_metadata = build_protocol_metadata(
            system_names=system_names,
            top_k=args.top_k,
            embedder_type=args.embedder,
            n_examples=len(examples),
        )
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nResults written → {out_path}")

        # Markdown table with CI
        ci_note = (
            f"Item-level CI over top_k x N_queries items "
            f"(N={len(QUERIES)} queries x top_k items). "
            f"Bootstrap percentile, n={args.n_bootstrap}, seed={args.seed}."
        )
        md_path = out_path.with_suffix(".md")
        lines = [
            "# Comparative benchmark results",
            "",
            "These numbers come from a **controlled synthetic benchmark** on",
            "`affect_reference_v1`. They measure mood-congruent retrieval behavior under a",
            "small public probe, not general downstream answer quality or production",
            "superiority across memory systems.",
            "",
            "Interpretation guardrails:",
            "",
            "- `Recall@k` here means quadrant-level affect congruence, not QA accuracy.",
            "- AFT receives explicit query affect (`valence`, `arousal`) in this protocol.",
            "- General-purpose systems such as Mem0 and LangMem are being evaluated on a task",
            "  narrower than their intended product surface.",
            f"- CI note: {ci_note}",
            "",
            (
                "| System | Recall@k [95% CI] | Encode ms/item "
                "| Retrieve p50 ms | Retrieve p95 ms | Status |"
            ),
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in rows:
            k = r.get("top_k", args.top_k)
            recall_val = r.get(f"recall@{k}", "—")
            ci_data = r.get("ci")
            if ci_data and isinstance(recall_val, (int, float)):
                recall_str = format_point_ci(
                    ci_data["point"], ci_data["ci_lower"], ci_data["ci_upper"]
                )
            else:
                recall_str = str(recall_val)
            lines.append(
                f"| {r['system']} | {recall_str} "
                f"| {r['encode_ms_avg']} | {r['retrieve_p50_ms']} "
                f"| {r['retrieve_p95_ms']} | {r['status']} |"
            )
        if pairwise_rows:
            lines.extend(
                [
                    "",
                    "## Pairwise vs naive_cosine",
                    "",
                    "Two-sided tests: paired bootstrap p-value and exact McNemar p-value.",
                    "H0: no difference. CI excludes 0 ↔ difference is credible at 95% level.",
                    "",
                    (
                        "| System | Δ [95% CI] | p (bootstrap) | p (McNemar) "
                        "| N items | Discordant | Padded |"
                    ),
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for pr in pairwise_rows:
                diff_str = format_point_ci(pr["diff"], pr["ci_lower"], pr["ci_upper"])
                lines.append(
                    f"| {pr['system']} | {diff_str} | "
                    f"{pr['p_bootstrap']:.4f} | {pr['p_mcnemar']:.4f} | "
                    f"{pr['n_items']} | {pr['n_discordant']} | {pr['n_padded']} |"
                )
        md_path.write_text("\n".join(lines) + "\n")
        print(f"Markdown  written → {md_path}")

        protocol_path = out_path.with_suffix(".protocol.json")
        protocol_path.write_text(json.dumps(protocol_metadata, indent=2) + "\n")
        print(f"Protocol  written → {protocol_path}")


if __name__ == "__main__":
    main()
