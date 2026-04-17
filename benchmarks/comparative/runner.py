"""Comparative benchmark runner.

Measures recall@k and latency for AFT vs baselines on affect_reference_v1.jsonl.

Usage::

    python -m benchmarks.comparative.runner
    python -m benchmarks.comparative.runner --systems aft,naive_cosine
    python -m benchmarks.comparative.runner --top-k 5 --out results.csv

The benchmark encodes all 258 examples, then issues mood-congruent queries
(one per Russell quadrant) and measures recall@k: how many of the top-k
results belong to the correct quadrant (same valence sign x arousal level).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from benchmarks.comparative.adapters.aft import AFTAdapter
from benchmarks.comparative.adapters.base import MemoryAdapter
from benchmarks.comparative.adapters.naive_cosine import NaiveCosineAdapter

ROOT = Path(__file__).parent.parent.parent
DATASET = ROOT / "benchmarks" / "datasets" / "affect_reference_v1.jsonl"
DEFAULT_OUT = ROOT / "benchmarks" / "comparative" / "results.csv"

# ---------------------------------------------------------------------------
# Mood-congruent query set — one representative query per quadrant
# ---------------------------------------------------------------------------

QUERIES = [
    {
        "query": "feeling joyful and enthusiastic about new opportunities",
        "valence_min": 0.0,
        "arousal_min": 0.5,
        "label": "Q1_joy",
    },
    {
        "query": "anxious and tense, something bad might happen",
        "valence_max": 0.0,
        "arousal_min": 0.5,
        "label": "Q2_fear",
    },
    {
        "query": "sad, hopeless, and exhausted",
        "valence_max": 0.0,
        "arousal_max": 0.5,
        "label": "Q3_sadness",
    },
    {
        "query": "calm, peaceful, and contentedly at rest",
        "valence_min": 0.0,
        "arousal_max": 0.5,
        "label": "Q4_calm",
    },
]


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
) -> float:
    """Fraction of top-k results that are mood-congruent with the query."""
    if not retrieved:
        return 0.0
    top = retrieved[:k]
    congruent = sum(
        1 for item in top if item.id in id_map and _is_congruent(id_map[item.id], query_spec)
    )
    return congruent / min(k, len(top))


def run_adapter(
    adapter: MemoryAdapter,
    examples: list[dict],  # type: ignore[type-arg]
    top_k: int = 5,
) -> dict:  # type: ignore[type-arg]
    """Run one full benchmark pass for *adapter*. Returns metrics dict."""
    adapter.reset()

    # Encode
    t0 = time.perf_counter()
    id_map = _build_id_map(examples, adapter)
    encode_total = time.perf_counter() - t0
    encode_ms = (encode_total / len(examples)) * 1000

    # Retrieve — one query per quadrant, measure latency
    recall_scores: list[float] = []
    latencies_ms: list[float] = []

    for q in QUERIES:
        t1 = time.perf_counter()
        retrieved = adapter.retrieve(q["query"], top_k=top_k)
        latency = (time.perf_counter() - t1) * 1000
        latencies_ms.append(latency)
        recall_scores.append(_recall_at_k(retrieved, id_map, q, top_k))

    lat_sorted = sorted(latencies_ms)
    p50 = statistics.median(lat_sorted)
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)] if len(lat_sorted) > 1 else lat_sorted[0]

    return {
        "system": adapter.name,
        f"recall@{top_k}": round(statistics.mean(recall_scores), 4),
        "encode_ms_avg": round(encode_ms, 2),
        "retrieve_p50_ms": round(p50, 2),
        "retrieve_p95_ms": round(p95, 2),
        "n_examples": len(examples),
        "top_k": top_k,
        "status": "ok",
        "reason": "",
    }


def _make_adapters(system_names: list[str]) -> list[MemoryAdapter]:
    registry: dict[str, type[MemoryAdapter]] = {
        "aft": AFTAdapter,
        "naive_cosine": NaiveCosineAdapter,
    }
    # Attempt optional adapters
    try:
        from benchmarks.comparative.adapters.mem0_adapter import Mem0Adapter

        registry["mem0"] = Mem0Adapter
    except ImportError:
        pass

    adapters: list[MemoryAdapter] = []
    for name in system_names:
        cls = registry.get(name)
        if cls is None:
            print(f"[WARN] Unknown system '{name}' — skipping")
            continue
        adapters.append(cls())
    return adapters


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparative memory benchmark")
    parser.add_argument(
        "--systems",
        default="aft,naive_cosine",
        help="Comma-separated list of systems (default: aft,naive_cosine)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    examples = _load_dataset()
    print(f"Dataset: {len(examples)} examples")

    system_names = [s.strip() for s in args.systems.split(",")]
    adapters = _make_adapters(system_names)

    rows: list[dict] = []  # type: ignore[type-arg]

    for adapter in adapters:
        # Check if adapter has an availability flag (optional adapters)
        if hasattr(adapter, "available") and not adapter.available:  # type: ignore[union-attr]
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
            metrics = run_adapter(adapter, examples, top_k=args.top_k)
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

    # Write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written → {out_path}")

        # Also write a Markdown table
        md_path = out_path.with_suffix(".md")
        lines = [
            "# Comparative benchmark results",
            "",
            "| System | Recall@k | Encode ms/item | Retrieve p50 ms | Retrieve p95 ms | Status |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in rows:
            k = r["top_k"]
            lines.append(
                f"| {r['system']} | {r.get(f'recall@{k}', '—')} "
                f"| {r['encode_ms_avg']} | {r['retrieve_p50_ms']} "
                f"| {r['retrieve_p95_ms']} | {r['status']} |"
            )
        md_path.write_text("\n".join(lines) + "\n")
        print(f"Markdown  written → {md_path}")


if __name__ == "__main__":
    main()
