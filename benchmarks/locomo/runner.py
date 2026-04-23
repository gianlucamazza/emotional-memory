"""LoCoMo benchmark runner.

Usage::

    uv run python -m benchmarks.locomo.runner --systems aft naive_rag
    uv run python -m benchmarks.locomo.runner --limit-conversations 2 --limit-qa 10  # dry run

Environment variables (same as LLM test suite):
  EMOTIONAL_MEMORY_LLM_API_KEY   required for answer generation + judge
  EMOTIONAL_MEMORY_LLM_BASE_URL  default https://api.openai.com/v1
  EMOTIONAL_MEMORY_LLM_MODEL     default gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.locomo.adapters.aft import AFTLoCoMoAdapter
from benchmarks.locomo.adapters.base import LoCoMoAdapter, call_llm
from benchmarks.locomo.adapters.naive_rag import NaiveRAGLoCoMoAdapter
from benchmarks.locomo.dataset import LoCoMoDataset, load_dataset
from benchmarks.locomo.scoring import (
    build_judge_prompt,
    is_adversarial_correct,
    parse_judge_response,
    score_predictions,
)

_HERE = Path(__file__).parent
DEFAULT_OUT_JSON = _HERE / "results.json"
DEFAULT_OUT_MD = _HERE / "results.md"

DEFAULT_SYSTEMS = ["aft", "naive_rag"]


def _make_adapter(name: str) -> LoCoMoAdapter:
    if name == "aft":
        return AFTLoCoMoAdapter()
    if name == "naive_rag":
        return NaiveRAGLoCoMoAdapter()
    raise ValueError(f"Unknown system: {name!r}. Choices: aft, naive_rag")


def _run_judge(predictions: list[dict[str, Any]], *, verbose: bool = False) -> None:
    """Augment predictions in-place with 'judge_correct' field."""
    for i, pred in enumerate(predictions):
        if pred.get("is_adversarial"):
            pred["judge_correct"] = is_adversarial_correct(pred["prediction"])
            continue
        prompt = build_judge_prompt(pred["question"], pred["gold"], pred["prediction"])
        try:
            response = call_llm(prompt, temperature=0.0)
            pred["judge_correct"] = parse_judge_response(response)
        except Exception as exc:
            if verbose:
                print(f"  Judge error on item {i}: {exc}")
            pred["judge_correct"] = None


def run_benchmark(
    dataset: LoCoMoDataset,
    *,
    systems: list[str] = DEFAULT_SYSTEMS,
    run_judge: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all *systems* on every conversation in *dataset*.

    Returns a results dict with per-system predictions + aggregate scores.
    """
    system_results: list[dict[str, Any]] = []

    for system_name in systems:
        adapter = _make_adapter(system_name)
        all_predictions: list[dict[str, Any]] = []

        for conv in dataset.conversations:
            if verbose:
                print(f"  [{system_name}] {conv.sample_id} — {len(conv.qa_pairs)} QA pairs …")
            preds = adapter.run_conversation(conv)
            all_predictions.extend(preds)

        if run_judge:
            if verbose:
                print(f"  [{system_name}] Running LLM judge on {len(all_predictions)} items …")
            _run_judge(all_predictions, verbose=verbose)

        scores = score_predictions(all_predictions)
        system_results.append(
            {
                "system": system_name,
                "n_predictions": len(all_predictions),
                "scores": scores,
                "predictions": all_predictions,
            }
        )

    return {
        "benchmark": "locomo_v1",
        "dataset": "locomo10",
        "n_conversations": len(dataset.conversations),
        "n_qa_total": dataset.total_qa,
        "systems": system_results,
    }


def _render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# LoCoMo Benchmark Results",
        "",
        f"Dataset: {results['dataset']} "
        f"({results['n_conversations']} conversations, {results['n_qa_total']} QA pairs)",
        "",
        "## Aggregate Scores",
        "",
        "| System | N | F1 | BLEU-1 | Judge Acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for sys in results["systems"]:
        agg = sys["scores"]["aggregate"]
        judge = f"{agg['judge_accuracy']:.3f}" if "judge_accuracy" in agg else "—"
        lines.append(
            f"| `{sys['system']}` | {agg['n']} | {agg['f1']:.3f} | {agg['bleu1']:.3f} | {judge} |"
        )
    lines += ["", "## By Category", ""]

    # collect all category names across all systems
    all_cats: set[str] = set()
    for sys in results["systems"]:
        all_cats.update(sys["scores"]["by_category"].keys())

    for cat in sorted(all_cats):
        lines += [
            f"### {cat}",
            "",
            "| System | N | F1 | BLEU-1 | Judge Acc |",
            "|---|---:|---:|---:|---:|",
        ]
        for sys in results["systems"]:
            cat_scores = sys["scores"]["by_category"].get(cat)
            if cat_scores is None:
                continue
            judge = (
                f"{cat_scores['judge_accuracy']:.3f}" if "judge_accuracy" in cat_scores else "—"
            )
            lines.append(
                f"| `{sys['system']}` | {cat_scores['n']} "
                f"| {cat_scores['f1']:.3f} | {cat_scores['bleu1']:.3f} | {judge} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_results(
    results: dict[str, Any],
    *,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # strip raw predictions from JSON to keep file size manageable
    slim = {k: v for k, v in results.items() if k != "systems"}
    slim["systems"] = [
        {k: v for k, v in sys.items() if k != "predictions"} for sys in results["systems"]
    ]
    out_json.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(results), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LoCoMo benchmark.")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=DEFAULT_SYSTEMS,
        choices=["aft", "naive_rag"],
        metavar="SYSTEM",
        help="Systems to evaluate (default: aft naive_rag).",
    )
    parser.add_argument(
        "--limit-conversations",
        type=int,
        default=None,
        metavar="N",
        help="Only load the first N conversations (cost control).",
    )
    parser.add_argument(
        "--limit-qa",
        type=int,
        default=None,
        metavar="N",
        help="Cap QA pairs per conversation (cost control).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM-as-judge step (compute F1/BLEU only).",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    print("Loading LoCoMo dataset …")
    dataset = load_dataset(
        limit_conversations=args.limit_conversations,
        limit_qa_per_conversation=args.limit_qa,
    )
    print(f"Loaded {len(dataset.conversations)} conversations, {dataset.total_qa} QA pairs.")

    results = run_benchmark(
        dataset,
        systems=args.systems,
        run_judge=not args.no_judge,
        verbose=True,
    )

    write_results(results, out_json=args.out_json, out_md=args.out_md)
    print(f"\nResults written to {args.out_json} and {args.out_md}")

    print("\n=== Aggregate Scores ===")
    for sys in results["systems"]:
        agg = sys["scores"]["aggregate"]
        judge_str = f"  judge={agg['judge_accuracy']:.3f}" if "judge_accuracy" in agg else ""
        print(f"  {sys['system']:12s}  F1={agg['f1']:.3f}  BLEU-1={agg['bleu1']:.3f}{judge_str}")


if __name__ == "__main__":
    main()
