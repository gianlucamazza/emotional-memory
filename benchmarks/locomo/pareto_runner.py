"""LoCoMo Pareto sweep runner — Addendum J (frozen protocol).

Implements the stratified subsample (50 QA x 4 categories, seed=42) and the
10-config weight sweep pre-registered in benchmarks/preregistration_addendum_j.md
(commit 7bcd663).  Do NOT modify the WEIGHT_CONFIGS or SAMPLE_SEED constants
after the first run — they are part of the frozen pre-registration.

Usage::

    uv run python -m benchmarks.locomo.pareto_runner           # full sweep
    uv run python -m benchmarks.locomo.pareto_runner --dry-run # 4 QA/cat, 2 configs, no judge

Environment variables: same as benchmarks/locomo/runner.py (see its docstring).
Prefer ``make bench-locomo-pareto`` — it exports .env and sets PYTHONUNBUFFERED=1.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from benchmarks.locomo.adapters.aft import AFTLoCoMoAdapter
from benchmarks.locomo.adapters.naive_rag import NaiveRAGLoCoMoAdapter
from benchmarks.locomo.dataset import (
    QA_CATEGORY_NAMES,
    Conversation,
    LoCoMoDataset,
    load_dataset,
)
from benchmarks.locomo.runner import (
    _append_checkpoint,
    _load_checkpoint,
    _run_judge,
)
from benchmarks.locomo.scoring import score_predictions, token_f1
from emotional_memory import EmotionalMemoryConfig
from emotional_memory.retrieval import RetrievalConfig

_HERE = Path(__file__).parent
DEFAULT_OUT_JSON = _HERE / "pareto_results.json"
DEFAULT_OUT_MD = _HERE / "pareto_results.md"
DEFAULT_CHECKPOINT = _HERE / "pareto_results.checkpoint.jsonl"

# ---------------------------------------------------------------------------
# Frozen pre-registration constants (Add. J §Weight grid + §Sampling protocol)
# ---------------------------------------------------------------------------

SAMPLE_SEED: int = 42
SAMPLE_PER_CATEGORY: int = 50
PARETO_CATEGORIES: tuple[int, ...] = (1, 2, 3, 4)  # excludes adversarial cat-5

# Weight order: [semantic, mood_congruence, affect_proximity, momentum, recency, resonance]
WEIGHT_CONFIGS: dict[str, list[float]] = {
    "W0": [0.35, 0.25, 0.15, 0.10, 0.10, 0.05],  # S1 default (baseline)
    "W1": [0.60, 0.15, 0.10, 0.05, 0.05, 0.05],  # High semantic
    "W2": [0.50, 0.30, 0.10, 0.05, 0.05, 0.00],  # High semantic + mood, no resonance
    "W3": [0.40, 0.35, 0.10, 0.05, 0.05, 0.05],  # Elevated mood
    "W4": [0.35, 0.20, 0.10, 0.05, 0.25, 0.05],  # High recency
    "W5": [0.35, 0.10, 0.10, 0.05, 0.35, 0.05],  # Very high recency (temporal QA)
    "W6": [0.35, 0.05, 0.05, 0.05, 0.45, 0.05],  # Recency-dominant
    "W7": [0.70, 0.10, 0.08, 0.04, 0.04, 0.04],  # Very high semantic
    "W8": [0.50, 0.10, 0.10, 0.10, 0.10, 0.10],  # Uniform non-semantic
    "W9": [0.45, 0.20, 0.12, 0.08, 0.10, 0.05],  # Midpoint S1 + W7
}


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------


def _stratified_subsample(
    dataset: LoCoMoDataset,
    *,
    seed: int = SAMPLE_SEED,
    per_category: int = SAMPLE_PER_CATEGORY,
) -> LoCoMoDataset:
    """Return a new LoCoMoDataset with qa_pairs limited to *per_category* per
    non-adversarial category (globally across all conversations, shuffled with
    *seed*).  Conversation objects with zero sampled QA pairs are dropped.
    """
    # Collect all (conv_idx, qa_idx) per category
    cat_index: dict[int, list[tuple[int, int]]] = {c: [] for c in PARETO_CATEGORIES}
    for ci, conv in enumerate(dataset.conversations):
        for qi, qa in enumerate(conv.qa_pairs):
            if qa.category in PARETO_CATEGORIES:
                cat_index[qa.category].append((ci, qi))

    rng = random.Random(seed)
    selected: set[tuple[int, int]] = set()
    for pairs in cat_index.values():
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        selected.update(shuffled[:per_category])

    # Rebuild conversations keeping only sampled qa_pairs (all sessions intact)
    new_convs: list[Conversation] = []
    for ci, conv in enumerate(dataset.conversations):
        filtered = [qa for qi, qa in enumerate(conv.qa_pairs) if (ci, qi) in selected]
        if filtered:
            new_convs.append(
                Conversation(
                    sample_id=conv.sample_id,
                    speaker_a=conv.speaker_a,
                    speaker_b=conv.speaker_b,
                    sessions=conv.sessions,
                    qa_pairs=filtered,
                )
            )
    return LoCoMoDataset(conversations=new_convs)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _aggregate_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Return aggregate + per-category F1/judge_acc for *predictions*."""
    scores = score_predictions(predictions)
    return scores


def _aggregate_f1(predictions: list[dict[str, Any]]) -> float:
    non_adv = [p for p in predictions if not p.get("is_adversarial")]
    if not non_adv:
        return float("nan")
    return sum(token_f1(str(p["prediction"]), str(p["gold"])) for p in non_adv) / len(non_adv)


# ---------------------------------------------------------------------------
# Pareto sweep
# ---------------------------------------------------------------------------


def run_pareto_sweep(
    dataset: LoCoMoDataset,
    *,
    weight_configs: dict[str, list[float]] | None = None,
    baseline: str = "naive_rag",
    checkpoint: Path = DEFAULT_CHECKPOINT,
    run_judge: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the 10-config AFT weight sweep + naive_rag baseline on *dataset*.

    Returns a results dict with per-config predictions and aggregate scores.
    """
    if weight_configs is None:
        weight_configs = WEIGHT_CONFIGS

    done = _load_checkpoint(checkpoint)
    if done and verbose:
        print(f"  Resuming: {len(done)} conversation(s) already in checkpoint.")

    config_results: list[dict[str, Any]] = []

    # Iterate AFT weight configs first, then the naive_rag baseline
    configs_to_run: list[tuple[str, Any]] = [
        (config_id, weight) for config_id, weight in weight_configs.items()
    ]
    configs_to_run.append((baseline, None))

    for config_id, weights in configs_to_run:
        if weights is not None:
            adapter: AFTLoCoMoAdapter | NaiveRAGLoCoMoAdapter = AFTLoCoMoAdapter(
                config=EmotionalMemoryConfig(retrieval=RetrievalConfig(base_weights=weights))
            )
        else:
            adapter = NaiveRAGLoCoMoAdapter()

        if verbose:
            label = f"weights={weights}" if weights is not None else "naive_rag"
            print(f"\n[{config_id}] {label}")

        all_predictions: list[dict[str, Any]] = []

        for conv in dataset.conversations:
            key = (config_id, conv.sample_id)
            if key in done:
                if verbose:
                    print(f"  [{config_id}] {conv.sample_id} — skipped (checkpoint)")
                all_predictions.extend(done[key])
                continue

            if verbose:
                print(f"  [{config_id}] {conv.sample_id} — {len(conv.qa_pairs)} QA pairs …")
            preds = adapter.run_conversation(conv)

            if run_judge:
                if verbose:
                    n_non_adv = sum(1 for p in preds if not p.get("is_adversarial"))
                    print(f"  [{config_id}] {conv.sample_id} — judging {n_non_adv} items …")
                _run_judge(preds, verbose=verbose)

            all_predictions.extend(preds)
            _append_checkpoint(checkpoint, config_id, conv.sample_id, preds)

        scores = score_predictions(all_predictions)
        config_results.append(
            {
                "config_id": config_id,
                "weights": weights,
                "n_predictions": len(all_predictions),
                "scores": scores,
                "predictions": all_predictions,
            }
        )
        if verbose:
            agg = scores.get("aggregate", {})
            print(
                f"  [{config_id}] aggregate F1={agg.get('f1', float('nan')):.4f}"
                + (
                    f"  judge_acc={agg.get('judge_accuracy', float('nan')):.4f}"
                    if run_judge
                    else ""
                )
            )

    out: dict[str, Any] = {
        "benchmark": "locomo_pareto_sweep_v1",
        "protocol": "benchmarks/preregistration_addendum_j.md",
        "sample_seed": SAMPLE_SEED,
        "sample_per_category": SAMPLE_PER_CATEGORY,
        "n_conversations": len(dataset.conversations),
        "n_qa_total": dataset.total_qa,
        "configs": config_results,
    }
    out["pareto_table"] = _compute_pareto_table(out)
    return out


# ---------------------------------------------------------------------------
# Pareto analysis
# ---------------------------------------------------------------------------

_PARETO_MIN_IMPROVEMENT: float = 0.01  # Add. J §Primary analysis
_PARETO_MAX_REGRESSION: float = 0.05  # aggregate F1 must not drop by more


def _compute_pareto_table(results: dict[str, Any]) -> dict[str, Any]:
    """Build (config x category) F1 matrix + Pareto flags per Add. J §Hj1."""
    configs_by_id: dict[str, dict[str, Any]] = {c["config_id"]: c for c in results["configs"]}
    baseline_data = configs_by_id.get("naive_rag")
    w0_data = configs_by_id.get("W0")

    category_names = [QA_CATEGORY_NAMES[c] for c in PARETO_CATEGORIES]

    def _cat_f1(config_data: dict[str, Any], cat_name: str) -> float:
        return float(
            config_data["scores"].get("by_category", {}).get(cat_name, {}).get("f1", float("nan"))
        )

    def _agg_f1(config_data: dict[str, Any]) -> float:
        return float(config_data["scores"].get("aggregate", {}).get("f1", float("nan")))

    rows: list[dict[str, Any]] = []
    pareto_favourable: list[str] = []
    hj1_pass = False

    for config_data in results["configs"]:
        config_id = config_data["config_id"]
        row: dict[str, Any] = {
            "config_id": config_id,
            "aggregate_f1": _agg_f1(config_data),
            "by_category": {},
        }

        if w0_data is not None:
            row["delta_f1_vs_W0"] = row["aggregate_f1"] - _agg_f1(w0_data)
        if baseline_data is not None:
            row["delta_f1_vs_naive_rag"] = row["aggregate_f1"] - _agg_f1(baseline_data)

        is_pareto = False
        for cat_name in category_names:
            cat_f1 = _cat_f1(config_data, cat_name)
            cat_row: dict[str, Any] = {"f1": cat_f1}
            if w0_data is not None:
                cat_row["delta_vs_W0"] = cat_f1 - _cat_f1(w0_data, cat_name)
            if baseline_data is not None:
                cat_row["delta_vs_naive_rag"] = cat_f1 - _cat_f1(baseline_data, cat_name)
                cat_row["closes_gap_to_naive"] = cat_f1 >= _cat_f1(baseline_data, cat_name)
            row["by_category"][cat_name] = cat_row

            # Pareto condition: improves ≥+0.01 on this category vs W0 without
            # aggregate regression > -0.05 vs W0
            if config_id not in ("W0", "naive_rag") and w0_data is not None and not is_pareto:
                cat_improvement = cat_f1 - _cat_f1(w0_data, cat_name)
                agg_regression = _agg_f1(w0_data) - row["aggregate_f1"]
                if (
                    cat_improvement >= _PARETO_MIN_IMPROVEMENT
                    and agg_regression <= _PARETO_MAX_REGRESSION
                ):
                    is_pareto = True

        row["pareto_dominates_W0"] = is_pareto
        if is_pareto:
            pareto_favourable.append(config_id)

        # Hj1 per Add. J §Primary analysis: aft_W.F1(C) ≥ naive_rag.F1(C) for
        # at least one (W, C) pair — distinct from the Pareto-vs-W0 condition.
        if config_id not in ("W0", "naive_rag") and baseline_data is not None:
            for cat_name in category_names:
                if row["by_category"].get(cat_name, {}).get("closes_gap_to_naive", False):
                    hj1_pass = True
                    break

        rows.append(row)

    return {
        "hj1_verdict": "PASS" if hj1_pass else "FAIL",
        "pareto_favourable_configs": pareto_favourable,
        "rows": rows,
        "category_names": category_names,
    }


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def _write_markdown(results: dict[str, Any], path: Path) -> None:
    table = results["pareto_table"]
    hj1 = table["hj1_verdict"]
    pareto = table["pareto_favourable_configs"]
    cat_names = table["category_names"]

    lines: list[str] = [
        "# LoCoMo Pareto Sweep Results — Addendum J",
        "",
        "**Protocol:** `benchmarks/preregistration_addendum_j.md` (commit `7bcd663`)",
        f"**Sample:** {results['sample_per_category']} QA x 4 categories = "
        f"{results['n_qa_total']} QA pairs total, seed={results['sample_seed']}",
        f"**N conversations:** {results['n_conversations']}",
        "",
        f"## Hj1 Verdict: **{hj1}**",
        "",
    ]

    if pareto:
        lines += [
            "Pareto-favourable configs (≥+0.01 on ≥1 category vs W0, aggregate regression ≤0.05):",
            "",
        ]
        for cfg in pareto:
            lines.append(f"- **{cfg}**")
        lines.append("")
    else:
        lines += [
            "No weight config improves any category by ≥+0.01 vs W0 without aggregate regression.",
            "",
        ]

    # Aggregate table
    lines += [
        "## Aggregate F1 (all 4 categories pooled)",
        "",
        "| Config | F1 | judge_acc | Δ vs W0 | Δ vs naive_rag | Pareto? |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table["rows"]:
        cfg = row["config_id"]
        f1 = row["aggregate_f1"]
        dw0 = row.get("delta_f1_vs_W0", float("nan"))
        dnr = row.get("delta_f1_vs_naive_rag", float("nan"))
        pareto_flag = "✓" if row.get("pareto_dominates_W0") else ""
        # judge_acc from scores
        scores_row: dict[str, Any] = next(
            (c for c in results["configs"] if c["config_id"] == cfg), {}
        )
        ja = scores_row.get("scores", {}).get("aggregate", {}).get("judge_accuracy", float("nan"))
        f1_s = f"{f1:.4f}" if f1 == f1 else "—"
        ja_s = f"{ja:.4f}" if ja == ja else "—"
        dw0_s = (f"+{dw0:.4f}" if dw0 >= 0 else f"{dw0:.4f}") if dw0 == dw0 else "—"
        dnr_s = (f"+{dnr:.4f}" if dnr >= 0 else f"{dnr:.4f}") if dnr == dnr else "—"
        lines.append(f"| {cfg} | {f1_s} | {ja_s} | {dw0_s} | {dnr_s} | {pareto_flag} |")
    lines.append("")

    # Per-category F1 matrix
    lines += [
        "## Per-category F1 matrix",
        "",
        "| Config | " + " | ".join(cat_names) + " |",
        "|---|" + "|".join("---:" for _ in cat_names) + "|",
    ]
    for row in table["rows"]:
        cfg = row["config_id"]
        cats = row.get("by_category", {})
        cells = []
        for cat in cat_names:
            f1 = cats.get(cat, {}).get("f1", float("nan"))
            cells.append(f"{f1:.4f}" if f1 == f1 else "—")
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    lines.append("")

    # Per-category Δ vs naive_rag
    lines += [
        "## Per-category Δ F1 vs naive_rag",
        "",
        "| Config | " + " | ".join(cat_names) + " |",
        "|---|" + "|".join("---:" for _ in cat_names) + "|",
    ]
    for row in table["rows"]:
        cfg = row["config_id"]
        cats = row.get("by_category", {})
        cells = []
        for cat in cat_names:
            d = cats.get(cat, {}).get("delta_vs_naive_rag", float("nan"))
            if d != d:
                cells.append("—")
            else:
                cells.append(f"+{d:.4f}" if d >= 0 else f"{d:.4f}")
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LoCoMo Pareto weight sweep (Addendum J pre-registered protocol)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use 4 QA per category instead of 50; useful for smoke testing.",
    )
    parser.add_argument(
        "--limit-configs",
        type=int,
        default=None,
        metavar="N",
        help="Only run the first N weight configs (W0…W{N-1}) plus naive_rag.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge step (F1 only).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        metavar="PATH",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
        metavar="PATH",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUT_MD,
        metavar="PATH",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    verbose = not args.quiet
    per_category = 4 if args.dry_run else SAMPLE_PER_CATEGORY

    if verbose:
        print("Loading LoCoMo dataset …")
    full_dataset = load_dataset()

    if verbose:
        print(
            f"Subsampling: {per_category} QA x {len(PARETO_CATEGORIES)} categories"
            f" (seed={SAMPLE_SEED}) ..."
        )
    sub_dataset = _stratified_subsample(full_dataset, seed=SAMPLE_SEED, per_category=per_category)
    if verbose:
        print(
            f"  Subsample: {sub_dataset.total_qa} QA across "
            f"{len(sub_dataset.conversations)} conversations"
        )

    weight_configs = WEIGHT_CONFIGS
    if args.limit_configs is not None:
        keys = list(WEIGHT_CONFIGS.keys())[: args.limit_configs]
        weight_configs = {k: WEIGHT_CONFIGS[k] for k in keys}

    if verbose:
        print(f"Running Pareto sweep: {len(weight_configs)} AFT configs + naive_rag …")

    results = run_pareto_sweep(
        sub_dataset,
        weight_configs=weight_configs,
        checkpoint=args.checkpoint,
        run_judge=not args.no_judge,
        verbose=verbose,
    )

    # Write outputs
    args.output_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if verbose:
        print(f"\nWrote {args.output_json}")

    _write_markdown(results, args.output_md)

    table = results["pareto_table"]
    print(f"\nHj1 verdict: {table['hj1_verdict']}")
    if table["pareto_favourable_configs"]:
        print(f"Pareto-favourable configs: {table['pareto_favourable_configs']}")
    else:
        print("No Pareto-favourable config found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
