"""LoCoMo benchmark runner.

Usage::

    uv run python -m benchmarks.locomo.runner --systems aft naive_rag
    uv run python -m benchmarks.locomo.runner --limit-conversations 2 --limit-qa 10  # dry run

Environment variables (same as LLM test suite):
  EMOTIONAL_MEMORY_LLM_API_KEY         required for answer generation + judge
  EMOTIONAL_MEMORY_LLM_BASE_URL        default https://api.openai.com/v1
  EMOTIONAL_MEMORY_LLM_MODEL           default gpt-4o-mini (project .env pins gpt-5-mini)
  EMOTIONAL_MEMORY_LLM_REASONING_EFFORT  reasoning budget for o-series / gpt-5 models
                                         (minimal / low / medium / high); omitted if empty

Prefer `make bench-locomo` over calling this module directly — it exports the .env
and sets PYTHONUNBUFFERED=1 for real-time progress output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.common.statistics import (
    bootstrap_ci,
    ci_payload,
    cohens_d_paired,
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap_diff,
)
from benchmarks.locomo.adapters.aft import AFTLoCoMoAdapter
from benchmarks.locomo.adapters.base import LoCoMoAdapter, call_llm
from benchmarks.locomo.adapters.naive_rag import NaiveRAGLoCoMoAdapter
from benchmarks.locomo.dataset import LoCoMoDataset, load_dataset
from benchmarks.locomo.scoring import (
    build_judge_prompt,
    is_adversarial_correct,
    parse_judge_response,
    score_predictions,
    token_f1,
)

DEFAULT_N_BOOTSTRAP = 2000

_HERE = Path(__file__).parent
DEFAULT_OUT_JSON = _HERE / "results.json"
DEFAULT_OUT_MD = _HERE / "results.md"
DEFAULT_CHECKPOINT = _HERE / "results.checkpoint.jsonl"

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
        prompt = build_judge_prompt(
            str(pred["question"]), str(pred["gold"]), str(pred["prediction"])
        )
        try:
            response = call_llm(prompt, temperature=0.0)
            pred["judge_correct"] = parse_judge_response(response)
        except Exception as exc:
            if verbose:
                print(f"  Judge error on item {i}: {exc}")
            pred["judge_correct"] = None


def _load_checkpoint(
    checkpoint_path: Path,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Return {(system, sample_id): judged_predictions} from a JSONL checkpoint."""
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not checkpoint_path.exists():
        return result
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entry: dict[str, Any] = json.loads(line)
            result[(entry["system"], entry["sample_id"])] = entry["predictions"]
    return result


def _append_checkpoint(
    checkpoint_path: Path,
    system: str,
    sample_id: str,
    predictions: list[dict[str, Any]],
) -> None:
    """Append one completed (system, conversation) record to the checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps({"system": system, "sample_id": sample_id, "predictions": predictions})
            + "\n"
        )


def run_benchmark(
    dataset: LoCoMoDataset,
    *,
    systems: list[str] = DEFAULT_SYSTEMS,
    run_judge: bool = True,
    verbose: bool = True,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run all *systems* on every conversation in *dataset*.

    When *checkpoint* is set, completed (system, conversation) pairs are saved to
    a JSONL file after each conversation finishes (including the judge step).  On
    restart the file is read and those pairs are skipped, so a failed run resumes
    from where it left off rather than starting over.

    Returns a results dict with per-system predictions + aggregate scores.
    """
    done: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if checkpoint is not None:
        done = _load_checkpoint(checkpoint)
        if done and verbose:
            print(f"  Resuming: {len(done)} conversation(s) already in checkpoint.")

    system_results: list[dict[str, Any]] = []

    for system_name in systems:
        adapter = _make_adapter(system_name)
        all_predictions: list[dict[str, Any]] = []

        for conv in dataset.conversations:
            key = (system_name, conv.sample_id)
            if key in done:
                if verbose:
                    print(f"  [{system_name}] {conv.sample_id} — skipped (checkpoint)")
                all_predictions.extend(done[key])
                continue

            if verbose:
                print(f"  [{system_name}] {conv.sample_id} — {len(conv.qa_pairs)} QA pairs …")
            preds = adapter.run_conversation(conv)

            if run_judge:
                if verbose:
                    n_non_adv = sum(1 for p in preds if not p.get("is_adversarial"))
                    print(f"  [{system_name}] {conv.sample_id} — judging {n_non_adv} items …")
                _run_judge(preds, verbose=verbose)

            all_predictions.extend(preds)

            if checkpoint is not None:
                _append_checkpoint(checkpoint, system_name, conv.sample_id, preds)

        scores = score_predictions(all_predictions)
        system_results.append(
            {
                "system": system_name,
                "n_predictions": len(all_predictions),
                "scores": scores,
                "predictions": all_predictions,
            }
        )

    out = {
        "benchmark": "locomo_v1",
        "dataset": "locomo10",
        "n_conversations": len(dataset.conversations),
        "n_qa_total": dataset.total_qa,
        "systems": system_results,
    }
    if len(system_results) >= 2 and run_judge:
        out["hypothesis_tests"] = _compute_hypothesis_tests(
            out, n_bootstrap=DEFAULT_N_BOOTSTRAP, seed=0
        )
    return out


def _compute_hypothesis_tests(
    results: dict[str, Any],
    *,
    baseline: str = "naive_rag",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute H1 (F1) and H2 (judge_acc) hypothesis tests vs. *baseline*.

    Pairs predictions by (sample_id, question) — requires both systems ran
    on the same conversations.  Returns a dict with per-hypothesis bootstrap
    diff, McNemar (H2 only), Holm-adjusted p-values, and an overall gate
    assessment.
    """
    sys_by_name = {s["system"]: s for s in results["systems"]}
    if baseline not in sys_by_name:
        return {}

    if "aft" not in sys_by_name:
        return {}
    aft_preds_indexed: dict[tuple[str, str], dict[str, Any]] = {}
    base_preds_indexed: dict[tuple[str, str], dict[str, Any]] = {}

    for sys_data in [sys_by_name.get("aft"), sys_by_name.get(baseline)]:
        if sys_data is None:
            continue
        target_dict = aft_preds_indexed if sys_data["system"] == "aft" else base_preds_indexed
        for pred in sys_data.get("predictions", []):
            key = (str(pred.get("sample_id", "")), str(pred.get("question", "")))
            target_dict[key] = pred

    if not aft_preds_indexed or not base_preds_indexed:
        return {}

    # Align paired predictions on the same QA items
    common_keys = sorted(set(aft_preds_indexed) & set(base_preds_indexed))
    if not common_keys:
        return {}

    aft_f1, base_f1 = [], []
    aft_judge: list[float] = []
    base_judge: list[float] = []
    only_aft, only_base = 0, 0

    for key in common_keys:
        ap = aft_preds_indexed[key]
        bp = base_preds_indexed[key]
        if ap.get("is_adversarial") or bp.get("is_adversarial"):
            continue
        aft_f1.append(token_f1(str(ap.get("prediction", "")), str(ap.get("gold", ""))))
        base_f1.append(token_f1(str(bp.get("prediction", "")), str(bp.get("gold", ""))))
        aj = ap.get("judge_correct")
        bj = bp.get("judge_correct")
        if aj is not None and bj is not None:
            aft_judge.append(float(aj))
            base_judge.append(float(bj))
            if aj and not bj:
                only_aft += 1
            elif bj and not aj:
                only_base += 1

    raw_p_values: list[float] = []
    h1: dict[str, Any] = {}
    h2: dict[str, Any] = {}

    if aft_f1:
        diff, lo, hi, p_two = paired_bootstrap_diff(
            aft_f1, base_f1, n_bootstrap=n_bootstrap, seed=seed
        )
        p_one = p_two / 2.0 if diff > 0 else 1.0
        d = cohens_d_paired(aft_f1, base_f1)
        aft_ci = bootstrap_ci(aft_f1, n_bootstrap=n_bootstrap, seed=seed)
        base_ci_val = bootstrap_ci(base_f1, n_bootstrap=n_bootstrap, seed=seed)
        h1 = {
            "metric": "f1",
            "aft": ci_payload(*aft_ci, n_bootstrap=n_bootstrap),
            "baseline": ci_payload(*base_ci_val, n_bootstrap=n_bootstrap),
            "diff": round(diff, 4),
            "diff_ci_lower": round(lo, 4),
            "diff_ci_upper": round(hi, 4),
            "p_bootstrap_onetailed": round(p_one, 4),
            "cohens_d": round(d, 4),
            "n_pairs": len(aft_f1),
        }
        raw_p_values.append(p_two)

    if aft_judge:
        diff2, lo2, hi2, p_two2 = paired_bootstrap_diff(
            aft_judge, base_judge, n_bootstrap=n_bootstrap, seed=seed
        )
        p_one2 = p_two2 / 2.0 if diff2 > 0 else 1.0
        d2 = cohens_d_paired(aft_judge, base_judge)
        aft_j_ci = bootstrap_ci(aft_judge, n_bootstrap=n_bootstrap, seed=seed)
        base_j_ci = bootstrap_ci(base_judge, n_bootstrap=n_bootstrap, seed=seed)
        p_mc = mcnemar_exact(only_aft, only_base)
        h2 = {
            "metric": "judge_accuracy",
            "aft": ci_payload(*aft_j_ci, n_bootstrap=n_bootstrap),
            "baseline": ci_payload(*base_j_ci, n_bootstrap=n_bootstrap),
            "diff": round(diff2, 4),
            "diff_ci_lower": round(lo2, 4),
            "diff_ci_upper": round(hi2, 4),
            "p_bootstrap_onetailed": round(p_one2, 4),
            "p_mcnemar_twotailed": round(p_mc, 4),
            "cohens_d": round(d2, 4),
            "n_pairs": len(aft_judge),
            "discordant_only_aft": only_aft,
            "discordant_only_baseline": only_base,
        }
        raw_p_values.append(p_two2)

    # Holm correction across H1 and H2 using two-tailed p-values
    adj = holm_bonferroni(raw_p_values)
    idx = 0
    if h1:
        h1["p_adj_holm"] = round(adj[idx], 4)
        h1["pass"] = h1["diff"] > 0 and h1["p_adj_holm"] < 0.05
        idx += 1
    if h2:
        h2["p_adj_holm"] = round(adj[idx], 4)
        h2["pass"] = h2["diff"] > 0 and h2["p_adj_holm"] < 0.05

    gate1 = h1.get("pass", False) or h2.get("pass", False)

    return {
        "baseline": baseline,
        "n_bootstrap": n_bootstrap,
        "H1": h1,
        "H2": h2,
        "gate1_pass": gate1,
        "gate1_label": "PASS" if gate1 else "FAIL",
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

    ht = results.get("hypothesis_tests")
    if ht:
        lines += [
            "## Hypothesis Tests (pre-registration S1)",
            "",
            f"Gate 1: **{ht['gate1_label']}**  ",
            f"(n_bootstrap={ht['n_bootstrap']}, Holm-Bonferroni correction across H1+H2)",
            "",
            "| Hypothesis | Metric | AFT | Baseline | Δ [95%CI] | p_one | p_adj | Result |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ]
        for hyp_key in ("H1", "H2"):
            h = ht.get(hyp_key)
            if not h:
                continue
            aft_pt = h["aft"]["point"]
            base_pt = h["baseline"]["point"]
            diff = h["diff"]
            ci_lo = h["diff_ci_lower"]
            ci_hi = h["diff_ci_upper"]
            p_one = h["p_bootstrap_onetailed"]
            p_adj = h["p_adj_holm"]
            verdict = "✓ PASS" if h["pass"] else "✗ FAIL"
            lines.append(
                f"| **{hyp_key}** | {h['metric']} "
                f"| {aft_pt:.3f} | {base_pt:.3f} "
                f"| {diff:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}] "
                f"| {p_one:.4f} | {p_adj:.4f} | **{verdict}** |"
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
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint/resume (start fresh every run).",
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

    checkpoint = None if args.no_checkpoint else DEFAULT_CHECKPOINT

    results = run_benchmark(
        dataset,
        systems=args.systems,
        run_judge=not args.no_judge,
        verbose=True,
        checkpoint=checkpoint,
    )

    write_results(results, out_json=args.out_json, out_md=args.out_md)
    print(f"\nResults written to {args.out_json} and {args.out_md}")

    print("\n=== Aggregate Scores ===")
    for sys in results["systems"]:
        agg = sys["scores"]["aggregate"]
        judge_str = f"  judge={agg['judge_accuracy']:.3f}" if "judge_accuracy" in agg else ""
        print(f"  {sys['system']:12s}  F1={agg['f1']:.3f}  BLEU-1={agg['bleu1']:.3f}{judge_str}")

    ht = results.get("hypothesis_tests")
    if ht:
        print(f"\n=== Gate 1: {ht['gate1_label']} ===")
        for hyp in ("H1", "H2"):
            h = ht.get(hyp)
            if not h:
                continue
            verdict = "PASS" if h["pass"] else "FAIL"
            print(
                f"  {hyp} ({h['metric']}): Δ={h['diff']:+.4f} "
                f"p_adj={h['p_adj_holm']:.4f}  → {verdict}"
            )


if __name__ == "__main__":
    main()
