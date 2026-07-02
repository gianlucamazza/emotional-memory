"""DailyDialog affect-conditioned retrieval benchmark runner (Hk1).

Usage::

    # Build personas first (one-time):
    uv run python -m benchmarks.dailydialog.persona_builder --n 120 --seed 0

    # Run benchmark:
    uv run python -m benchmarks.dailydialog.runner

    # Dry run (5 personas):
    uv run python -m benchmarks.dailydialog.runner --dry-run

Prefer ``make bench-dailydialog`` over calling this module directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.common.statistics import (
    cohens_d_paired,
    format_point_ci,
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap_diff,
)
from benchmarks.dailydialog.adapters.aft import AFTDailyDialogAdapter
from benchmarks.dailydialog.adapters.base import DailyDialogAdapter, PersonaRunResult
from benchmarks.dailydialog.adapters.naive_cosine import NaiveCosineDailyDialogAdapter
from benchmarks.dailydialog.dataset import (
    DailyDialogPersonaDataset,
    PersonaQuery,
    load_personas,
)
from benchmarks.dailydialog.query_generator import ALL_QUERY_TYPES
from benchmarks.dailydialog.scoring import hit_at_k, top1_correct

_HERE = Path(__file__).parent
DEFAULT_PERSONA_FILE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "datasets"
    / "dailydialog_personas_v1.json"
)
DEFAULT_OUT_JSON = _HERE / "results.json"
DEFAULT_OUT_MD = _HERE / "results.md"
DEFAULT_OUT_PROTOCOL = _HERE / "results.protocol.json"

DEFAULT_N_BOOTSTRAP = 10_000
DEFAULT_SYSTEMS = ["aft", "naive_cosine"]


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------


def _make_adapter(
    name: str, *, embedder_name: str = "multilingual-e5-small"
) -> DailyDialogAdapter:
    if name == "aft":
        return AFTDailyDialogAdapter(embedder_name=embedder_name)
    if name == "naive_cosine":
        return NaiveCosineDailyDialogAdapter(embedder_name=embedder_name)
    raise ValueError(f"Unknown system: {name!r}. Choices: aft, naive_cosine")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _compute_stats(
    aft_scores: list[float],
    baseline_scores: list[float],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Compute paired bootstrap diff, McNemar, and Cohen's d."""
    diff, lo, hi, p_two = paired_bootstrap_diff(
        aft_scores, baseline_scores, n_bootstrap=n_bootstrap, seed=seed
    )
    # One-tailed p (directional H0: AFT ≤ baseline)
    p_one = p_two / 2.0 if diff >= 0 else 1.0 - p_two / 2.0

    only_aft = sum(1 for a, b in zip(aft_scores, baseline_scores, strict=False) if a > b)
    only_baseline = sum(1 for a, b in zip(aft_scores, baseline_scores, strict=False) if b > a)
    p_mcnemar = mcnemar_exact(only_aft, only_baseline)
    d = cohens_d_paired(aft_scores, baseline_scores)

    return {
        "delta": diff,
        "ci_lower": lo,
        "ci_upper": hi,
        "p_bootstrap_onetail": p_one,
        "p_mcnemar": p_mcnemar,
        "cohens_d": d,
        "n": len(aft_scores),
        "only_aft_correct": only_aft,
        "only_baseline_correct": only_baseline,
    }


def _apply_holm(stats_map: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Apply Holm correction across m=len(stats_map) tests."""
    keys = list(stats_map.keys())
    raw_ps = [stats_map[k]["p_bootstrap_onetail"] for k in keys]
    adj = holm_bonferroni(raw_ps)
    for k, p_adj in zip(keys, adj, strict=False):
        stats_map[k]["p_holm"] = p_adj
        stats_map[k]["pass_holm"] = (
            p_adj < 0.05 and stats_map[k]["delta"] > 0 and stats_map[k]["ci_lower"] > 0
        )
    return stats_map


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_benchmark(
    dataset: DailyDialogPersonaDataset,
    *,
    systems: list[str] = DEFAULT_SYSTEMS,
    embedder_name: str = "multilingual-e5-small",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    verbose: bool = True,
    limit_personas: int | None = None,
) -> dict[str, Any]:
    """Run *systems* on every persona in *dataset*.

    Returns a results dict compatible with ``write_results()``.
    """
    personas = dataset.personas
    if limit_personas is not None:
        personas = personas[:limit_personas]

    if verbose:
        print(f"Running {len(systems)} system(s) on {len(personas)} personas …")

    # Collect binary correctness vectors per system per query type + aggregate
    # Shape: {system_name: {query_type | "aggregate": [0/1, ...]}}
    top1_vecs: dict[str, dict[str, list[int]]] = {
        s: {qt: [] for qt in [*ALL_QUERY_TYPES, "aggregate"]} for s in systems
    }
    hitk_vecs: dict[str, dict[str, list[int]]] = {
        s: {qt: [] for qt in [*ALL_QUERY_TYPES, "aggregate"]} for s in systems
    }

    adapters = {s: _make_adapter(s, embedder_name=embedder_name) for s in systems}

    for persona_idx, persona in enumerate(personas):
        if verbose and persona_idx % 20 == 0:
            print(f"  persona {persona_idx}/{len(personas)} …")
        for sys_name, adapter in adapters.items():
            run_results: list[PersonaRunResult] = adapter.run_persona(persona)
            for row in run_results:
                qt = row["query_type"]
                retrieved: list[str] = row["retrieved_session_ids"]

                pq = PersonaQuery(
                    query_id=row["query_id"],
                    query_type=qt,
                    text=row["query_text"],
                    target_session_id=row["target_session_id"],
                    distractor_session_ids=(),
                    top_k=row["top_k"],
                )
                c1 = int(top1_correct(retrieved, pq))
                ck = int(hit_at_k(retrieved, pq))

                top1_vecs[sys_name][qt].append(c1)
                top1_vecs[sys_name]["aggregate"].append(c1)
                hitk_vecs[sys_name][qt].append(ck)
                hitk_vecs[sys_name]["aggregate"].append(ck)

    if verbose:
        print("Computing statistics …")

    # Aggregate metrics per system
    system_results: list[dict[str, Any]] = []
    for sys_name in systems:
        agg_top1 = top1_vecs[sys_name]["aggregate"]
        agg_hitk = hitk_vecs[sys_name]["aggregate"]
        n_queries = len(agg_top1)
        top1_acc = sum(agg_top1) / n_queries if n_queries else 0.0
        hit_acc = sum(agg_hitk) / n_queries if n_queries else 0.0

        per_type: list[dict[str, Any]] = []
        for qt in ALL_QUERY_TYPES:
            v = top1_vecs[sys_name][qt]
            n = len(v)
            per_type.append(
                {
                    "query_type": qt,
                    "n": n,
                    "top1_accuracy": sum(v) / n if n else 0.0,
                    "hit_at_k": sum(hitk_vecs[sys_name][qt]) / n if n else 0.0,
                }
            )

        system_results.append(
            {
                "system": sys_name,
                "n_queries": n_queries,
                "top1_accuracy": top1_acc,
                "hit_at_k": hit_acc,
                "per_type": per_type,
                "_top1_vec": agg_top1,
                "_hitk_vec": agg_hitk,
                "_top1_vecs_by_type": {qt: top1_vecs[sys_name][qt] for qt in ALL_QUERY_TYPES},
            }
        )

    # Pairwise comparisons: AFT vs each baseline
    comparisons: list[dict[str, Any]] = []
    aft_sys = next((s for s in system_results if s["system"] == "aft"), None)
    if aft_sys is not None:
        for base_sys in system_results:
            if base_sys["system"] == "aft":
                continue
            # Aggregate comparison + per-type (Holm family: aggregate + 3 directional types)
            family_keys = ["aggregate", *ALL_QUERY_TYPES]
            stats_map: dict[str, dict[str, Any]] = {}

            for key in family_keys:
                if key == "aggregate":
                    a_vec = [float(x) for x in aft_sys["_top1_vec"]]
                    b_vec = [float(x) for x in base_sys["_top1_vec"]]
                else:
                    a_vec = [float(x) for x in aft_sys["_top1_vecs_by_type"][key]]
                    b_vec = [float(x) for x in base_sys["_top1_vecs_by_type"].get(key, [])]
                if len(a_vec) != len(b_vec) or not a_vec:
                    continue
                stats_map[key] = _compute_stats(a_vec, b_vec, n_bootstrap=n_bootstrap, seed=seed)

            stats_map = _apply_holm(stats_map)

            comparisons.append(
                {
                    "system": "aft",
                    "baseline": base_sys["system"],
                    "stats": stats_map,
                }
            )

    # Clean up internal vecs before serialising
    for s in system_results:
        s.pop("_top1_vec", None)
        s.pop("_hitk_vec", None)
        s.pop("_top1_vecs_by_type", None)

    return {
        "benchmark": "hk1_dailydialog",
        "dataset_version": dataset.version,
        "n_personas": len(personas),
        "embedder": embedder_name,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "systems": system_results,
        "pairwise_comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _verdict(stats: dict[str, Any]) -> str:
    if stats.get("pass_holm"):
        d = float(stats.get("delta", 0))
        p = float(stats.get("p_holm", 1))
        return f"**PASS** Δ={d:+.3f} p_holm={p:.3f}"
    d = float(stats.get("delta", 0))
    p = float(stats.get("p_holm", 1))
    return f"FAIL Δ={d:+.3f} p_holm={p:.3f}"


def write_results(
    results: dict[str, Any],
    *,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
    out_protocol: Path | None = DEFAULT_OUT_PROTOCOL,
) -> None:
    """Write JSON + Markdown + optional protocol artefacts."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Hk1 — DailyDialog Affect-Conditioned Retrieval Benchmark")
    lines.append("")
    lines.append(
        f"**Personas:** {results['n_personas']}  "
        f"**Embedder:** `{results['embedder']}`  "
        f"**Bootstrap:** n={results['n_bootstrap']}, seed={results['seed']}"
    )
    lines.append("")

    # Aggregate table
    lines.append("## Aggregate Results")
    lines.append("")
    lines.append("| System | N queries | top1_accuracy | hit@k |")
    lines.append("|---|---|---|---|")
    lines.extend(
        f"| {s['system']} | {s['n_queries']} | {s['top1_accuracy']:.3f} | {s['hit_at_k']:.3f} |"
        for s in results["systems"]
    )
    lines.append("")

    # Per-type breakdown
    lines.append("## By Query Type")
    lines.append("")
    for s in results["systems"]:
        lines.append(f"### {s['system']}")
        lines.append("")
        lines.append("| Query type | N | top1_accuracy | hit@k |")
        lines.append("|---|---|---|---|")
        lines.extend(
            f"| {pt['query_type']} | {pt['n']} "
            f"| {pt['top1_accuracy']:.3f} | {pt['hit_at_k']:.3f} |"
            for pt in s["per_type"]
        )
        lines.append("")

    # Pairwise comparisons
    if results.get("pairwise_comparisons"):
        lines.append("## Pairwise Comparisons (AFT vs baseline, Holm-corrected)")
        lines.append("")
        for cmp in results["pairwise_comparisons"]:
            lines.append(f"### AFT vs {cmp['baseline']}")
            lines.append("")
            lines.append("| Key | Δ | CI | p_one | p_holm | d | Verdict |")
            lines.append("|---|---|---|---|---|---|---|")
            for key, st in cmp["stats"].items():
                ci = format_point_ci(
                    float(st.get("delta", 0)),
                    float(st.get("ci_lower", 0)),
                    float(st.get("ci_upper", 0)),
                    dp=3,
                )
                lines.append(
                    f"| {key} | {st.get('delta', 0):+.3f} | {ci} "
                    f"| {st.get('p_bootstrap_onetail', 1):.3f} "
                    f"| {st.get('p_holm', 1):.3f} "
                    f"| {st.get('cohens_d', 0):.3f} "
                    f"| {_verdict(st)} |"
                )
            lines.append("")

        # Overall verdict
        aft_vs_naive = next(
            (c for c in results["pairwise_comparisons"] if c["baseline"] == "naive_cosine"),
            None,
        )
        if aft_vs_naive:
            agg = aft_vs_naive["stats"].get("aggregate", {})
            lines.append("## Hk1 Decision")
            lines.append("")
            lines.append(f"Aggregate: {_verdict(agg)}")
            lines.append("")
            n_type_pass = sum(
                1
                for qt in [
                    "emotion_state_recall",
                    "affect_conditioned_content",
                    "affective_trajectory",
                ]
                if aft_vs_naive["stats"].get(qt, {}).get("pass_holm", False)
            )
            lines.append(f"Types passing Holm: {n_type_pass}/3 directional types")
            hk1_pass = agg.get("pass_holm", False) and n_type_pass >= 2
            lines.append("")
            lines.append(f"**Hk1 verdict: {'PASS' if hk1_pass else 'FAIL'}**")
            lines.append("")
            lines.append(
                "Headline metric: `top1_accuracy`. "
                "See `benchmarks/preregistration_addendum_k_dailydialog.md` for decision rule."
            )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    if out_protocol is not None:
        protocol: dict[str, Any] = {
            "benchmark": results["benchmark"],
            "pre_registration": "benchmarks/preregistration_addendum_k_dailydialog.md",
            "systems": [s["system"] for s in results["systems"]],
            "n_personas": results["n_personas"],
            "embedder": results["embedder"],
            "n_bootstrap": results["n_bootstrap"],
            "seed": results["seed"],
            "primary_metrics": ["top1_accuracy", "hit_at_k"],
            "query_types": ALL_QUERY_TYPES,
            "hypothesis": "Hk1: AFT top1_accuracy > naive_cosine on affect-conditioned recall",
            "decision_rule": (
                "PASS iff: p_holm<0.05 AND delta>0 AND ci_lower>0 on aggregate, "
                "and >=2/3 directional types also PASS individually"
            ),
        }
        out_protocol.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Hk1 DailyDialog retrieval benchmark")
    p.add_argument(
        "--personas",
        type=Path,
        default=DEFAULT_PERSONA_FILE,
        help="Path to dailydialog_personas_v1.json",
    )
    p.add_argument(
        "--systems",
        nargs="+",
        default=DEFAULT_SYSTEMS,
        choices=["aft", "naive_cosine"],
    )
    p.add_argument(
        "--embedder",
        default="multilingual-e5-small",
        choices=["multilingual-e5-small", "sbert-bge"],
    )
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    p.add_argument("--out-protocol", type=Path, default=DEFAULT_OUT_PROTOCOL)
    p.add_argument("--dry-run", action="store_true", help="Limit to 5 personas for testing")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = load_personas(args.personas)
    limit = 5 if args.dry_run else None
    results = run_benchmark(
        dataset,
        systems=list(args.systems),
        embedder_name=args.embedder,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        verbose=not args.quiet,
        limit_personas=limit,
    )
    if args.dry_run:
        # Never clobber committed scored artifacts with smoke output.
        if args.out_json == DEFAULT_OUT_JSON:
            args.out_json = args.out_json.with_name("results.dry.json")
        if args.out_md == DEFAULT_OUT_MD:
            args.out_md = args.out_md.with_name("results.dry.md")
        if args.out_protocol == DEFAULT_OUT_PROTOCOL:
            args.out_protocol = args.out_protocol.with_name("results.protocol.dry.json")
    write_results(
        results,
        out_json=args.out_json,
        out_md=args.out_md,
        out_protocol=args.out_protocol,
    )
    print(f"\nResults written to {args.out_json}")

    # Print quick summary
    for s in results["systems"]:
        print(f"  {s['system']}: top1={s['top1_accuracy']:.3f}  hit@k={s['hit_at_k']:.3f}")

    cmp = next(
        (
            c
            for c in results.get("pairwise_comparisons", [])
            if c.get("baseline") == "naive_cosine"
        ),
        None,
    )
    if cmp:
        agg = cmp["stats"].get("aggregate", {})
        print(
            f"\nHk1 aggregate: Δ={agg.get('delta', 0):+.3f} "
            f"[{agg.get('ci_lower', 0):+.3f}, {agg.get('ci_upper', 0):+.3f}] "
            f"p_holm={agg.get('p_holm', 1):.3f} "
            f"d={agg.get('cohens_d', 0):.3f} "
            f"→ {'PASS' if agg.get('pass_holm') else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
