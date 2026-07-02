"""Addendum X runner — third-party emotion-triggered retrieval (MADial-Bench EN).

Primary contrast (Hx1): ``aft_query_appraised`` vs ``naive_cosine`` on per-query
nDCG@5, one-tailed paired bootstrap (n=10k, seed=0), decision rule ex-ante.
Pre-registration: ``benchmarks/preregistration_addendum_x_madialbench_third_party.md``
(incl. Amendment A1).

Exploratory arms (``aft_full_stack``, ``mem0``) are pre-declared as droppable;
this runner implements the two primary arms — the closure records the drop
decision explicitly.

Usage::

    make bench-x-madial                                       # scored run (needs API key)
    uv run python -m benchmarks.madialbench.runner --dry-run  # smoke: 10 queries, no LLM
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from benchmarks.common.statistics import cohens_d_paired, paired_bootstrap_diff
from benchmarks.madialbench.adapters import (
    AFTQueryAppraisedMadialAdapter,
    MadialAdapter,
    NaiveCosineMadialAdapter,
)
from benchmarks.madialbench.dataset import (
    NEGATIVE_EMOTIONS,
    MadialDataset,
    load_dataset,
)
from benchmarks.madialbench.metrics import K_GRID, METRICS, ndcg_at_k
from emotional_memory.embedders import SentenceTransformerEmbedder

_HERE = Path(__file__).parent
DEFAULT_OUT_JSON = _HERE / "results.json"
DEFAULT_OUT_MD = _HERE / "results.md"
DEFAULT_OUT_PROTOCOL = _HERE / "results.protocol.json"

DEFAULT_N_BOOTSTRAP = 10_000
PRIMARY = "aft_query_appraised"
BASELINE = "naive_cosine"
PRIMARY_K = 5
RETRIEVE_K = max(K_GRID)
D2_VALENCE_GAP = 0.2


def _compute_stats(
    a_scores: list[float], b_scores: list[float], *, n_bootstrap: int, seed: int
) -> dict[str, Any]:
    diff, lo, hi, p_two = paired_bootstrap_diff(
        a_scores, b_scores, n_bootstrap=n_bootstrap, seed=seed
    )
    p_one = p_two / 2.0 if diff >= 0 else 1.0 - p_two / 2.0
    diffs = [a - b for a, b in zip(a_scores, b_scores, strict=True)]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1)) if n > 1 else float("nan")
    # One-tailed alpha=.05, power=.80: z_{.95} + z_{.80} = 1.645 + 0.842.
    mde = 2.487 * sd / math.sqrt(n) if n else float("nan")
    return {
        "delta": diff,
        "ci_lower": lo,
        "ci_upper": hi,
        "p_bootstrap_onetail": p_one,
        "cohens_d": cohens_d_paired(a_scores, b_scores),
        "n": n,
        "sd_paired_diff": sd,
        "mde_80pct_power": mde,
    }


def _auc(positives: list[float], negatives: list[float]) -> float:
    """Mann-Whitney AUC: P(pos > neg) + 0.5 * P(tie)."""
    if not positives or not negatives:
        return float("nan")
    wins = ties = 0
    for p in positives:
        for q in negatives:
            if p > q:
                wins += 1
            elif p == q:
                ties += 1
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def run_benchmark(
    dataset: MadialDataset,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    queries = list(dataset.queries[:10] if dry_run else dataset.queries)
    memories = list(dataset.memories)

    # One embedder instance shared by both arms ("share the same embedder",
    # pre-registration §Protocol) — halves model RAM, identical scores.
    embedder = SentenceTransformerEmbedder.make_bge_small()
    adapters: dict[str, MadialAdapter] = {
        BASELINE: NaiveCosineMadialAdapter(embedder=embedder),
        PRIMARY: AFTQueryAppraisedMadialAdapter(dry_run=dry_run, embedder=embedder),
    }

    # {arm: {metric@k: [per-query score]}} — queries in file order for pairing.
    grid: dict[str, dict[str, list[float]]] = {
        arm: {f"{m}@{k}": [] for m in METRICS for k in K_GRID} for arm in adapters
    }
    primary_scores: dict[str, list[float]] = {arm: [] for arm in adapters}

    for arm, adapter in adapters.items():
        if verbose:
            print(f"[{arm}] ingesting {len(memories)} memories …")
        adapter.ingest(memories)
        for qi, query in enumerate(queries):
            if verbose and qi % 40 == 0:
                print(f"[{arm}] query {qi}/{len(queries)} …")
            retrieved = adapter.retrieve(query.text, top_k=RETRIEVE_K)
            for mname, fn in METRICS.items():
                for k in K_GRID:
                    grid[arm][f"{mname}@{k}"].append(fn(query.gold_ids, retrieved, k))
            primary_scores[arm].append(ndcg_at_k(query.gold_ids, retrieved, PRIMARY_K))

    if verbose:
        print("Computing statistics …")
    hx1 = _compute_stats(
        primary_scores[PRIMARY], primary_scores[BASELINE], n_bootstrap=n_bootstrap, seed=seed
    )
    hx1_pass = hx1["p_bootstrap_onetail"] < 0.05 and hx1["delta"] > 0 and hx1["ci_lower"] > 0

    secondary = {
        key: _compute_stats(
            grid[PRIMARY][key], grid[BASELINE][key], n_bootstrap=n_bootstrap, seed=seed
        )
        for key in sorted(grid[PRIMARY])
    }

    # Diagnostics (AFT arm).
    aft = adapters[PRIMARY]
    assert isinstance(aft, AFTQueryAppraisedMadialAdapter)
    happy_v = [aft.encoded_affect[m.memory_id][0] for m in memories if m.emotion == "Happy"]
    neg_v = [
        aft.encoded_affect[m.memory_id][0] for m in memories if m.emotion in NEGATIVE_EMOTIONS
    ]
    d1 = {
        "auc_happy_vs_negative": _auc(happy_v, neg_v),
        "n_happy": len(happy_v),
        "n_negative": len(neg_v),
        "mean_valence_happy": sum(happy_v) / len(happy_v) if happy_v else float("nan"),
        "mean_valence_negative": sum(neg_v) / len(neg_v) if neg_v else float("nan"),
    }

    bank_mean_v = sum(v for v, _ in aft.encoded_affect.values()) / len(aft.encoded_affect)
    n_discriminative = 0
    for query in queries:
        gold_vs = [aft.encoded_affect[g][0] for g in query.gold_ids if g in aft.encoded_affect]
        if gold_vs and abs(sum(gold_vs) / len(gold_vs) - bank_mean_v) > D2_VALENCE_GAP:
            n_discriminative += 1
    d2 = {
        "bank_mean_valence": bank_mean_v,
        "gap_threshold": D2_VALENCE_GAP,
        "n_affect_discriminative": n_discriminative,
        "share_affect_discriminative": n_discriminative / len(queries) if queries else 0.0,
    }

    for adapter in adapters.values():
        adapter.close()
    return {
        "benchmark": "addendum_x_madialbench",
        "pre_registration": ("benchmarks/preregistration_addendum_x_madialbench_third_party.md"),
        "dry_run": dry_run,
        "n_queries": len(queries),
        "n_memories": len(memories),
        "embedder": "bge-small-en-v1.5",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "arms": {
            arm: {key: sum(v) / len(v) if v else 0.0 for key, v in grid[arm].items()}
            for arm in adapters
        },
        "hx1": {"metric": f"ndcg@{PRIMARY_K}", **hx1, "pass": hx1_pass},
        "secondary_contrasts": secondary,
        "diagnostic_d1": d1,
        "diagnostic_d2": d2,
        "exploratory_arms": {
            "aft_full_stack": "not run (Amendment A1: conditional, decision in closure)",
            "mem0": "not run (pre-declared droppable, decision in closure)",
        },
    }


def write_results(
    results: dict[str, Any],
    *,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
    out_protocol: Path | None = DEFAULT_OUT_PROTOCOL,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    hx1 = results["hx1"]
    lines = [
        "# Addendum X — Third-party Retrieval on MADial-Bench EN (Hx1)",
        "",
        f"**Queries:** {results['n_queries']}  **Memories:** {results['n_memories']}  "
        f"**Embedder:** `{results['embedder']}`  "
        f"**Bootstrap:** n={results['n_bootstrap']}, seed={results['seed']}"
        + ("  **[DRY RUN — not a scored result]**" if results["dry_run"] else ""),
        "",
        "## Metric grid (per-arm means)",
        "",
        "| Metric | " + " | ".join(results["arms"]) + " |",
        "|---|" + "---|" * len(results["arms"]),
    ]
    keys = sorted(next(iter(results["arms"].values())))
    for key in keys:
        row = " | ".join(f"{results['arms'][arm][key]:.3f}" for arm in results["arms"])
        lines.append(f"| {key} | {row} |")
    lines += [
        "",
        "## Hx1 — aft_query_appraised vs naive_cosine",
        "",
        f"Metric: **{hx1['metric']}**  Δ={hx1['delta']:+.3f} "
        f"[{hx1['ci_lower']:+.3f}, {hx1['ci_upper']:+.3f}]  "
        f"p_one={hx1['p_bootstrap_onetail']:.4f}  d={hx1['cohens_d']:.3f}",
        f"MDE (80% power): {hx1['mde_80pct_power']:.3f} "
        f"(sd of paired diffs {hx1['sd_paired_diff']:.3f}, N={hx1['n']})",
        "",
        f"**Hx1 verdict: {'PASS' if hx1['pass'] else 'FAIL'}**"
        + (" *(dry run — no verdict)*" if results["dry_run"] else ""),
        "",
        "## Diagnostics",
        "",
        f"D1 (appraisal vs third-party labels): AUC(Happy vs negative) = "
        f"{results['diagnostic_d1']['auc_happy_vs_negative']:.3f} "
        f"(n={results['diagnostic_d1']['n_happy']}/{results['diagnostic_d1']['n_negative']}; "
        f"mean valence {results['diagnostic_d1']['mean_valence_happy']:+.3f} vs "
        f"{results['diagnostic_d1']['mean_valence_negative']:+.3f})",
        f"D2 (corpus affect-discriminativeness): "
        f"{results['diagnostic_d2']['share_affect_discriminative']:.1%} of queries have "
        f"|gold-set mean valence - bank mean| > {results['diagnostic_d2']['gap_threshold']}",
        "",
        "Decision rule: `benchmarks/preregistration_addendum_x_madialbench_third_party.md`.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    if out_protocol is not None:
        out_protocol.write_text(
            json.dumps(
                {
                    "benchmark": results["benchmark"],
                    "pre_registration": results["pre_registration"],
                    "arms": list(results["arms"]),
                    "primary_contrast": f"{PRIMARY} vs {BASELINE}",
                    "primary_metric": f"ndcg@{PRIMARY_K}",
                    "n_queries": results["n_queries"],
                    "embedder": results["embedder"],
                    "n_bootstrap": results["n_bootstrap"],
                    "seed": results["seed"],
                    "dry_run": results["dry_run"],
                    "decision_rule": (
                        "PASS iff p_onetail<0.05 AND delta>0 AND ci_lower>0 on ndcg@5 "
                        "(single family member, m=1)"
                    ),
                    "hx1_pass": results["hx1"]["pass"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Addendum X MADial-Bench benchmark")
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    p.add_argument("--out-protocol", type=Path, default=DEFAULT_OUT_PROTOCOL)
    p.add_argument(
        "--dry-run", action="store_true", help="Smoke: 10 queries, keyword appraiser, no LLM"
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = load_dataset()
    results = run_benchmark(
        dataset,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )
    out_json, out_md, out_protocol = args.out_json, args.out_md, args.out_protocol
    if args.dry_run:
        # Never clobber committed scored artifacts with smoke output.
        if out_json == DEFAULT_OUT_JSON:
            out_json = out_json.with_name("results.dry.json")
        if out_md == DEFAULT_OUT_MD:
            out_md = out_md.with_name("results.dry.md")
        if out_protocol == DEFAULT_OUT_PROTOCOL:
            out_protocol = out_protocol.with_name("results.protocol.dry.json")
    write_results(results, out_json=out_json, out_md=out_md, out_protocol=out_protocol)
    args.out_json = out_json
    hx1 = results["hx1"]
    print(f"\nResults written to {args.out_json}")
    print(
        f"Hx1 ({hx1['metric']}): Δ={hx1['delta']:+.3f} "
        f"[{hx1['ci_lower']:+.3f}, {hx1['ci_upper']:+.3f}] "
        f"p_one={hx1['p_bootstrap_onetail']:.4f} → "
        f"{'PASS' if hx1['pass'] else 'FAIL'}"
        + (" (dry run — not a scored result)" if results["dry_run"] else "")
    )


if __name__ == "__main__":
    main()
