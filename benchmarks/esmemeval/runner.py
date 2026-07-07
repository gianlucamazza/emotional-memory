"""Addendum X2 runner — third-party longitudinal QA retrieval (ES-MemEval/EvoEmo).

Primary contrast (Hx2): ``aft_query_appraised`` vs ``naive_cosine`` on per-query
upstream-verbatim nDCG@4 over the N=1,133 in-family queries, one-tailed paired
bootstrap (n=10k, seed=0), decision rule ex-ante. Pre-registration:
``benchmarks/preregistration_addendum_x2_esmemeval_third_party.md``.

The exploratory arm (``aft_full_stack``) is pre-declared as droppable; this
runner implements the two primary arms — the closure records the drop decision
explicitly.

Usage::

    make bench-x2-esmem                                      # scored run (needs API key)
    uv run python -m benchmarks.esmemeval.runner --dry-run   # smoke: 10 queries, no LLM
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from random import Random
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from benchmarks.common.statistics import cohens_d_paired, paired_bootstrap_diff
from benchmarks.esmemeval.adapters import (
    AFTQueryAppraisedEsmemAdapter,
    EsmemAdapter,
    NaiveCosineEsmemAdapter,
)
from benchmarks.esmemeval.dataset import (
    EsmemDataset,
    build_pools,
    emotion_class,
    load_dataset,
)
from benchmarks.esmemeval.metrics import (
    PRIMARY_K,
    STD_K_GRID,
    STD_METRICS,
    UPSTREAM_K_GRID,
    UPSTREAM_METRICS,
    upstream_ndcg_at_k,
)
from emotional_memory.embedders import SentenceTransformerEmbedder

_HERE = Path(__file__).parent
DEFAULT_OUT_JSON = _HERE / "results.json"
DEFAULT_OUT_MD = _HERE / "results.md"
DEFAULT_OUT_PROTOCOL = _HERE / "results.protocol.json"

DEFAULT_N_BOOTSTRAP = 10_000
PRIMARY = "aft_query_appraised"
BASELINE = "naive_cosine"
RETRIEVE_K = max(STD_K_GRID)
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


def _auc_bootstrap_ci(
    positives: list[float], negatives: list[float], *, n_bootstrap: int, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap CI for the AUC (both classes resampled).

    Pre-registered because the positive class is tiny (~7 sessions) — D1 is
    read jointly with this CI.
    """
    if not positives or not negatives:
        return (float("nan"), float("nan"))
    rng = Random(seed)
    stats = sorted(
        _auc(
            [positives[rng.randrange(len(positives))] for _ in positives],
            [negatives[rng.randrange(len(negatives))] for _ in negatives],
        )
        for _ in range(n_bootstrap)
    )
    return (stats[int(0.025 * n_bootstrap)], stats[int(0.975 * n_bootstrap)])


def run_benchmark(
    dataset: EsmemDataset,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    queries = list(dataset.queries[:10] if dry_run else dataset.queries)
    sessions = list(dataset.sessions)
    pools = build_pools(dataset)  # POOL_SEED fixed ex-ante, independent of --seed

    # One embedder instance shared by both arms ("share the same embedder",
    # pre-registration §Protocol) — halves model RAM, identical scores.
    embedder = SentenceTransformerEmbedder.make_bge_small()
    adapters: dict[str, EsmemAdapter] = {
        BASELINE: NaiveCosineEsmemAdapter(embedder=embedder),
        PRIMARY: AFTQueryAppraisedEsmemAdapter(dry_run=dry_run, embedder=embedder),
    }

    metric_keys = [f"{m}@{k}" for m in UPSTREAM_METRICS for k in UPSTREAM_K_GRID] + [
        f"{m}@{k}" for m in STD_METRICS for k in STD_K_GRID
    ]
    # {arm: {metric@k: [per-query score]}} — queries in file order for pairing.
    grid: dict[str, dict[str, list[float]]] = {
        arm: {key: [] for key in metric_keys} for arm in adapters
    }
    primary_scores: dict[str, list[float]] = {arm: [] for arm in adapters}

    for arm, adapter in adapters.items():
        if verbose:
            print(f"[{arm}] ingesting {len(sessions)} session documents …")
        adapter.ingest(sessions)
        for qi, query in enumerate(queries):
            if verbose and qi % 100 == 0:
                print(f"[{arm}] query {qi}/{len(queries)} …")
            retrieved = adapter.retrieve(query.text, pools[query.query_id], top_k=RETRIEVE_K)
            for mname, fn in UPSTREAM_METRICS.items():
                for k in UPSTREAM_K_GRID:
                    grid[arm][f"{mname}@{k}"].append(fn(query.gold_keys, retrieved, k))
            for mname, fn in STD_METRICS.items():
                for k in STD_K_GRID:
                    grid[arm][f"{mname}@{k}"].append(fn(query.gold_keys, retrieved, k))
            primary_scores[arm].append(upstream_ndcg_at_k(query.gold_keys, retrieved, PRIMARY_K))

    if verbose:
        print("Computing statistics …")
    hx2 = _compute_stats(
        primary_scores[PRIMARY], primary_scores[BASELINE], n_bootstrap=n_bootstrap, seed=seed
    )
    hx2_pass = hx2["p_bootstrap_onetail"] < 0.05 and hx2["delta"] > 0 and hx2["ci_lower"] > 0

    secondary = {
        key: _compute_stats(
            grid[PRIMARY][key], grid[BASELINE][key], n_bootstrap=n_bootstrap, seed=seed
        )
        for key in sorted(grid[PRIMARY])
    }

    # Per-capability breakdown of the primary metric (descriptive, non-gating).
    per_capability: dict[str, dict[str, Any]] = {}
    for cap in sorted({q.capability for q in queries}):
        idx = [i for i, q in enumerate(queries) if q.capability == cap]
        a = [primary_scores[PRIMARY][i] for i in idx]
        b = [primary_scores[BASELINE][i] for i in idx]
        per_capability[cap] = {
            "n": len(idx),
            PRIMARY: sum(a) / len(a),
            BASELINE: sum(b) / len(b),
            "delta": sum(a) / len(a) - sum(b) / len(b),
        }

    # Diagnostics (AFT arm).
    aft = adapters[PRIMARY]
    assert isinstance(aft, AFTQueryAppraisedEsmemAdapter)
    pos_v = [
        aft.encoded_affect[s.key][0] for s in sessions if emotion_class(s.emotion) == "positive"
    ]
    neg_v = [
        aft.encoded_affect[s.key][0] for s in sessions if emotion_class(s.emotion) == "negative"
    ]
    auc = _auc(pos_v, neg_v)
    auc_ci = _auc_bootstrap_ci(pos_v, neg_v, n_bootstrap=n_bootstrap, seed=seed)
    d1 = {
        "auc_positive_vs_negative": auc,
        "auc_ci95": list(auc_ci),
        "n_positive": len(pos_v),
        "n_negative": len(neg_v),
        "mean_valence_positive": sum(pos_v) / len(pos_v) if pos_v else float("nan"),
        "mean_valence_negative": sum(neg_v) / len(neg_v) if neg_v else float("nan"),
    }

    # D2 against the same seeker's bank mean (banks are per-seeker here).
    seeker_valences: dict[str, list[float]] = {}
    for s in sessions:
        seeker_valences.setdefault(s.seeker_id, []).append(aft.encoded_affect[s.key][0])
    seeker_mean = {sid: sum(vs) / len(vs) for sid, vs in seeker_valences.items()}
    n_discriminative = 0
    for query in queries:
        gold_vs = [aft.encoded_affect[k][0] for k in query.gold_keys if k in aft.encoded_affect]
        if (
            gold_vs
            and abs(sum(gold_vs) / len(gold_vs) - seeker_mean[query.seeker_id]) > D2_VALENCE_GAP
        ):
            n_discriminative += 1
    d2 = {
        "bank_scope": "per_seeker",
        "gap_threshold": D2_VALENCE_GAP,
        "n_affect_discriminative": n_discriminative,
        "share_affect_discriminative": n_discriminative / len(queries) if queries else 0.0,
    }

    results: dict[str, Any] = {
        "benchmark": "addendum_x2_esmemeval",
        "pre_registration": ("benchmarks/preregistration_addendum_x2_esmemeval_third_party.md"),
        "dry_run": dry_run,
        "n_queries": len(queries),
        "n_zero_gold_excluded": dataset.n_zero_gold,
        "n_sessions": len(sessions),
        "pool_size": len(next(iter(pools.values()))),
        "embedder": "bge-small-en-v1.5",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "arms": {
            arm: {key: sum(v) / len(v) if v else 0.0 for key, v in grid[arm].items()}
            for arm in adapters
        },
        "hx2": {"metric": f"u_ndcg@{PRIMARY_K}", **hx2, "pass": hx2_pass},
        "secondary_contrasts": secondary,
        "per_capability_primary_metric": per_capability,
        "diagnostic_d1": d1,
        "diagnostic_d2": d2,
        "exploratory_arms": {
            "aft_full_stack": "not run (pre-declared droppable, decision in closure)",
        },
    }
    if not dry_run:
        # Zero-gold-inclusive variant (N=1,427): the excluded queries score 0
        # on all metrics for every system (upstream behavior), so the inclusive
        # mean is an exact rescale of the in-family mean.
        scale = len(queries) / (len(queries) + dataset.n_zero_gold)
        results["zero_gold_inclusive_n"] = len(queries) + dataset.n_zero_gold
        results["arms_zero_gold_inclusive"] = {
            arm: {key: val * scale for key, val in results["arms"][arm].items()}
            for arm in adapters
        }

    # Aligned per-query (cosine, aft, appraised valence) for the Addendum Y gate
    # (query order = file order; valence keyed by query text, deterministic).
    results["_per_query_gate"] = {
        "metric": f"u_ndcg@{PRIMARY_K}",
        "cosine": list(primary_scores[BASELINE]),
        "aft": list(primary_scores[PRIMARY]),
        "valence": [aft.appraised_query_affect[q.text].valence for q in queries],
    }

    for adapter in adapters.values():
        adapter.close()
    return results


def gate_inputs(*, dry_run: bool = False) -> dict[str, Any]:
    """Addendum Y: aligned per-query (cosine, aft, valence) + metric for this corpus."""
    results = run_benchmark(load_dataset(), dry_run=dry_run, verbose=False)
    pq = results["_per_query_gate"]
    return {"corpus": "esmemeval", **pq}


def write_results(
    results: dict[str, Any],
    *,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
    out_protocol: Path | None = DEFAULT_OUT_PROTOCOL,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    hx2 = results["hx2"]
    d1 = results["diagnostic_d1"]
    d2 = results["diagnostic_d2"]
    lines = [
        "# Addendum X2 — Third-party Retrieval on ES-MemEval/EvoEmo (Hx2)",
        "",
        f"**Queries:** {results['n_queries']} in-family "
        f"({results['n_zero_gold_excluded']} zero-gold excluded)  "
        f"**Sessions:** {results['n_sessions']}  **Pool:** {results['pool_size']}  "
        f"**Embedder:** `{results['embedder']}`  "
        f"**Bootstrap:** n={results['n_bootstrap']}, seed={results['seed']}"
        + ("  **[DRY RUN — not a scored result]**" if results["dry_run"] else ""),
        "",
        "## Metric grid (per-arm means; `u_*` = upstream-verbatim formulas)",
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
        "## Hx2 — aft_query_appraised vs naive_cosine",
        "",
        f"Metric: **{hx2['metric']}** (upstream-verbatim formula)  Δ={hx2['delta']:+.3f} "
        f"[{hx2['ci_lower']:+.3f}, {hx2['ci_upper']:+.3f}]  "
        f"p_one={hx2['p_bootstrap_onetail']:.4f}  d={hx2['cohens_d']:.3f}",
        f"MDE (80% power): {hx2['mde_80pct_power']:.3f} "
        f"(sd of paired diffs {hx2['sd_paired_diff']:.3f}, N={hx2['n']})",
        "",
        f"**Hx2 verdict: {'PASS' if hx2['pass'] else 'FAIL'}**"
        + (" *(dry run — no verdict)*" if results["dry_run"] else ""),
        "",
        "## Per-capability (primary metric, descriptive)",
        "",
        "| Capability | n | aft | cosine | Δ |",
        "|---|---|---|---|---|",
    ]
    for cap, row in results["per_capability_primary_metric"].items():
        lines.append(
            f"| {cap} | {row['n']} | {row['aft_query_appraised']:.3f} "
            f"| {row['naive_cosine']:.3f} | {row['delta']:+.3f} |"
        )
    lines += [
        "",
        "## Diagnostics",
        "",
        f"D1 (appraisal vs third-party labels): AUC(positive vs negative) = "
        f"{d1['auc_positive_vs_negative']:.3f} "
        f"[{d1['auc_ci95'][0]:.3f}, {d1['auc_ci95'][1]:.3f}] "
        f"(n={d1['n_positive']}/{d1['n_negative']}; "
        f"mean valence {d1['mean_valence_positive']:+.3f} vs "
        f"{d1['mean_valence_negative']:+.3f})",
        f"D2 (corpus affect-discriminativeness, per-seeker banks): "
        f"{d2['share_affect_discriminative']:.1%} of queries have "
        f"|gold-set mean valence - seeker bank mean| > {d2['gap_threshold']}",
        "",
        "Decision rule: `benchmarks/preregistration_addendum_x2_esmemeval_third_party.md`.",
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
                    "primary_metric": f"u_ndcg@{PRIMARY_K} (upstream-verbatim formula)",
                    "n_queries": results["n_queries"],
                    "n_zero_gold_excluded": results["n_zero_gold_excluded"],
                    "pool_size": results["pool_size"],
                    "embedder": results["embedder"],
                    "n_bootstrap": results["n_bootstrap"],
                    "seed": results["seed"],
                    "dry_run": results["dry_run"],
                    "decision_rule": (
                        "PASS iff p_onetail<0.05 AND delta>0 AND ci_lower>0 on "
                        "upstream-verbatim ndcg@4 (single family member, m=1)"
                    ),
                    "hx2_pass": results["hx2"]["pass"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Addendum X2 ES-MemEval benchmark")
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
    hx2 = results["hx2"]
    print(f"\nResults written to {out_json}")
    print(
        f"Hx2 ({hx2['metric']}): Δ={hx2['delta']:+.3f} "
        f"[{hx2['ci_lower']:+.3f}, {hx2['ci_upper']:+.3f}] "
        f"p_one={hx2['p_bootstrap_onetail']:.4f} → "
        f"{'PASS' if hx2['pass'] else 'FAIL'}"
        + (" (dry run — not a scored result)" if results["dry_run"] else "")
    )


if __name__ == "__main__":
    main()
