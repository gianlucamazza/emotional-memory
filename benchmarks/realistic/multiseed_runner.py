"""Multi-seed robustness sweep for the realistic replay benchmark.

Runs the realistic replay benchmark across several RNG seeds and reports the
cross-seed variance of the headline metrics (`top1_accuracy`, `hit@k`) and of the
AFT-vs-baseline difference.

Why this exists: the limitation tracked in ``docs/research/08_limitations.md``
(sec 2.9) is that confirmatory studies pin a single seed, so cross-run stability
was never characterized. This harness closes that gap.

Each seed is executed in an **isolated subprocess** that invokes the canonical
``benchmarks.realistic.runner`` — i.e. exactly how the benchmark is normally run
and reported (one process per run). This is deliberate: the engine stamps
encode/retrieve with real wall-clock time and ACT-R decay tracks
``now - encoded_at``, so running several full benchmarks back-to-back inside a
*single* process can perturb decay enough to flip a near-tie query (correct
production behaviour, not a defect; see docs/research/08_limitations.md sec 2.9).
Subprocess isolation matches the canonical execution model, makes the determinism
verdict trustworthy, and reuses the canonical runner verbatim (no divergent
scoring logic).

Empirically, with a deterministic embedder (``hash``, ``sbert-bge``, …) over a
fixed dataset the retrieval is *near*-deterministic: the RNG seed itself moves
nothing (only the bootstrap CI bounds jitter), but the result is **not** bit-stable
across fresh sweeps. Because each subprocess is launched at a slightly different
wall-clock moment and ACT-R decay tracks ``now - encoded_at``, a query sitting at a
numerical tie can flip between seeds *even with subprocess isolation* — so a fresh
``make bench-multiseed`` reports ``retrieval_deterministic=True`` most of the time
but occasionally ``False`` (cross-seed stdev up to ~0.0025), and the absolute mean
drifts a little across sweeps. This residual variance is timing-driven, sits well
inside the bootstrap CIs, and does not change the scientific conclusion (AFT still
clears the baseline by far more than the spread). The sweep ``verifies`` this
(``retrieval_deterministic``) per run rather than assuming exact determinism; it
would surface larger variance for a stochastic embedder or a regenerated dataset.

Offline: with ``--embedder hash`` this runs with no model download and no network.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.common.statistics import DEFAULT_N_BOOTSTRAP
from benchmarks.realistic.runner import ROOT

DEFAULT_SEEDS = [0, 1, 7, 42, 123]
DEFAULT_DATASET = ROOT / "benchmarks" / "datasets" / "realistic_recall_v2.json"
DEFAULT_OUT_JSON = ROOT / "benchmarks" / "realistic" / "multiseed_results.json"
DEFAULT_OUT_MD = ROOT / "benchmarks" / "realistic" / "multiseed_results.md"


def _per_query_top1(result: dict[str, Any]) -> dict[str, list[tuple[str, int]]]:
    """Ordered (query_id, top1_hit) pairs per system — the determinism fingerprint."""
    fingerprint: dict[str, list[tuple[str, int]]] = {}
    for system in result["systems"]:
        fingerprint[system["system"]] = [
            (query["query_id"], int(query["top1_hit"]))
            for scenario in system["scenarios"]
            for session in scenario["sessions"]
            for query in session["queries"]
        ]
    return fingerprint


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 6),
        "stdev": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "spread": round(max(values) - min(values), 6),
    }


def _run_single_seed(
    *,
    dataset_path: Path,
    seed: int,
    systems: list[str],
    embedder_name: str,
    n_bootstrap: int,
    top_k: int | None,
    workdir: Path,
) -> dict[str, Any]:
    """Run the canonical realistic runner for one seed in an isolated subprocess."""
    out_json = workdir / f"seed_{seed}.json"
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.realistic.runner",
        "--dataset",
        str(dataset_path),
        "--embedder",
        embedder_name,
        "--systems",
        ",".join(systems),
        "--seed",
        str(seed),
        "--n-bootstrap",
        str(n_bootstrap),
        "--out-json",
        str(out_json),
        "--out-md",
        str(workdir / f"seed_{seed}.md"),
        "--out-protocol",
        str(workdir / f"seed_{seed}.protocol.json"),
    ]
    if top_k is not None:
        cmd += ["--top-k", str(top_k)]
    # Command is fully controlled: sys.executable + a fixed module + validated args.
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, text=True)  # noqa: S603
    result: dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8"))
    return result


def run_multiseed(
    *,
    dataset_path: Path,
    seeds: list[int],
    systems: list[str],
    top_k: int | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    embedder_name: str = "hash",
) -> dict[str, Any]:
    """Run the realistic benchmark once per seed (isolated) and aggregate."""
    if not seeds:
        raise ValueError("seeds must be a non-empty list")

    per_seed: list[dict[str, Any]] = []
    fingerprints: list[dict[str, list[tuple[str, int]]]] = []
    top1_by_system: dict[str, list[float]] = {}
    hit_by_system: dict[str, list[float]] = {}
    diff_by_pair: dict[str, list[float]] = {}
    benchmark_name = ""
    version = ""

    with tempfile.TemporaryDirectory(prefix="emotional-memory-multiseed-") as tmp:
        workdir = Path(tmp)
        for seed in seeds:
            result = _run_single_seed(
                dataset_path=dataset_path,
                seed=seed,
                systems=systems,
                embedder_name=embedder_name,
                n_bootstrap=n_bootstrap,
                top_k=top_k,
                workdir=workdir,
            )
            fingerprints.append(_per_query_top1(result))
            benchmark_name = result["benchmark"]
            version = result["version"]

            seed_systems: dict[str, dict[str, float]] = {}
            for system in result["systems"]:
                name = system["system"]
                top1 = float(system["aggregate_metrics"]["top1_accuracy"])
                hit = float(system["aggregate_metrics"]["hit_at_k"])
                seed_systems[name] = {"top1_accuracy": top1, "hit_at_k": hit}
                top1_by_system.setdefault(name, []).append(top1)
                hit_by_system.setdefault(name, []).append(hit)

            seed_pairs: dict[str, float] = {}
            for row in result["pairwise_comparisons"]:
                if row["metric"] != "top1":
                    continue
                key = f"{row['system']}_vs_{row['baseline']}"
                seed_pairs[key] = float(row["diff"])
                diff_by_pair.setdefault(key, []).append(float(row["diff"]))

            per_seed.append({"seed": seed, "systems": seed_systems, "top1_diff": seed_pairs})

    retrieval_deterministic = all(fp == fingerprints[0] for fp in fingerprints)

    return {
        "benchmark": benchmark_name,
        "version": version,
        "embedder": embedder_name,
        "seeds": seeds,
        "n_bootstrap": n_bootstrap,
        "isolation": "subprocess-per-seed",
        "retrieval_deterministic": retrieval_deterministic,
        "per_seed": per_seed,
        "cross_seed": {
            "top1_accuracy": {name: _summarize(vals) for name, vals in top1_by_system.items()},
            "hit_at_k": {name: _summarize(vals) for name, vals in hit_by_system.items()},
            "top1_diff": {pair: _summarize(vals) for pair, vals in diff_by_pair.items()},
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    det = report["retrieval_deterministic"]
    lines = [
        "# Realistic Replay — Multi-Seed Robustness Sweep",
        "",
        f"Dataset: `{report['benchmark']}` v{report['version']} · "
        f"embedder: `{report['embedder']}` · seeds: {report['seeds']} · "
        f"bootstrap n={report['n_bootstrap']} · isolation: {report['isolation']}.",
        "",
        (
            "**Retrieval determinism (this run):** "
            + (
                "✅ per-query top-1 outcomes were identical across all seeds in this "
                "sweep — point estimates did not move; only bootstrap CI bounds jittered."
                if det
                else "⚠️ per-query outcomes differed across seeds in this sweep — a "
                "near-tie query flipped (see table)."
            )
        ),
        "",
        (
            "**Caveat — near-deterministic, not bit-stable.** The RNG seed moves nothing, "
            "but the engine stamps encode/retrieve with real wall-clock time and ACT-R "
            "decay tracks `now - encoded_at`. Each seed runs in its own subprocess launched "
            "at a slightly different instant, so a query at a numerical tie can flip *even "
            "with subprocess isolation*. A fresh `make bench-multiseed` therefore reports "
            "`retrieval_deterministic=True` most of the time but occasionally `False` "
            "(observed cross-seed stdev up to ~0.0025), and the absolute mean drifts a "
            "little across sweeps. This variance is timing-driven, sits well inside the "
            "bootstrap CIs, and does not change the AFT-vs-baseline conclusion. See "
            "`docs/research/08_limitations.md` §2.9."
        ),
        "",
        "## Cross-seed `top1_accuracy`",
        "",
        "| System | mean | stdev | min | max | spread |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in report["cross_seed"]["top1_accuracy"].items():
        lines.append(
            f"| `{name}` | {s['mean']:.4f} | {s['stdev']:.4f} | "
            f"{s['min']:.4f} | {s['max']:.4f} | {s['spread']:.4f} |"
        )
    diffs = report["cross_seed"]["top1_diff"]
    if diffs:
        lines.extend(
            [
                "",
                "## Cross-seed top-1 Δ (vs baseline)",
                "",
                "| Comparison | mean Δ | stdev | min | max | spread |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for pair, s in diffs.items():
            lines.append(
                f"| `{pair}` | {s['mean']:+.4f} | {s['stdev']:.4f} | "
                f"{s['min']:+.4f} | {s['max']:+.4f} | {s['spread']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], *, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed robustness sweep (realistic replay).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--systems",
        type=lambda value: [item.strip() for item in value.split(",") if item.strip()],
        default=["aft", "naive_cosine"],
    )
    parser.add_argument(
        "--seeds",
        type=lambda value: [int(item) for item in value.split(",") if item.strip()],
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument(
        "--embedder",
        type=str,
        default="hash",
        choices=["hash", "sbert-bge", "sbert-mini", "e5-small-v2", "multilingual-e5-small"],
    )
    args = parser.parse_args()

    report = run_multiseed(
        dataset_path=args.dataset,
        seeds=args.seeds,
        systems=args.systems,
        top_k=args.top_k,
        n_bootstrap=args.n_bootstrap,
        embedder_name=args.embedder,
    )
    write_report(report, out_json=args.out_json, out_md=args.out_md)
    print(
        f"multiseed sweep complete: embedder={report['embedder']} "
        f"seeds={report['seeds']} retrieval_deterministic={report['retrieval_deterministic']}"
    )


if __name__ == "__main__":
    main()
