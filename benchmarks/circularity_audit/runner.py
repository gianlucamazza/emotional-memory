"""Addendum U — circularity audit of realistic_recall_v2 (Hu1/Hu2).

Quantifies how AFT-favorable the benchmark is by construction. For each query it
computes, from the committed data + the headline embedder (no LLM):

- ``cosine_solvable``: the gold memory is the ``naive_cosine`` top-1 (semantics
  alone wins → affect cannot add value);
- ``affect_separating``: in (valence, arousal) space the gold is strictly the
  closest scenario candidate to the query ``state`` (Euclidean; ties → False).

It then partitions the 200 queries into a 2x2 (``cosine_solvable`` x
``affect_separating``), reports the AFT-cosine top-1 Δ per cell with bootstrap CIs,
and the contingency of ``affect_separating`` vs the author's ``challenge_type``.

Pre-registration: ``benchmarks/preregistration_addendum_u_circularity_audit.md``.

Usage::

    make bench-circularity-audit
    uv run python -m benchmarks.circularity_audit.runner --limit-scenarios 2   # quick check
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from benchmarks.common.statistics import DEFAULT_N_BOOTSTRAP, ci_payload, paired_bootstrap_diff
from benchmarks.realistic.runner import (
    ROOT,
    ReplayDataset,
    ReplayScenario,
    _build_embedder,
    _make_adapter,
    load_dataset,
)

DEFAULT_DATASET = ROOT / "benchmarks" / "datasets" / "realistic_recall_v2.json"
_HERE = Path(__file__).parent
DEFAULT_OUT_JSON = _HERE / "results.json"
DEFAULT_OUT_MD = _HERE / "results.md"


def _affect_separating(scenario: ReplayScenario, expected_ids: list[str], state: Any) -> bool:
    """True iff the gold memory is strictly the affect-closest candidate to the query state."""
    if state is None or not expected_ids:
        return False
    candidates = [e for s in scenario.sessions for e in s.events]
    gold = set(expected_ids)

    def dist(ev: Any) -> float:
        return math.hypot(ev.valence - state.valence, ev.arousal - state.arousal)

    gold_d = min((dist(e) for e in candidates if e.memory_id in gold), default=math.inf)
    other_d = min((dist(e) for e in candidates if e.memory_id not in gold), default=math.inf)
    return gold_d < other_d


def _run_system(
    system_name: str, dataset: ReplayDataset, *, workdir: Path, embedder_name: str
) -> dict[str, bool]:
    """Per-query top1_hit for one system, keyed by query_id."""
    embedder = _build_embedder(embedder_name)
    adapter = _make_adapter(system_name, workdir=workdir, embedder=embedder)
    adapter.reset()
    out: dict[str, bool] = {}
    for scenario in tqdm(dataset.scenarios, desc=system_name, unit="scenario"):
        alias_to_actual: dict[str, str] = {}
        for session in scenario.sessions:
            adapter.begin_session(session.session_id)
            for event in session.events:
                actual = adapter.encode(
                    memory_alias=event.memory_id,
                    content=event.content,
                    valence=event.valence,
                    arousal=event.arousal,
                    metadata=event.metadata,
                )
                alias_to_actual[event.memory_id] = actual
            for query in session.queries:
                top_k = query.top_k or dataset.default_top_k
                retrieved = adapter.retrieve(
                    query.query,
                    top_k=top_k,
                    valence=None if query.state is None else query.state.valence,
                    arousal=None if query.state is None else query.state.arousal,
                )
                expected = {
                    alias_to_actual[m] for m in query.expected_memory_ids if m in alias_to_actual
                }
                ids = [item.id for item in retrieved]
                out[query.query_id] = bool(ids) and ids[0] in expected
            adapter.end_session()
    adapter.close()
    return out


_CELL_FAVORABLE = "affect_only_can_help"  # not cosine-solvable AND affect-separating
_CELL_NEUTRAL = "neutral"  # everything else


def _cell(cosine_solvable: bool, affect_separating: bool) -> str:
    return _CELL_FAVORABLE if (not cosine_solvable and affect_separating) else _CELL_NEUTRAL


def _delta(rows: list[dict[str, Any]], *, n_bootstrap: int, seed: int) -> dict[str, Any]:
    aft = [float(r["aft_hit"]) for r in rows]
    cos = [float(r["cosine_hit"]) for r in rows]
    if not rows:
        return {"n": 0}
    diff, lo, hi, p = paired_bootstrap_diff(aft, cos, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "n": len(rows),
        "aft_top1": round(sum(aft) / len(aft), 4),
        "cosine_top1": round(sum(cos) / len(cos), 4),
        "delta": ci_payload(round(diff, 4), round(lo, 4), round(hi, 4), n_bootstrap=n_bootstrap),
        "p_two_sided": round(p, 4),
    }


def run(
    dataset: ReplayDataset, *, workdir: Path, embedder_name: str, n_bootstrap: int, seed: int
) -> dict[str, Any]:
    aft = _run_system("aft", dataset, workdir=workdir, embedder_name=embedder_name)
    cosine = _run_system("naive_cosine", dataset, workdir=workdir, embedder_name=embedder_name)

    per_query: list[dict[str, Any]] = []
    for scenario in dataset.scenarios:
        for session in scenario.sessions:
            for query in session.queries:
                qid = query.query_id
                if qid not in aft or qid not in cosine:
                    continue
                cosine_solvable = cosine[qid]
                affect_sep = _affect_separating(scenario, query.expected_memory_ids, query.state)
                per_query.append(
                    {
                        "query_id": qid,
                        "challenge_type": query.challenge_type,
                        "aft_hit": aft[qid],
                        "cosine_hit": cosine_solvable,
                        "affect_separating": affect_sep,
                        "cell": _cell(cosine_solvable, affect_sep),
                    }
                )

    n = len(per_query)
    fav = [r for r in per_query if r["cell"] == _CELL_FAVORABLE]
    neu = [r for r in per_query if r["cell"] == _CELL_NEUTRAL]

    # 2x2 partition counts
    grid: dict[str, int] = defaultdict(int)
    for r in per_query:
        grid[f"cosine_solvable={r['cosine_hit']},affect_separating={r['affect_separating']}"] += 1

    # Hu2: affect_separating x challenge_type contingency
    contingency: dict[str, dict[str, int]] = defaultdict(lambda: {"separating": 0, "not": 0})
    for r in per_query:
        contingency[r["challenge_type"]]["separating" if r["affect_separating"] else "not"] += 1

    d_fav = _delta(fav, n_bootstrap=n_bootstrap, seed=seed)
    d_neu = _delta(neu, n_bootstrap=n_bootstrap, seed=seed)
    d_all = _delta(per_query, n_bootstrap=n_bootstrap, seed=seed)

    # Hu1 verdict: advantage concentrated in favorable cell + neutral Δ CI includes 0
    hu1_pass = bool(
        d_fav.get("delta", {}).get("point", 0) > d_neu.get("delta", {}).get("point", 0)
        and d_neu.get("delta", {}).get("ci_lower", 0)
        <= 0
        <= d_neu.get("delta", {}).get("ci_upper", 0)
    )

    return {
        "benchmark": "circularity_audit_u",
        "dataset": dataset.name,
        "version": dataset.version,
        "embedder": embedder_name,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "n_queries": n,
        "favorable_fraction": round(len(fav) / n, 4) if n else float("nan"),
        "partition_2x2": dict(grid),
        "by_cell": {"affect_only_can_help": d_fav, "neutral": d_neu, "overall": d_all},
        "hu1_advantage_concentrated_in_favorable": hu1_pass,
        "hu2_affect_separating_by_challenge_type": {k: dict(v) for k, v in contingency.items()},
        "per_query": per_query,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    fav, neu, allc = (
        report["by_cell"]["affect_only_can_help"],
        report["by_cell"]["neutral"],
        report["by_cell"]["overall"],
    )

    def _row(label: str, d: dict[str, Any]) -> str:
        if not d.get("n"):
            return f"| {label} | 0 | — | — | — | — |"
        dd = d["delta"]
        ci = f"[{dd['ci_lower']:+.4f}, {dd['ci_upper']:+.4f}]"
        return (
            f"| {label} | {d['n']} | {d['aft_top1']:.4f} | {d['cosine_top1']:.4f} | "
            f"{dd['point']:+.4f} {ci} | {d['p_two_sided']:.4f} |"
        )

    hu1 = report["hu1_advantage_concentrated_in_favorable"]
    hu1_verdict = (
        "✅ PASS — benchmark is AFT-favorable by construction"
        if hu1
        else "✗ FAIL — advantage extends to the neutral subset"
    )
    meta = (
        f"Dataset: `{report['dataset']}` v{report['version']} · "
        f"embedder: `{report['embedder']}` · N={report['n_queries']} · "
        f"bootstrap n={report['n_bootstrap']} · seed={report['seed']}."
    )
    lines = [
        "# Circularity Audit of realistic_recall_v2 (Addendum U)",
        "",
        meta,
        "",
        "Cells computed from data + the headline embedder, **not** from the author's "
        "`challenge_type`. `affect_only_can_help` = `not cosine-solvable AND affect-separating`.",
        "",
        f"**AFT-favorable fraction:** {report['favorable_fraction']:.1%} of queries "
        f"({report['by_cell']['affect_only_can_help'].get('n', 0)}/{report['n_queries']}).",
        "",
        "## AFT-cosine top-1 Δ by cell",
        "",
        "| Cell | N | AFT | cosine | Δ [95% CI] | p |",
        "|---|---:|---:|---:|---|---:|",
        _row("affect_only_can_help", fav),
        _row("neutral", neu),
        _row("overall", allc),
        "",
        f"**Hu1** (advantage concentrated in favorable cell + neutral Δ CI includes 0): "
        f"{hu1_verdict}.",
        "",
        "## 2x2 partition",
        "",
        "| cosine_solvable | affect_separating | N |",
        "|---|---|---:|",
    ]
    for key, cnt in sorted(report["partition_2x2"].items()):
        cs = key.split(",")[0].split("=")[1]
        af = key.split(",")[1].split("=")[1]
        lines.append(f"| {cs} | {af} | {cnt} |")
    lines += [
        "",
        "## Hu2 — affect-separating vs author `challenge_type`",
        "",
        "| challenge_type | affect-separating | not-separating |",
        "|---|---:|---:|",
    ]
    for ct, v in sorted(report["hu2_affect_separating_by_challenge_type"].items()):
        lines.append(f"| {ct} | {v['separating']} | {v['not']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Addendum U — circularity audit of v2.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--embedder", type=str, default="sbert-bge")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-scenarios", type=int, default=None)
    args = parser.parse_args()

    import tempfile

    dataset = load_dataset(args.dataset)
    if args.limit_scenarios is not None:
        dataset = dataset.model_copy(
            update={"scenarios": dataset.scenarios[: args.limit_scenarios]}
        )

    with tempfile.TemporaryDirectory(prefix="emotional-memory-audit-") as tmp:
        report = run(
            dataset,
            workdir=Path(tmp),
            embedder_name=args.embedder,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out_md.write_text(_render_markdown(report), encoding="utf-8")
    print(
        f"circularity audit complete: favorable={report['favorable_fraction']:.1%} "
        f"Hu1={'PASS' if report['hu1_advantage_concentrated_in_favorable'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
