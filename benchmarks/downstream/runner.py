"""A3 downstream task — encode → retrieve → generate → judge (Addendum R, Hr1).

Measures whether AFT's retrieval-ranking advantage on the affect-discriminative
``realistic_recall_v2`` regime converts to **downstream answer quality** once an
LLM generator consumes the retrieved memories, versus a pure-cosine baseline.

Pre-registration: ``benchmarks/preregistration_addendum_r_downstream.md``.

Both systems share the embedder, ``top_k``, answer-generation prompt, LLM and
judge prompt; only the retrieval stage differs (full 6-signal AFT vs embedding
cosine). Gold answer for a query = the concatenated ``content`` of its
``expected_memory_ids``. The LLM-as-judge labels each generated answer
CORRECT/WRONG against that gold.

Environment variables (same convention as the LLM test suite):
  EMOTIONAL_MEMORY_LLM_API_KEY   required for generation + judge
  EMOTIONAL_MEMORY_LLM_BASE_URL  default https://api.openai.com/v1
  EMOTIONAL_MEMORY_LLM_MODEL     default gpt-4o-mini (.env pins gpt-5-mini)

Usage::

    make bench-a3                                          # full confirmatory run
    uv run python -m benchmarks.downstream.runner --no-judge --limit-scenarios 2  # dry run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from tqdm import tqdm

from benchmarks.common.statistics import (
    DEFAULT_N_BOOTSTRAP,
    ci_payload,
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap_diff,
)
from benchmarks.locomo.adapters.base import _ANSWER_SYSTEM, call_llm
from benchmarks.locomo.scoring import build_judge_prompt, parse_judge_response, token_f1
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
DEFAULT_SYSTEMS = ["aft", "naive_cosine"]


def _gold_answer(scenario: ReplayScenario, expected_ids: list[str]) -> str:
    """Concatenate the content of the expected memories (the reference answer)."""
    by_id = {
        event.memory_id: event.content for session in scenario.sessions for event in session.events
    }
    parts = [by_id[mid] for mid in expected_ids if mid in by_id]
    return " / ".join(parts)


def _run_system(
    system_name: str,
    dataset: ReplayDataset,
    *,
    workdir: Path,
    embedder_name: str,
    top_k_override: int | None,
    judge: bool,
) -> dict[str, dict[str, Any]]:
    """Run encode→retrieve→generate(→judge) for one system. Keyed by query_id."""
    embedder = _build_embedder(embedder_name)
    adapter = _make_adapter(system_name, workdir=workdir, embedder=embedder)
    adapter.reset()
    default_top_k = top_k_override or dataset.default_top_k
    out: dict[str, dict[str, Any]] = {}

    for scenario in tqdm(dataset.scenarios, desc=f"{system_name}", unit="scenario"):
        for session in scenario.sessions:
            adapter.begin_session(session.session_id)
            for event in session.events:
                adapter.encode(
                    memory_alias=event.memory_id,
                    content=event.content,
                    valence=event.valence,
                    arousal=event.arousal,
                    metadata=event.metadata,
                )
            for query in session.queries:
                effective_top_k = query.top_k or default_top_k
                retrieved = adapter.retrieve(
                    query.query,
                    top_k=effective_top_k,
                    valence=None if query.state is None else query.state.valence,
                    arousal=None if query.state is None else query.state.arousal,
                )
                retrieved_ids = [item.id for item in retrieved]
                gold = _gold_answer(scenario, query.expected_memory_ids)
                # top1_hit needs alias mapping; AFTReplayAdapter returns actual ids.
                # We compare on content instead: gold-bearing item ranked first.
                gold_contents = {
                    event.content
                    for s in scenario.sessions
                    for event in s.events
                    if event.memory_id in query.expected_memory_ids
                }
                top1_hit = bool(retrieved) and retrieved[0].text in gold_contents

                context = "\n".join(f"- {item.text}" for item in retrieved)
                prompt = f"Conversation excerpts:\n{context}\n\nQuestion: {query.query}\n\nAnswer:"
                answer = call_llm(prompt, system=_ANSWER_SYSTEM) if judge else "(dry-run: no LLM)"
                f1 = token_f1(answer, gold)
                judge_correct: bool | None = None
                if judge:
                    judge_prompt = build_judge_prompt(query.query, gold, answer)
                    judge_correct = parse_judge_response(call_llm(judge_prompt))

                out[query.query_id] = {
                    "query_id": query.query_id,
                    "challenge_type": query.challenge_type,
                    "scenario_id": scenario.scenario_id,
                    "top1_hit": top1_hit,
                    "f1": f1,
                    "judge_correct": judge_correct,
                    "answer": answer,
                    "gold": gold,
                    "retrieved_ids": retrieved_ids,
                }
            adapter.end_session()
    adapter.close()
    return out


def _paired(
    a: dict[str, dict[str, Any]],
    b: dict[str, dict[str, Any]],
    field: str,
) -> tuple[list[float], list[float]]:
    keys = [k for k in a if k in b]
    return (
        [float(a[k][field]) for k in keys],
        [float(b[k][field]) for k in keys],
    )


def _hypothesis(
    name: str,
    metric: str,
    aft: dict[str, dict[str, Any]],
    cos: dict[str, dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
    judge: bool,
) -> dict[str, Any]:
    if metric == "judge_correct" and not judge:
        return {"name": name, "metric": metric, "skipped": "no-judge dry run"}
    a, b = _paired(aft, cos, metric)
    diff, lo, hi, p = paired_bootstrap_diff(a, b, n_bootstrap=n_bootstrap, seed=seed)
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x > y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y > x)
    mcnemar = mcnemar_exact(only_a, only_b)
    return {
        "name": name,
        "metric": metric,
        "n": len(a),
        "aft_mean": round(sum(a) / len(a), 4) if a else float("nan"),
        "cosine_mean": round(sum(b) / len(b), 4) if b else float("nan"),
        "delta": ci_payload(round(diff, 4), round(lo, 4), round(hi, 4), n_bootstrap=n_bootstrap),
        "p_two_sided": round(p, 4),
        "mcnemar_p": round(mcnemar, 4),
        "discordant": {"aft_only": only_a, "cosine_only": only_b},
    }


def run(
    dataset: ReplayDataset,
    *,
    workdir: Path,
    embedder_name: str,
    top_k_override: int | None,
    n_bootstrap: int,
    seed: int,
    judge: bool,
) -> dict[str, Any]:
    per_system = {
        name: _run_system(
            name,
            dataset,
            workdir=workdir,
            embedder_name=embedder_name,
            top_k_override=top_k_override,
            judge=judge,
        )
        for name in DEFAULT_SYSTEMS
    }
    aft, cos = per_system["aft"], per_system["naive_cosine"]

    hr1 = _hypothesis(
        "Hr1", "judge_correct", aft, cos, n_bootstrap=n_bootstrap, seed=seed, judge=judge
    )
    hr2 = _hypothesis("Hr2", "f1", aft, cos, n_bootstrap=n_bootstrap, seed=seed, judge=judge)
    ranking = _hypothesis(
        "ranking_ref", "top1_hit", aft, cos, n_bootstrap=n_bootstrap, seed=seed, judge=True
    )

    tests = [h for h in (hr1, hr2) if "skipped" not in h]
    if tests:
        p_holm = holm_bonferroni([h["p_two_sided"] for h in tests])
        for h, padj in zip(tests, p_holm, strict=True):
            h["p_holm"] = round(padj, 4)
            # one-sided PASS: positive delta whose CI excludes 0 and survives Holm
            h["pass"] = bool(
                h["delta"]["point"] > 0 and h["delta"]["ci_lower"] > 0 and h["p_holm"] < 0.05
            )

    return {
        "benchmark": "downstream_a3",
        "dataset": dataset.name,
        "version": dataset.version,
        "embedder": embedder_name,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "judge_enabled": judge,
        "systems": DEFAULT_SYSTEMS,
        "hypotheses": {"Hr1": hr1, "Hr2": hr2},
        "ranking_reference": ranking,
        "per_query": {"aft": aft, "naive_cosine": cos},
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Downstream A3 — encode→retrieve→generate→judge (Addendum R)",
        "",
        f"Dataset: `{report['dataset']}` v{report['version']} · embedder: "
        f"`{report['embedder']}` · bootstrap n={report['n_bootstrap']} · seed={report['seed']} · "
        f"judge={'on' if report['judge_enabled'] else 'OFF (dry-run)'}.",
        "",
        "Both systems share the generator, LLM and judge; only retrieval differs "
        "(full AFT vs embedding cosine). Gold = content of `expected_memory_ids`.",
        "",
        "## Hypothesis tests",
        "",
        "| Hyp | Metric | N | AFT | cosine | Δ [95% CI] | p | p_holm | McNemar | Result |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for h in (report["hypotheses"]["Hr1"], report["hypotheses"]["Hr2"]):
        if "skipped" in h:
            lines.append(
                f"| **{h['name']}** | {h['metric']} | — | — | — | — | — | — | — | "
                f"_skipped ({h['skipped']})_ |"
            )
            continue
        d = h["delta"]
        verdict = "✅ PASS" if h.get("pass") else "✗ FAIL"
        lines.append(
            f"| **{h['name']}** | {h['metric']} | {h['n']} | {h['aft_mean']:.4f} | "
            f"{h['cosine_mean']:.4f} | {d['point']:+.4f} [{d['ci_lower']:+.4f}, "
            f"{d['ci_upper']:+.4f}] | {h['p_two_sided']:.4f} | "
            f"{h.get('p_holm', float('nan')):.4f} | {h['mcnemar_p']:.4f} | **{verdict}** |"
        )
    r = report["ranking_reference"]
    rd = r["delta"]
    lines += [
        "",
        "## Ranking reference (retrieval top-1, not a hypothesis)",
        "",
        f"AFT top1 {r['aft_mean']:.4f} vs cosine {r['cosine_mean']:.4f} · "
        f"Δ {rd['point']:+.4f} [{rd['ci_lower']:+.4f}, {rd['ci_upper']:+.4f}] · "
        "shows whether the ranking edge converts to answer quality above.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="A3 downstream generate→judge (Addendum R).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--embedder", type=str, default="sbert-bge")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-scenarios", type=int, default=None)
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip all LLM calls (pipeline dry-run; no generation, no judge).",
    )
    args = parser.parse_args()

    import tempfile

    dataset = load_dataset(args.dataset)
    if args.limit_scenarios is not None:
        dataset = dataset.model_copy(
            update={"scenarios": dataset.scenarios[: args.limit_scenarios]}
        )

    with tempfile.TemporaryDirectory(prefix="emotional-memory-a3-") as tmp:
        report = run(
            dataset,
            workdir=Path(tmp),
            embedder_name=args.embedder,
            top_k_override=args.top_k,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            judge=not args.no_judge,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out_md.write_text(_render_markdown(report), encoding="utf-8")
    hr1 = report["hypotheses"]["Hr1"]
    verdict = hr1.get("skipped") or ("PASS" if hr1.get("pass") else "FAIL")
    print(f"downstream A3 complete: judge={not args.no_judge} Hr1={verdict}")


if __name__ == "__main__":
    main()
