"""Addendum T — retrieve-time query appraisal vs oracle state-injection (Ht1/Ht2).

Tests whether appraising the query text (direct-VAD) and injecting that as the
retrieve-time affective state can substitute for the oracle ``query.state`` that
drives the benchmark's AFT advantage — i.e. whether the headline +0.205 is
production-reachable without preset affect.

Three arms on realistic_recall_v2 (same embedder/top_k; only the query-affect source differs):
  - ``cosine``               — NaiveCosineReplayAdapter (no affect)
  - ``aft_oracle``           — AFT with oracle ``query.state`` (the upper bound = headline)
  - ``aft_query_appraised``  — AFT with the query's affect appraised by DIRECT_VAD_SCHEMA

Pre-registration: ``benchmarks/preregistration_addendum_t_query_appraisal.md``.

Usage::

    make bench-query-appraisal                                          # full run (needs API key)
    uv run python -m benchmarks.query_appraisal.runner --limit-scenarios 2   # quick check
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

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
from emotional_memory import DIRECT_VAD_SCHEMA
from emotional_memory.appraisal_llm import LLMAppraisalConfig, LLMAppraisalEngine
from emotional_memory.llm_http import OpenAICompatibleLLMConfig, make_httpx_llm

DEFAULT_DATASET = ROOT / "benchmarks" / "datasets" / "realistic_recall_v2.json"
_HERE = Path(__file__).parent
DEFAULT_OUT_JSON = _HERE / "results.json"
DEFAULT_OUT_MD = _HERE / "results.md"


def _affect_separating(scenario: ReplayScenario, expected_ids: list[str], state: Any) -> bool:
    """Addendum U favorable criterion: gold strictly affect-closest to the query state."""
    if state is None or not expected_ids:
        return False
    candidates = [e for s in scenario.sessions for e in s.events]
    gold = set(expected_ids)

    def dist(ev: Any) -> float:
        return math.hypot(ev.valence - state.valence, ev.arousal - state.arousal)

    gold_d = min((dist(e) for e in candidates if e.memory_id in gold), default=math.inf)
    other_d = min((dist(e) for e in candidates if e.memory_id not in gold), default=math.inf)
    return gold_d < other_d


def _appraise_queries(dataset: ReplayDataset) -> dict[str, tuple[float, float]]:
    """Appraise every query text with direct-VAD; returns query_id -> (valence, arousal)."""
    cfg = OpenAICompatibleLLMConfig.from_env()
    if cfg is None:
        raise RuntimeError("EMOTIONAL_MEMORY_LLM_API_KEY not set — cannot appraise queries.")
    engine = LLMAppraisalEngine(
        llm=make_httpx_llm(cfg),
        config=LLMAppraisalConfig(
            cache_size=0, fallback_on_error=True, appraisal_schema=DIRECT_VAD_SCHEMA
        ),
    )
    queries = [q for sc in dataset.scenarios for s in sc.sessions for q in s.queries]
    out: dict[str, tuple[float, float]] = {}
    for q in tqdm(queries, desc="appraise-query", unit="query"):
        ca = engine.appraise(q.query).to_core_affect()
        out[q.query_id] = (ca.valence, ca.arousal)
    return out


def _run_arm(
    arm: str,
    dataset: ReplayDataset,
    *,
    workdir: Path,
    embedder_name: str,
    appraised: dict[str, tuple[float, float]] | None,
) -> dict[str, bool]:
    """Per-query top1_hit for one arm, keyed by query_id.

    arm: 'cosine' | 'aft_oracle' | 'aft_query_appraised'.
    """
    system = "naive_cosine" if arm == "cosine" else "aft"
    embedder = _build_embedder(embedder_name)
    adapter = _make_adapter(system, workdir=workdir, embedder=embedder)
    adapter.reset()
    out: dict[str, bool] = {}
    for scenario in tqdm(dataset.scenarios, desc=arm, unit="scenario"):
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
                if arm == "cosine":
                    val = ar = None
                elif arm == "aft_oracle":
                    val = None if query.state is None else query.state.valence
                    ar = None if query.state is None else query.state.arousal
                else:  # aft_query_appraised
                    assert appraised is not None
                    val, ar = appraised.get(query.query_id, (None, None))
                retrieved = adapter.retrieve(query.query, top_k=top_k, valence=val, arousal=ar)
                expected = {
                    alias_to_actual[m] for m in query.expected_memory_ids if m in alias_to_actual
                }
                ids = [item.id for item in retrieved]
                out[query.query_id] = bool(ids) and ids[0] in expected
            adapter.end_session()
    adapter.close()
    return out


def _paired(
    keys: list[str], a: dict[str, bool], b: dict[str, bool]
) -> tuple[list[float], list[float]]:
    return ([float(a[k]) for k in keys], [float(b[k]) for k in keys])


def _delta(
    keys: list[str], a: dict[str, bool], b: dict[str, bool], *, n_bootstrap: int, seed: int
) -> dict[str, Any]:
    xa, xb = _paired(keys, a, b)
    if not keys:
        return {"n": 0}
    diff, lo, hi, p = paired_bootstrap_diff(xa, xb, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "n": len(keys),
        "a_top1": round(sum(xa) / len(xa), 4),
        "b_top1": round(sum(xb) / len(xb), 4),
        "delta": ci_payload(round(diff, 4), round(lo, 4), round(hi, 4), n_bootstrap=n_bootstrap),
        "p_two_sided": round(p, 4),
    }


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def run(
    dataset: ReplayDataset, *, workdir: Path, embedder_name: str, n_bootstrap: int, seed: int
) -> dict[str, Any]:
    appraised = _appraise_queries(dataset)
    cosine = _run_arm(
        "cosine", dataset, workdir=workdir, embedder_name=embedder_name, appraised=None
    )
    oracle = _run_arm(
        "aft_oracle", dataset, workdir=workdir, embedder_name=embedder_name, appraised=None
    )
    appr = _run_arm(
        "aft_query_appraised",
        dataset,
        workdir=workdir,
        embedder_name=embedder_name,
        appraised=appraised,
    )

    keys = [k for k in cosine if k in oracle and k in appr]
    favorable = []
    qstate: dict[str, Any] = {}
    for sc in dataset.scenarios:
        for s in sc.sessions:
            for q in s.queries:
                qstate[q.query_id] = q.state
                if _affect_separating(sc, q.expected_memory_ids, q.state) and not cosine.get(
                    q.query_id, True
                ):
                    favorable.append(q.query_id)

    appr_vs_cos = _delta(keys, appr, cosine, n_bootstrap=n_bootstrap, seed=seed)
    oracle_vs_cos = _delta(keys, oracle, cosine, n_bootstrap=n_bootstrap, seed=seed)
    appr_vs_oracle = _delta(keys, appr, oracle, n_bootstrap=n_bootstrap, seed=seed)
    fav_appr_vs_cos = _delta(favorable, appr, cosine, n_bootstrap=n_bootstrap, seed=seed)
    fav_oracle_vs_cos = _delta(favorable, oracle, cosine, n_bootstrap=n_bootstrap, seed=seed)

    # Diagnostic D: corr(appraised query affect, oracle state)
    diag_keys = [k for k in keys if qstate.get(k) is not None]
    appr_v = [appraised[k][0] for k in diag_keys]
    appr_a = [appraised[k][1] for k in diag_keys]
    orc_v = [qstate[k].valence for k in diag_keys]
    orc_a = [qstate[k].arousal for k in diag_keys]
    diagnostic = {
        "valence_r": round(_pearson(appr_v, orc_v), 4),
        "arousal_r": round(_pearson(appr_a, orc_a), 4),
    }

    od = oracle_vs_cos.get("delta", {}).get("point", 0.0)
    ad = appr_vs_cos.get("delta", {}).get("point", 0.0)
    recovery_fraction = round(ad / od, 4) if od else float("nan")
    ht1_pass = bool(appr_vs_cos.get("delta", {}).get("ci_lower", 0) > 0)

    return {
        "benchmark": "query_appraisal_t",
        "dataset": dataset.name,
        "version": dataset.version,
        "embedder": embedder_name,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "n_queries": len(keys),
        "arms": {
            "appraised_vs_cosine": appr_vs_cos,
            "oracle_vs_cosine": oracle_vs_cos,
            "appraised_vs_oracle": appr_vs_oracle,
        },
        "favorable_subset": {
            "n": len(favorable),
            "appraised_vs_cosine": fav_appr_vs_cos,
            "oracle_vs_cosine": fav_oracle_vs_cos,
        },
        "recovery_fraction": recovery_fraction,
        "diagnostic_appraised_vs_oracle_state_r": diagnostic,
        "ht1_appraised_beats_cosine": ht1_pass,
    }


def _fmt(d: dict[str, Any]) -> str:
    if not d.get("n"):
        return "n=0"
    dd = d["delta"]
    return (
        f"{d['a_top1']:.4f} vs {d['b_top1']:.4f} · Δ {dd['point']:+.4f} "
        f"[{dd['ci_lower']:+.4f}, {dd['ci_upper']:+.4f}] · p={d['p_two_sided']:.4f}"
    )


def _render_markdown(report: dict[str, Any]) -> str:
    a = report["arms"]
    diag = report["diagnostic_appraised_vs_oracle_state_r"]
    verdict = (
        "✅ PASS — query appraisal beats cosine (production-reachable)"
        if report["ht1_appraised_beats_cosine"]
        else "✗ FAIL — query appraisal does not beat cosine (AFT remains oracle-bound)"
    )
    meta = (
        f"Dataset: `{report['dataset']}` v{report['version']} · "
        f"embedder: `{report['embedder']}` · N={report['n_queries']} · "
        f"bootstrap n={report['n_bootstrap']} · seed={report['seed']}."
    )
    fav = report["favorable_subset"]
    lines = [
        "# Addendum T — retrieve-time query appraisal vs oracle state-injection",
        "",
        meta,
        "",
        "Three arms; only the query-affect source differs. `aft_query_appraised` injects the "
        "query's affect appraised by direct-VAD instead of the oracle `query.state`.",
        "",
        "## Top-1 (full set)",
        "",
        f"- **aft_oracle vs cosine** (upper bound): {_fmt(a['oracle_vs_cosine'])}",
        f"- **aft_query_appraised vs cosine** (Ht1): {_fmt(a['appraised_vs_cosine'])}",
        f"- aft_query_appraised vs aft_oracle (gap): {_fmt(a['appraised_vs_oracle'])}",
        "",
        f"**Recovery fraction** (appraised minus cosine)/(oracle minus cosine): "
        f"**{report['recovery_fraction']}**",
        "",
        f"**Ht1:** {verdict}",
        "",
        "## Affect-favorable subset (Addendum U criterion)",
        "",
        f"- N = {fav['n']}",
        f"- aft_oracle vs cosine: {_fmt(fav['oracle_vs_cosine'])}",
        f"- aft_query_appraised vs cosine: {_fmt(fav['appraised_vs_cosine'])}",
        "",
        "## Diagnostic — corr(appraised query affect, oracle state)",
        "",
        f"- valence r = {diag['valence_r']} · arousal r = {diag['arousal_r']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Addendum T — retrieve-time query appraisal.")
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

    with tempfile.TemporaryDirectory(prefix="emotional-memory-t-") as tmp:
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
        f"Addendum T complete: Ht1={'PASS' if report['ht1_appraised_beats_cosine'] else 'FAIL'} "
        f"recovery_fraction={report['recovery_fraction']}"
    )


if __name__ == "__main__":
    main()
