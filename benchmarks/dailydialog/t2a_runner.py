"""Addendum T2A — retrieve-time query appraisal on naturalistic dialogue (Ht2a).

Tests whether appraising the query text (direct-VAD, no oracle) recovers an
advantage on DailyDialog affect-conditioned recall, where stale-state AFT tied
cosine in Hk1 (Addendum K). Three arms on the same personas/embedder/top_k;
only the query-affect source differs:

  - ``naive_cosine``         — NaiveCosineDailyDialogAdapter (no affect). Baseline.
  - ``aft``                  — AFTDailyDialogAdapter (oracle session PAD at encode,
                               leftover runtime state at retrieve). = Hk1's AFT arm.
  - ``aft_query_appraised``  — AFT with the query's affect appraised by DIRECT_VAD_SCHEMA
                               and passed via the public ``query_affect`` API.

Pre-registration: ``benchmarks/preregistration_addendum_t2a_naturalistic_query_appraisal.md``.

Usage::

    make bench-t2a-dailydialog                                       # full run (needs API key)
    uv run python -m benchmarks.dailydialog.t2a_runner --dry-run     # quick check (5 personas)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from benchmarks.dailydialog.adapters.aft import AFTDailyDialogAdapter
from benchmarks.dailydialog.adapters.aft_query_appraised import (
    AFTQueryAppraisedDailyDialogAdapter,
)
from benchmarks.dailydialog.adapters.base import DailyDialogAdapter, PersonaRunResult
from benchmarks.dailydialog.adapters.naive_cosine import NaiveCosineDailyDialogAdapter
from benchmarks.dailydialog.dataset import (
    DailyDialogPersonaDataset,
    PersonaQuery,
    load_personas,
)
from benchmarks.dailydialog.query_generator import ALL_QUERY_TYPES
from benchmarks.dailydialog.runner import _apply_holm, _compute_stats
from benchmarks.dailydialog.scoring import hit_at_k, top1_correct

_HERE = Path(__file__).parent
DEFAULT_PERSONA_FILE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "datasets"
    / "dailydialog_personas_v1.json"
)
DEFAULT_OUT_JSON = _HERE / "t2a_results.json"
DEFAULT_OUT_MD = _HERE / "t2a_results.md"
DEFAULT_OUT_PROTOCOL = _HERE / "t2a_results.protocol.json"

DEFAULT_N_BOOTSTRAP = 10_000
PRIMARY = "aft_query_appraised"
BASELINE = "naive_cosine"
REFERENCE = "aft"
SYSTEMS = [BASELINE, REFERENCE, PRIMARY]
DIRECTIONAL_TYPES = [
    "emotion_state_recall",
    "affect_conditioned_content",
    "affective_trajectory",
]


def _make_adapter(name: str, *, embedder_name: str) -> DailyDialogAdapter:
    if name == "aft":
        return AFTDailyDialogAdapter(embedder_name=embedder_name)
    if name == "naive_cosine":
        return NaiveCosineDailyDialogAdapter(embedder_name=embedder_name)
    if name == "aft_query_appraised":
        return AFTQueryAppraisedDailyDialogAdapter(embedder_name=embedder_name)
    raise ValueError(f"Unknown system: {name!r}")


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    arr = np.corrcoef(np.asarray(xs), np.asarray(ys))
    return float(arr[0, 1])


def run_benchmark(
    dataset: DailyDialogPersonaDataset,
    *,
    embedder_name: str = "multilingual-e5-small",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    verbose: bool = True,
    limit_personas: int | None = None,
) -> dict[str, Any]:
    personas = dataset.personas
    if limit_personas is not None:
        personas = personas[:limit_personas]

    if verbose:
        print(f"Running {len(SYSTEMS)} arms on {len(personas)} personas …")

    # {system: {query_type | "aggregate": [0/1, ...]}}
    top1: dict[str, dict[str, list[int]]] = {
        s: {qt: [] for qt in [*ALL_QUERY_TYPES, "aggregate"]} for s in SYSTEMS
    }
    hitk: dict[str, dict[str, list[int]]] = {
        s: {qt: [] for qt in [*ALL_QUERY_TYPES, "aggregate"]} for s in SYSTEMS
    }

    adapters = {s: _make_adapter(s, embedder_name=embedder_name) for s in SYSTEMS}
    appraised = adapters[PRIMARY]
    assert isinstance(appraised, AFTQueryAppraisedDailyDialogAdapter)

    # Diagnostic: appraised query affect vs target-session oracle PAD.
    diag_qv: list[float] = []
    diag_qa: list[float] = []
    diag_tv: list[float] = []
    diag_ta: list[float] = []

    for persona_idx, persona in enumerate(personas):
        if verbose and persona_idx % 20 == 0:
            print(f"  persona {persona_idx}/{len(personas)} …")
        session_pad = {s.session_id: (s.valence, s.arousal) for s in persona.sessions}
        for sys_name, adapter in adapters.items():
            run_results: list[PersonaRunResult] = adapter.run_persona(persona)
            for row in run_results:
                qt = row["query_type"]
                retrieved = row["retrieved_session_ids"]
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
                top1[sys_name][qt].append(c1)
                top1[sys_name]["aggregate"].append(c1)
                hitk[sys_name][qt].append(ck)
                hitk[sys_name]["aggregate"].append(ck)

                if sys_name == PRIMARY:
                    ca = appraised.appraised_affect.get(row["query_text"])
                    tgt = session_pad.get(row["target_session_id"])
                    if ca is not None and tgt is not None:
                        diag_qv.append(ca.valence)
                        diag_qa.append(ca.arousal)
                        diag_tv.append(tgt[0])
                        diag_ta.append(tgt[1])

    if verbose:
        print("Computing statistics …")

    system_results: list[dict[str, Any]] = []
    for s in SYSTEMS:
        agg = top1[s]["aggregate"]
        n = len(agg)
        system_results.append(
            {
                "system": s,
                "n_queries": n,
                "top1_accuracy": sum(agg) / n if n else 0.0,
                "hit_at_k": sum(hitk[s]["aggregate"]) / n if n else 0.0,
                "per_type": [
                    {
                        "query_type": qt,
                        "n": len(top1[s][qt]),
                        "top1_accuracy": (
                            sum(top1[s][qt]) / len(top1[s][qt]) if top1[s][qt] else 0.0
                        ),
                    }
                    for qt in ALL_QUERY_TYPES
                ],
            }
        )

    # Pairwise comparisons.
    def _compare(a_sys: str, b_sys: str, *, holm_family: bool) -> dict[str, Any]:
        keys = ["aggregate", *ALL_QUERY_TYPES] if holm_family else ["aggregate"]
        stats_map: dict[str, dict[str, Any]] = {}
        for key in keys:
            a_vec = [float(x) for x in top1[a_sys][key]]
            b_vec = [float(x) for x in top1[b_sys][key]]
            if not a_vec or len(a_vec) != len(b_vec):
                continue
            stats_map[key] = _compute_stats(a_vec, b_vec, n_bootstrap=n_bootstrap, seed=seed)
        if holm_family:
            stats_map = _apply_holm(stats_map)
        return {"system": a_sys, "baseline": b_sys, "stats": stats_map}

    comparisons = [
        _compare(PRIMARY, BASELINE, holm_family=True),  # Ht2a primary
        _compare(PRIMARY, REFERENCE, holm_family=False),  # Ht2a-ref (vs stale-state AFT)
        _compare(REFERENCE, BASELINE, holm_family=False),  # Hk1 reproduction
    ]

    primary = comparisons[0]["stats"]
    agg = primary.get("aggregate", {})
    n_types_pass = sum(
        1 for qt in DIRECTIONAL_TYPES if primary.get(qt, {}).get("pass_holm", False)
    )
    ht2a_pass = bool(agg.get("pass_holm", False)) and n_types_pass >= 2

    return {
        "benchmark": "t2a_dailydialog_query_appraisal",
        "pre_registration": (
            "benchmarks/preregistration_addendum_t2a_naturalistic_query_appraisal.md"
        ),
        "dataset_version": dataset.version,
        "n_personas": len(personas),
        "embedder": embedder_name,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "systems": system_results,
        "pairwise_comparisons": comparisons,
        "diagnostic": {
            "n": len(diag_qv),
            "valence_r": _pearson(diag_qv, diag_tv),
            "arousal_r": _pearson(diag_qa, diag_ta),
        },
        "ht2a_pass": ht2a_pass,
        "n_directional_types_pass": n_types_pass,
    }


def _verdict(st: dict[str, Any]) -> str:
    d = float(st.get("delta", 0))
    p = float(st.get("p_holm", st.get("p_bootstrap_onetail", 1)))
    tag = (
        "PASS"
        if (
            st.get("pass_holm")
            or (st.get("delta", 0) > 0 and st.get("ci_lower", -1) > 0 and p < 0.05)
        )
        else "FAIL"
    )
    return f"{tag} Δ={d:+.3f} p={p:.3f}"


def write_results(
    results: dict[str, Any],
    *,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
    out_protocol: Path | None = DEFAULT_OUT_PROTOCOL,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = [
        "# Addendum T2A — Retrieve-time Query Appraisal on DailyDialog (Ht2a)",
        "",
        f"**Personas:** {results['n_personas']}  **Embedder:** `{results['embedder']}`  "
        f"**Bootstrap:** n={results['n_bootstrap']}, seed={results['seed']}",
        "",
        "## Aggregate",
        "",
        "| Arm | N | top1_accuracy | hit@k |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {s['system']} | {s['n_queries']} | {s['top1_accuracy']:.3f} | {s['hit_at_k']:.3f} |"
        for s in results["systems"]
    ]
    lines += ["", "## Contrasts", ""]
    labels = {
        0: "aft_query_appraised vs naive_cosine (Ht2a, Holm family)",
        1: "aft_query_appraised vs aft (Ht2a-ref)",
        2: "aft vs naive_cosine (Hk1 reproduction)",
    }
    for i, cmp in enumerate(results["pairwise_comparisons"]):
        lines += [f"### {labels.get(i, cmp['system'] + ' vs ' + cmp['baseline'])}", ""]
        lines += ["| Key | Δ | CI | p_one | p_holm | Verdict |", "|---|---|---|---|---|---|"]
        for key, st in cmp["stats"].items():
            lines.append(
                f"| {key} | {st.get('delta', 0):+.3f} "
                f"| [{st.get('ci_lower', 0):+.3f}, {st.get('ci_upper', 0):+.3f}] "
                f"| {st.get('p_bootstrap_onetail', 1):.3f} "
                f"| {st.get('p_holm', float('nan')):.3f} | {_verdict(st)} |"
            )
        lines.append("")

    diag = results["diagnostic"]
    _agg_verdict = _verdict(results["pairwise_comparisons"][0]["stats"].get("aggregate", {}))
    lines += [
        "## Diagnostic — appraised query affect vs target-session oracle PAD",
        "",
        f"N={diag['n']}  valence r={diag['valence_r']:.3f}  arousal r={diag['arousal_r']:.3f}",
        "",
        "## Ht2a Decision",
        "",
        f"Aggregate (appraised vs cosine): {_agg_verdict}",
        f"Directional types passing Holm: {results['n_directional_types_pass']}/3",
        "",
        f"**Ht2a verdict: {'PASS' if results['ht2a_pass'] else 'FAIL'}**",
        "",
        "Decision rule: see "
        "`benchmarks/preregistration_addendum_t2a_naturalistic_query_appraisal.md`.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    if out_protocol is not None:
        out_protocol.write_text(
            json.dumps(
                {
                    "benchmark": results["benchmark"],
                    "pre_registration": results["pre_registration"],
                    "arms": SYSTEMS,
                    "primary_contrast": f"{PRIMARY} vs {BASELINE}",
                    "primary_metric": "top1_accuracy",
                    "n_personas": results["n_personas"],
                    "embedder": results["embedder"],
                    "n_bootstrap": results["n_bootstrap"],
                    "seed": results["seed"],
                    "decision_rule": (
                        "PASS iff p_holm<0.05 AND delta>0 AND ci_lower>0 on aggregate, "
                        "and >=2/3 directional types also PASS individually"
                    ),
                    "ht2a_pass": results["ht2a_pass"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Addendum T2A DailyDialog query-appraisal benchmark"
    )
    p.add_argument("--personas", type=Path, default=DEFAULT_PERSONA_FILE)
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
    p.add_argument("--dry-run", action="store_true", help="Limit to 5 personas")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = load_personas(args.personas)
    results = run_benchmark(
        dataset,
        embedder_name=args.embedder,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        verbose=not args.quiet,
        limit_personas=5 if args.dry_run else None,
    )
    write_results(
        results, out_json=args.out_json, out_md=args.out_md, out_protocol=args.out_protocol
    )
    print(f"\nResults written to {args.out_json}")
    for s in results["systems"]:
        print(f"  {s['system']}: top1={s['top1_accuracy']:.3f}  hit@k={s['hit_at_k']:.3f}")
    agg = results["pairwise_comparisons"][0]["stats"].get("aggregate", {})
    print(
        f"\nHt2a aggregate (appraised vs cosine): Δ={agg.get('delta', 0):+.3f} "
        f"[{agg.get('ci_lower', 0):+.3f}, {agg.get('ci_upper', 0):+.3f}] "
        f"p_holm={agg.get('p_holm', 1):.3f} → {'PASS' if results['ht2a_pass'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
