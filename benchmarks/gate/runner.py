"""Addendum Y runner — query-affect-conditioned gate across four corpora.

The `gated` arm is an exact per-query selection between each corpus's existing
`naive_cosine` and `aft_query_appraised` arms, routed on the appraised query
valence (``abs(valence) < tau`` → cosine, else aft). See
``benchmarks/preregistration_addendum_y_query_affect_gate.md``.

Confirmatory family (Holm m=2, one-tailed):
- **Hg1 (recover):** ES-MemEval `gated > aft_query_appraised`.
- **Hg2 (preserve):** realistic_recall_v2 `gated > naive_cosine`.

Each corpus's `gate_inputs()` supplies aligned per-query (cosine, aft, valence).
Two corpora appraise without an LLM key (keyword fallback) and run in ``--dry-run``;
the curated and DailyDialog corpora require a key and run only in the scored pass.

Usage::

    make bench-y-gate                                  # scored, 4 corpora (needs API key)
    uv run python -m benchmarks.gate.runner --dry-run  # smoke: esmemeval + madial, no LLM
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from benchmarks.common.gate import DEFAULT_TAU, TAU_GRID, gate_report
from benchmarks.common.statistics import holm_bonferroni
from benchmarks.dailydialog.t2a_runner import gate_inputs as _t2a_inputs
from benchmarks.esmemeval.runner import gate_inputs as _esmem_inputs
from benchmarks.madialbench.runner import gate_inputs as _madial_inputs
from benchmarks.query_appraisal.runner import gate_inputs as _curated_inputs

_HERE = Path(__file__).parent
DEFAULT_OUT_JSON = _HERE / "results.json"
DEFAULT_OUT_MD = _HERE / "results.md"
DEFAULT_OUT_PROTOCOL = _HERE / "results.protocol.json"

DEFAULT_N_BOOTSTRAP = 10_000

# (corpus key, gate_inputs, dry_capable) — dry_capable corpora appraise without
# an LLM key (keyword fallback) and run in the no-LLM smoke.
CORPORA: list[tuple[str, Callable[..., dict[str, Any]], bool]] = [
    ("realistic_recall_v2", _curated_inputs, False),
    ("dailydialog_t2a", _t2a_inputs, False),
    ("esmemeval", _esmem_inputs, True),
    ("madialbench", _madial_inputs, True),
]

# Confirmatory family: (corpus, contrast key) for Hg1 and Hg2.
HG1 = ("esmemeval", "gated_vs_aft")  # recover
HG2 = ("realistic_recall_v2", "gated_vs_cosine")  # preserve


def run_benchmark(
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    corpora = [(k, fn) for (k, fn, dry_ok) in CORPORA if (dry_ok or not dry_run)]
    reports: dict[str, Any] = {}
    for key, fn in corpora:
        if verbose:
            print(f"[{key}] running arms + gate …")
        gi = fn(dry_run=dry_run)
        rep = gate_report(
            gi["cosine"],
            gi["aft"],
            gi["valence"],
            n_bootstrap=n_bootstrap,
            seed=seed,
            primary_tau=DEFAULT_TAU,
            tau_grid=TAU_GRID,
        )
        rep["metric"] = gi["metric"]
        reports[key] = rep

    # Confirmatory family Hg1 (recover) + Hg2 (preserve), Holm m=2.
    verdict: dict[str, Any] = {"evaluable": False}
    if not dry_run and HG1[0] in reports and HG2[0] in reports:
        hg1 = reports[HG1[0]]["primary"][HG1[1]]
        hg2 = reports[HG2[0]]["primary"][HG2[1]]
        p_holm = holm_bonferroni([hg1["p_bootstrap_onetail"], hg2["p_bootstrap_onetail"]])
        hg1_pass = p_holm[0] < 0.05 and hg1["delta"] > 0
        hg2_pass = p_holm[1] < 0.05 and hg2["delta"] > 0
        verdict = {
            "evaluable": True,
            "family": "Holm m=2, one-tailed",
            "hg1_recover": {
                "corpus": HG1[0],
                "contrast": HG1[1],
                "delta": hg1["delta"],
                "ci": [hg1["ci_lower"], hg1["ci_upper"]],
                "p_holm": p_holm[0],
                "pass": hg1_pass,
            },
            "hg2_preserve": {
                "corpus": HG2[0],
                "contrast": HG2[1],
                "delta": hg2["delta"],
                "ci": [hg2["ci_lower"], hg2["ci_upper"]],
                "p_holm": p_holm[1],
                "pass": hg2_pass,
            },
            "branch_a": hg1_pass and hg2_pass,
        }

    return {
        "benchmark": "addendum_y_query_affect_gate",
        "pre_registration": "benchmarks/preregistration_addendum_y_query_affect_gate.md",
        "dry_run": dry_run,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "primary_tau": DEFAULT_TAU,
        "tau_grid": list(TAU_GRID),
        "corpora": reports,
        "verdict": verdict,
    }


def _fmt_contrast(c: dict[str, Any]) -> str:
    return (
        f"Δ={c['delta']:+.3f} [{c['ci_lower']:+.3f}, {c['ci_upper']:+.3f}] "
        f"p1={c['p_bootstrap_onetail']:.4f} d={c['cohens_d']:.3f}"
    )


def write_results(
    results: dict[str, Any],
    *,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
    out_protocol: Path | None = DEFAULT_OUT_PROTOCOL,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Addendum Y — Query-affect-conditioned gate (Hg1 recover / Hg2 preserve)",
        "",
        f"**τ (primary):** {results['primary_tau']}  **Bootstrap:** n={results['n_bootstrap']}, "
        f"seed={results['seed']}"
        + ("  **[DRY RUN — not a scored result]**" if results["dry_run"] else ""),
        "",
        "## Per-corpus (primary τ)",
        "",
        "| Corpus | metric | n | route→cos | gated | cosine | aft | recov "
        "| gated-cos | gated-aft |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, rep in results["corpora"].items():
        p = rep["primary"]
        lines.append(
            f"| {key} | {rep['metric']} | {rep['n_queries']} | {p['routing_rate']:.1%} | "
            f"{p['gated_mean']:.3f} | {p['cosine_mean']:.3f} | {p['aft_mean']:.3f} | "
            f"{p['recovery_fraction']:.2f} | {_fmt_contrast(p['gated_vs_cosine'])} | "
            f"{_fmt_contrast(p['gated_vs_aft'])} |"
        )
    v = results["verdict"]
    lines += ["", "## Verdict (Holm m=2)", ""]
    if v.get("evaluable"):
        for h in ("hg1_recover", "hg2_preserve"):
            hh = v[h]
            lines.append(
                f"- **{h}** ({hh['corpus']} {hh['contrast']}): Δ={hh['delta']:+.3f} "
                f"[{hh['ci'][0]:+.3f}, {hh['ci'][1]:+.3f}] p_holm={hh['p_holm']:.4f} → "
                f"{'PASS' if hh['pass'] else 'FAIL'}"
            )
        lines.append("")
        lines.append(f"**Branch A (both PASS): {v['branch_a']}**")
    else:
        lines.append("_Not evaluable (dry run or missing corpus)._")
    lines += ["", "Decision rule: `benchmarks/preregistration_addendum_y_query_affect_gate.md`."]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    if out_protocol is not None:
        out_protocol.write_text(
            json.dumps(
                {
                    "benchmark": results["benchmark"],
                    "pre_registration": results["pre_registration"],
                    "primary_tau": results["primary_tau"],
                    "confirmatory_family": "Holm m=2: Hg1 esmemeval gated_vs_aft, "
                    "Hg2 realistic_recall_v2 gated_vs_cosine",
                    "dry_run": results["dry_run"],
                    "branch_a": results["verdict"].get("branch_a"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Addendum Y query-affect gate")
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="Smoke: esmemeval + madial, no LLM")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_benchmark(
        n_bootstrap=args.n_bootstrap, seed=args.seed, dry_run=args.dry_run, verbose=not args.quiet
    )
    out_json, out_md, out_protocol = DEFAULT_OUT_JSON, DEFAULT_OUT_MD, DEFAULT_OUT_PROTOCOL
    if args.dry_run:
        out_json = out_json.with_name("results.dry.json")
        out_md = out_md.with_name("results.dry.md")
        out_protocol = out_protocol.with_name("results.protocol.dry.json")
    write_results(results, out_json=out_json, out_md=out_md, out_protocol=out_protocol)
    print(f"\nResults written to {out_json}")
    v = results["verdict"]
    if v.get("evaluable"):
        print(
            f"Hg1 recover: {'PASS' if v['hg1_recover']['pass'] else 'FAIL'} · "
            f"Hg2 preserve: {'PASS' if v['hg2_preserve']['pass'] else 'FAIL'} · "
            f"Branch A: {v['branch_a']}"
        )
    else:
        print("Verdict not evaluable (dry run or missing corpus).")


if __name__ == "__main__":
    main()
