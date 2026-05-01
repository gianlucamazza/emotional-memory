"""Appraisal confound study.

Isolates whether the AFT retrieval advantage comes from the 6-signal architecture
or from the appraisal engine's affect inference.

Three conditions, all using the same embedder and dataset (realistic_recall_v1):

  aft_noAppraisal  — AFTReplayAdapter with no appraisal engine; affect is set
                     manually from the scenario's preset (valence, arousal) values.
                     This is the existing realistic-benchmark AFT behaviour.

  aft_keyword      — AFTReplayAdapter + KeywordAppraisalEngine; affect is inferred
                     from content without LLM involvement. Deterministic, no API key.

  naive_cosine     — semantic cosine baseline. Same embedder, no affective signals.

Primary hypothesis (Ha2):
  aft_keyword.top1_accuracy > naive_cosine.top1_accuracy  (Δ > 0.05, one-tailed alpha=0.05)
  → if Ha2 holds: AFT architecture adds value beyond raw cosine, even without LLM.

Secondary hypothesis (Hb2):
  |aft_keyword.top1_accuracy - aft_noAppraisal.top1_accuracy| < 0.05
  → if Hb2 holds: preset and inferred affect are functionally equivalent on this dataset.

Note on aft_keyword: KeywordAppraisalEngine is injected after EmotionalMemory
creation and overwrites the scenario's preset valence/arousal at encode time.
Hb2 FAIL (keyword << noAppraisal) means keyword inference is destructive on
this dataset, not neutral. The aft_noAppraisal > naive_cosine comparison
(descriptive, not a pre-registered hypothesis) is the cleaner architecture-
attribution signal and aligns with Study S2 (no appraisal engine in S2 either).

Pre-registration: benchmarks/preregistration_addendum_v2.md Addendum A
  seed=42, n_bootstrap=10000, one-tailed alpha=0.05 for Ha2, no Holm correction.

Usage::

    uv run python -m benchmarks.appraisal_confound.runner
    uv run python -m benchmarks.appraisal_confound.runner --embedder sbert-bge
    uv run python -m benchmarks.appraisal_confound.runner --embedder hash --n-bootstrap 500
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.common.statistics import (
    bootstrap_ci,
    ci_payload,
    cohens_d_paired,
    format_point_ci,
    paired_bootstrap_diff,
)
from benchmarks.realistic.adapters import (
    AFTReplayAdapter,
    NaiveCosineReplayAdapter,
    ReplayAdapter,
)
from benchmarks.realistic.runner import (
    DATASET,
    load_dataset,
    run_system_on_scenario,
    validate_dataset_difficulty,
)

_HERE = Path(__file__).resolve().parent
DEFAULT_OUT_JSON = _HERE / "results.json"
DEFAULT_OUT_MD = _HERE / "results.md"
DEFAULT_OUT_PROTOCOL = _HERE / "results.protocol.json"

SYSTEMS = ["aft_noAppraisal", "aft_keyword", "naive_cosine"]
_PREREG_N_BOOTSTRAP = 10_000
_PREREG_SEED = 42
_HA2_ONE_SIDED_ALPHA = 0.05
_HA2_EFFECT_THRESHOLD = 0.05


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


class AFTKeywordReplayAdapter(AFTReplayAdapter):
    """AFTReplayAdapter with KeywordAppraisalEngine injected at session start."""

    name = "aft_keyword"

    def begin_session(self, session_id: str) -> Any:
        from emotional_memory.appraisal_llm import KeywordAppraisalEngine

        result = super().begin_session(session_id)
        if self._engine is not None:
            self._engine._appraisal_engine = KeywordAppraisalEngine()
        return result


def _make_adapter(
    system_name: str,
    *,
    workdir: Path,
    embedder: Any = None,
) -> ReplayAdapter:
    if system_name == "aft_noAppraisal":
        return AFTReplayAdapter(workdir / system_name, embedder=embedder)
    if system_name == "aft_keyword":
        return AFTKeywordReplayAdapter(workdir / system_name, embedder=embedder)
    if system_name == "naive_cosine":
        return NaiveCosineReplayAdapter(embedder=embedder)
    raise ValueError(f"Unknown system: {system_name!r}")


def _build_embedder(name: str | None) -> Any:
    if name is None or name == "hash":
        return None
    try:
        from emotional_memory.embedders import SentenceTransformerEmbedder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers required: pip install 'emotional-memory[sentence-transformers]'"
        ) from exc
    if name == "sbert-bge":
        return SentenceTransformerEmbedder.make_bge_small()
    raise ValueError(f"Unknown embedder: {name!r}. Choices: hash, sbert-bge")


def _collect_top1_flags(scenario_reports: list[dict[str, Any]]) -> list[float]:
    return [
        1.0 if qr.get("top1_hit") else 0.0
        for scenario_report in scenario_reports
        for session_report in scenario_report.get("sessions", [])
        for qr in session_report.get("queries", [])
    ]


def run_study(
    dataset_path: Path = DATASET,
    *,
    embedder_name: str | None = None,
    n_bootstrap: int = _PREREG_N_BOOTSTRAP,
    seed: int = _PREREG_SEED,
) -> dict[str, Any]:
    _seed_everything(seed)

    dataset = load_dataset(dataset_path)
    validate_dataset_difficulty(dataset, top_k=dataset.default_top_k)

    embedder = _build_embedder(embedder_name)

    system_scenario_reports: dict[str, list[dict[str, Any]]] = {s: [] for s in SYSTEMS}

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        for system_name in SYSTEMS:
            adapter = _make_adapter(system_name, workdir=workdir, embedder=embedder)
            for scenario in dataset.scenarios:
                adapter.reset()
                # Pass n_bootstrap=1: per-scenario CIs are discarded; all
                # bootstrap happens once below on the full flag lists.
                report = run_system_on_scenario(
                    adapter,
                    scenario,
                    default_top_k=dataset.default_top_k,
                    n_bootstrap=1,
                    seed=seed,
                )
                system_scenario_reports[system_name].append(report)
            adapter.close()

    # per-system aggregate
    system_results: dict[str, dict[str, Any]] = {}
    for system_name in SYSTEMS:
        flags = _collect_top1_flags(system_scenario_reports[system_name])
        mean, lo, hi = bootstrap_ci(flags, n_bootstrap=n_bootstrap, seed=seed)
        system_results[system_name] = {
            "n_queries": len(flags),
            "top1_accuracy": round(mean, 4),
            "ci_95": ci_payload(mean, lo, hi, n_bootstrap=n_bootstrap),
        }

    # Ha2: aft_keyword > naive_cosine
    kw_flags = _collect_top1_flags(system_scenario_reports["aft_keyword"])
    cos_flags = _collect_top1_flags(system_scenario_reports["naive_cosine"])
    no_flags = _collect_top1_flags(system_scenario_reports["aft_noAppraisal"])

    ha2_delta, ha2_lo, ha2_hi, ha2_p = paired_bootstrap_diff(
        kw_flags, cos_flags, n_bootstrap=n_bootstrap, seed=seed
    )
    hb2_delta, hb2_lo, hb2_hi, hb2_p = paired_bootstrap_diff(
        kw_flags, no_flags, n_bootstrap=n_bootstrap, seed=seed
    )
    # Hd1 (Addendum D): aft_noAppraisal > naive_cosine, Δ > 0.10
    hd1_delta, hd1_lo, hd1_hi, hd1_p = paired_bootstrap_diff(
        no_flags, cos_flags, n_bootstrap=n_bootstrap, seed=seed
    )
    d_ha2 = cohens_d_paired(kw_flags, cos_flags)
    d_hb2 = cohens_d_paired(kw_flags, no_flags)
    d_hd1 = cohens_d_paired(no_flags, cos_flags)

    # Pre-reg Addendum A: one-tailed alpha=0.05 + Δ > 0.05 practical threshold.
    # One-sided p = p_two_sided / 2 (bootstrap is symmetric, direction is fixed).
    ha2_pass = (ha2_p / 2.0) < _HA2_ONE_SIDED_ALPHA and ha2_delta > _HA2_EFFECT_THRESHOLD
    hb2_pass = abs(hb2_delta) < 0.05 and hb2_lo > -0.05 and hb2_hi < 0.05
    hd1_pass = hd1_delta > 0.10 and hd1_lo > 0.0

    hypotheses = {
        "Hd1": {
            "description": "aft_noAppraisal.top1 > naive_cosine.top1, Δ > 0.10 (Addendum D)",
            "result": "PASS" if hd1_pass else "FAIL",
            "delta": round(hd1_delta, 4),
            "ci_95": ci_payload(hd1_delta, hd1_lo, hd1_hi, n_bootstrap=n_bootstrap),
            "p_two_sided": round(hd1_p, 4),
            "cohens_d": round(d_hd1, 4),
            "interpretation": (
                "AFT architecture (no appraisal, preset affect) reliably outperforms naive cosine."
                if hd1_pass
                else "AFT without appraisal does not show a practically significant advantage "
                "over naive cosine at Δ > 0.10 threshold."
            ),
        },
        "Ha2": {
            "description": (
                f"aft_keyword.top1 > naive_cosine.top1 "
                f"(Δ > {_HA2_EFFECT_THRESHOLD}, one-tailed alpha={_HA2_ONE_SIDED_ALPHA})"
            ),
            "result": "PASS" if ha2_pass else "FAIL",
            "delta": round(ha2_delta, 4),
            "ci_95": ci_payload(ha2_delta, ha2_lo, ha2_hi, n_bootstrap=n_bootstrap),
            "p_one_sided": round(ha2_p / 2.0, 4),
            "p_two_sided": round(ha2_p, 4),
            "cohens_d": round(d_ha2, 4),
            "interpretation": (
                "AFT architecture adds retrieval value beyond raw cosine (keyword appraisal)."
                if ha2_pass
                else "No significant advantage of AFT+keyword over naive cosine. "
                "Architecture benefit not confirmed on this dataset."
            ),
        },
        "Hb2": {
            "description": "|aft_keyword - aft_noAppraisal| < 0.05 (equivalence)",
            "result": "PASS" if hb2_pass else "FAIL",
            "delta": round(hb2_delta, 4),
            "ci_95": ci_payload(hb2_delta, hb2_lo, hb2_hi, n_bootstrap=n_bootstrap),
            "p_two_sided": round(hb2_p, 4),
            "cohens_d": round(d_hb2, 4),
            "interpretation": (
                "Keyword inference ≈ preset affect. Architecture is the dominant driver."
                if hb2_pass
                else "Keyword appraisal meaningfully changes retrieval vs preset affect. "
                "Appraisal inference is not neutral."
            ),
        },
    }

    return {
        "study": "appraisal_confound",
        "dataset": dataset.name,
        "dataset_version": dataset.version,
        "embedder": embedder_name or "hash",
        "n_scenarios": len(dataset.scenarios),
        "n_queries": len(kw_flags),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "systems": system_results,
        "hypotheses": hypotheses,
    }


def _build_protocol(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "study": results["study"],
        "dataset": results["dataset"],
        "dataset_version": results["dataset_version"],
        "embedder": results["embedder"],
        "systems": list(results["systems"].keys()),
        "n_bootstrap": results["n_bootstrap"],
        "seed": results["seed"],
        "hypotheses": {
            k: {
                "description": v["description"],
                "result": v["result"],
            }
            for k, v in results["hypotheses"].items()
        },
        "interpretation_notes": [
            "Ha2: one-tailed alpha=0.05 + Δ>0.05 (pre-reg Addendum A). "
            "p_one_sided = p_two_sided / 2 (bootstrap symmetric, direction fixed).",
            "Hb2: equivalence; |aft_keyword - aft_noAppraisal| CI fully inside ±0.05.",
            "No Holm correction: single primary hypothesis in this study (pre-reg Addendum A).",
            "aft_noAppraisal vs naive_cosine (descriptive, not pre-registered): "
            "architecture-attribution signal; aligns with Study S2 (no appraisal in S2 either).",
            "Hb2 FAIL means keyword appraisal overrides preset affect destructively on "
            "this dataset — keyword-inferred affect is noisier than hand-curated presets.",
        ],
    }


def _render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Appraisal Confound Study",
        "",
        f"Dataset: {results['dataset']} v{results['dataset_version']}  "
        f"({results['n_scenarios']} scenarios, {results['n_queries']} queries)  ",
        f"Embedder: `{results['embedder']}`  "
        f"n_bootstrap: {results['n_bootstrap']}  seed: {results['seed']}",
        "",
        "## System Results",
        "",
        "| System | N | top1_acc | 95% CI |",
        "|---|---:|---:|---|",
    ]
    for sys_name, sys in results["systems"].items():
        ci = sys["ci_95"]
        lines.append(
            f"| `{sys_name}` | {sys['n_queries']} "
            f"| {sys['top1_accuracy']:.3f} "
            f"| [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}] |"
        )
    lines += ["", "## Hypothesis Tests", ""]
    for hyp_name, hyp in results["hypotheses"].items():
        verdict = "✓ PASS" if hyp["result"] == "PASS" else "✗ FAIL"
        ci = hyp["ci_95"]
        p_str = (
            f"p_one_sided = {hyp['p_one_sided']:.4f}"
            if "p_one_sided" in hyp
            else f"p_two_sided = {hyp['p_two_sided']:.4f}"
        )
        lines += [
            f"### {hyp_name} — {verdict}",
            "",
            f"**{hyp['description']}**",
            "",
            f"Δ = {format_point_ci(hyp['delta'], ci['ci_lower'], ci['ci_upper'])}  "
            f"{p_str}  Cohen's d = {hyp['cohens_d']:.3f}",
            "",
            f"*{hyp['interpretation']}*",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Appraisal confound study.")
    parser.add_argument("--embedder", default="sbert-bge", choices=["hash", "sbert-bge"])
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=_PREREG_N_BOOTSTRAP,
        help=f"Bootstrap samples (pre-reg default: {_PREREG_N_BOOTSTRAP})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_PREREG_SEED,
        help=f"RNG seed (pre-reg default: {_PREREG_SEED})",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-protocol", type=Path, default=DEFAULT_OUT_PROTOCOL)
    args = parser.parse_args()

    print(
        f"Running appraisal confound study "
        f"(embedder={args.embedder}, n_bootstrap={args.n_bootstrap}, seed={args.seed}) …"
    )
    results = run_study(
        embedder_name=args.embedder,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.out_md.write_text(_render_markdown(results), encoding="utf-8")
    args.out_protocol.write_text(json.dumps(_build_protocol(results), indent=2), encoding="utf-8")

    print(f"\nResults written to {args.out_json}, {args.out_md}, {args.out_protocol}")
    print("\n=== Hypothesis Summary ===")
    for hyp_name, hyp in results["hypotheses"].items():
        p_label = hyp.get("p_one_sided", hyp["p_two_sided"])
        print(
            f"  {hyp_name}: {hyp['result']}  Δ={hyp['delta']:.3f}  "
            f"p={p_label:.4f}  d={hyp['cohens_d']:.3f}"
        )


if __name__ == "__main__":
    main()
