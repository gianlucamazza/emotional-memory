"""Diagnostica residui SEC-by-SEC del LLM appraisal (WP-1a).

Confronta AppraisalVector prodotto da LLMAppraisalEngine con i valori
oracle valence/arousal del dataset realistic_recall_v3.json. Riporta:
  - Bias (mean residual), Std, MAE, Pearson r per valence e arousal.
  - Statistiche descrittive delle 5 dimensioni SEC (no oracle).
  - Confusion matrix del segno valence (LLM vs oracle).
  - Bootstrap 95% CI su bias.

Decision tree nel report:
  - Bias sistematico → fix prompt (P1b).
  - Varianza alta, bias ~ 0 → confidence gating (P1c).
  - Entrambi → reframing onesto: documentare che zero-shot LLM appraisal
    non è sufficiente; supporto a appraisal fine-tuned (P1d).

Pre-registration: benchmarks/appraisal_diagnostics/protocol.md
Dataset:          benchmarks/datasets/realistic_recall_v3.json
Output:           benchmarks/appraisal_diagnostics/results.diagnostic.{json,md}
Seed:             42 (frozen per protocollo)

Usage::

    # dry-run (no LLM call, smoke test):
    uv run python -m benchmarks.appraisal_diagnostics.runner --dry-run

    # full run (requires EMOTIONAL_MEMORY_LLM_API_KEY):
    uv run python -m benchmarks.appraisal_diagnostics.runner --n 200 --seed 42 \\
        --out-json benchmarks/appraisal_diagnostics/results.diagnostic.json \\
        --out-md   benchmarks/appraisal_diagnostics/results.diagnostic.md
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from benchmarks.common.statistics import bootstrap_ci

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

DEFAULT_DATASET = _ROOT / "benchmarks" / "datasets" / "realistic_recall_v3.json"
DEFAULT_OUT_JSON = _HERE / "results.diagnostic.json"
DEFAULT_OUT_MD = _HERE / "results.diagnostic.md"

_PREREG_SEED = 42
_N_BOOTSTRAP = 10_000
_CI = 0.95

# Decision-tree thresholds (pre-registered)
_BIAS_THRESHOLD = 0.10  # |mean residual| > this → systematic bias
_STD_THRESHOLD = 0.30  # std residual > this → high variance


# ---------------------------------------------------------------------------
# Dataset models
# ---------------------------------------------------------------------------


class _Event(BaseModel):
    memory_id: str
    content: str
    valence: float
    arousal: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class _Session(BaseModel):
    session_id: str
    description: str
    events: list[_Event] = Field(default_factory=list)


class _Scenario(BaseModel):
    scenario_id: str
    description: str
    sessions: list[_Session]


class _Dataset(BaseModel):
    name: str
    version: str
    description: str
    scenarios: list[_Scenario]


def _load_dataset(path: Path) -> _Dataset:
    return _Dataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _all_events(dataset: _Dataset) -> list[_Event]:
    events: list[_Event] = []
    for sc in dataset.scenarios:
        for se in sc.sessions:
            events.extend(se.events)
    return events


# ---------------------------------------------------------------------------
# LLM / dry-run helpers
# ---------------------------------------------------------------------------


def _require_llm() -> Any:
    from emotional_memory.llm_http import make_httpx_llm_from_env

    llm = make_httpx_llm_from_env()
    if llm is None:
        raise RuntimeError(
            "EMOTIONAL_MEMORY_LLM_API_KEY is not set.\n"
            "Set the variable in .env or environment, or use --dry-run."
        )
    return llm


class _FixedAppraisalEngine:
    """Returns a fixed AppraisalVector for every input (dry-run / tests)."""

    def __init__(
        self,
        novelty: float = 0.5,
        goal_relevance: float = 0.2,
        coping_potential: float = 0.6,
        norm_congruence: float = 0.1,
        self_relevance: float = 0.4,
    ) -> None:
        from emotional_memory.appraisal import AppraisalVector

        self._fixed = AppraisalVector(
            novelty=novelty,
            goal_relevance=goal_relevance,
            coping_potential=coping_potential,
            norm_congruence=norm_congruence,
            self_relevance=self_relevance,
        )

    def appraise(self, _text: str, context: dict[str, Any] | None = None) -> Any:
        return self._fixed


def _build_engine(dry_run: bool) -> Any:
    if dry_run:
        return _FixedAppraisalEngine()
    from emotional_memory.appraisal_llm import LLMAppraisalEngine

    return LLMAppraisalEngine(llm=_require_llm())


# ---------------------------------------------------------------------------
# Per-dimension analysis
# ---------------------------------------------------------------------------

_SEC_DIMS = ["novelty", "goal_relevance", "coping_potential", "norm_congruence", "self_relevance"]
_DERIVED = ["valence", "arousal"]


def _sec_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _residual_stats(
    residuals: list[float],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    arr = np.asarray(residuals, dtype=np.float64)
    bias, lo, hi = bootstrap_ci(residuals, n_bootstrap=n_bootstrap, seed=seed)
    mae = float(np.mean(np.abs(arr)))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    # Pearson r with "zero" oracle residual → not meaningful here; omit r for residuals.
    return {
        "n": len(residuals),
        "bias_mean": bias,
        "bias_ci_lo": lo,
        "bias_ci_hi": hi,
        "std": std,
        "mae": mae,
    }


def _pearson_r(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    ax, ay = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(ax, ay)
    return float(c[0, 1])


def _confusion_sign(llm: list[float], oracle: list[float]) -> dict[str, int]:
    """Valence sign confusion matrix: TP/FP/TN/FN (oracle positive = oracle >= 0)."""
    counts: dict[str, int] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for lv, ov in zip(llm, oracle, strict=True):
        pred_pos = lv >= 0.0
        true_pos = ov >= 0.0
        if pred_pos and true_pos:
            counts["TP"] += 1
        elif pred_pos and not true_pos:
            counts["FP"] += 1
        elif not pred_pos and not true_pos:
            counts["TN"] += 1
        else:
            counts["FN"] += 1
    total = sum(counts.values())
    counts["accuracy"] = (counts["TP"] + counts["TN"]) / total if total else 0  # type: ignore[assignment]
    return counts


def _decision_tree(valence_stats: dict[str, Any], arousal_stats: dict[str, Any]) -> str:
    bias_v = abs(valence_stats["bias_mean"])
    std_v = valence_stats["std"]
    bias_a = abs(arousal_stats["bias_mean"])
    std_a = arousal_stats["std"]

    has_bias = bias_v > _BIAS_THRESHOLD or bias_a > _BIAS_THRESHOLD
    has_variance = std_v > _STD_THRESHOLD or std_a > _STD_THRESHOLD

    if has_bias and has_variance:
        return (
            "P1d — BOTH systematic bias and high variance detected. "
            "Zero-shot LLM appraisal is unreliable. Recommended action: "
            "document limitation; consider pluggable fine-tuned appraisal (P1d)."
        )
    if has_bias:
        return (
            "P1b — Systematic bias detected (|mean residual| > threshold). "
            "Recommended action: fix LLM appraisal prompt to reduce directional error."
        )
    if has_variance:
        return (
            "P1c — High variance, low bias. "
            "Recommended action: add confidence gating — suppress appraisal signal "
            "when LLM confidence is low."
        )
    return (
        "P1_OK — Both bias and variance within thresholds. "
        "LLM appraisal may be usable; investigate other confounders for Hg1 drop."
    )


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------


def run(
    *,
    dataset_path: Path = DEFAULT_DATASET,
    n: int | None = None,
    seed: int = _PREREG_SEED,
    n_bootstrap: int = _N_BOOTSTRAP,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    dataset = _load_dataset(dataset_path)
    events = _all_events(dataset)
    if n is not None and n < len(events):
        rng = random.Random(seed)
        events = rng.sample(events, n)

    engine = _build_engine(dry_run)

    # Collect per-event results
    sec_collected: dict[str, list[float]] = {dim: [] for dim in _SEC_DIMS}
    llm_valence: list[float] = []
    llm_arousal: list[float] = []
    oracle_valence: list[float] = []
    oracle_arousal: list[float] = []
    latencies_ms: list[float] = []

    for event in events:
        if verbose:
            print(f"  Appraising [{event.memory_id}] …")
        t0 = time.perf_counter()
        av = engine.appraise(event.content)
        elapsed = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed)

        # SEC dimensions
        for dim in _SEC_DIMS:
            sec_collected[dim].append(float(getattr(av, dim)))

        # Derived affect
        ca = av.to_core_affect()
        llm_valence.append(ca.valence)
        llm_arousal.append(ca.arousal)
        oracle_valence.append(event.valence)
        oracle_arousal.append(event.arousal)

    # Residuals
    res_valence = [lv - ov for lv, ov in zip(llm_valence, oracle_valence, strict=True)]
    res_arousal = [la - oa for la, oa in zip(llm_arousal, oracle_arousal, strict=True)]

    # Statistics
    valence_stats = _residual_stats(res_valence, n_bootstrap=n_bootstrap, seed=seed)
    valence_stats["pearson_r_with_oracle"] = _pearson_r(llm_valence, oracle_valence)
    arousal_stats = _residual_stats(res_arousal, n_bootstrap=n_bootstrap, seed=seed)
    arousal_stats["pearson_r_with_oracle"] = _pearson_r(llm_arousal, oracle_arousal)

    sec_stats = {dim: _sec_stats(sec_collected[dim]) for dim in _SEC_DIMS}
    confusion = _confusion_sign(llm_valence, oracle_valence)
    decision = _decision_tree(valence_stats, arousal_stats)

    lat_arr = np.asarray(latencies_ms)
    latency_stats = {
        "mean_ms": float(np.mean(lat_arr)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "total_s": float(np.sum(lat_arr) / 1000.0),
    }

    return {
        "protocol_version": "1.0",
        "dry_run": dry_run,
        "n_events": len(events),
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "bias_threshold": _BIAS_THRESHOLD,
        "std_threshold": _STD_THRESHOLD,
        "valence_residuals": valence_stats,
        "arousal_residuals": arousal_stats,
        "sec_descriptive": sec_stats,
        "valence_sign_confusion": confusion,
        "latency": latency_stats,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _fmt(v: float, dp: int = 3) -> str:
    return f"{v:.{dp}f}" if not math.isnan(v) else "NaN"


def render_md(result: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Appraisal Diagnostics — WP-1a Report",
        "",
        f"N events: {result['n_events']} | seed: {result['seed']} | dry_run: {result['dry_run']}",
        "",
    ]

    # Residual table
    lines += [
        "## Residuals (LLM - oracle)",
        "",
        "| Dimension | Bias (mean) | 95% CI | Std | MAE | Pearson r |",
        "|---|---|---|---|---|---|",
    ]
    for label, key in [("Valence", "valence_residuals"), ("Arousal", "arousal_residuals")]:
        s = result[key]
        ci = f"[{_fmt(s['bias_ci_lo'])}, {_fmt(s['bias_ci_hi'])}]"
        lines.append(
            f"| {label} | {_fmt(s['bias_mean'])} | {ci} | {_fmt(s['std'])} "
            f"| {_fmt(s['mae'])} | {_fmt(s['pearson_r_with_oracle'])} |"
        )
    lines.append("")

    # SEC descriptive
    lines += [
        "## SEC Dimension Descriptives (LLM output, no oracle)",
        "",
        "| Dimension | Mean | Std | Min | Max |",
        "|---|---|---|---|---|",
    ]
    for dim, s in result["sec_descriptive"].items():
        lines.append(
            f"| {dim} | {_fmt(s['mean'])} | {_fmt(s['std'])} "
            f"| {_fmt(s['min'])} | {_fmt(s['max'])} |"
        )
    lines.append("")

    # Confusion
    c = result["valence_sign_confusion"]
    lines += [
        "## Valence Sign Confusion (LLM vs oracle)",
        "",
        f"TP={c['TP']} FP={c['FP']} TN={c['TN']} FN={c['FN']} "
        f"accuracy={_fmt(float(c['accuracy']), 2)}",
        "",
    ]

    # Latency
    lat = result["latency"]
    lines += [
        "## Latency",
        "",
        f"Mean: {_fmt(lat['mean_ms'], 1)} ms | P95: {_fmt(lat['p95_ms'], 1)} ms "
        f"| Total: {_fmt(lat['total_s'], 1)} s",
        "",
    ]

    # Decision
    lines += [
        "## Decision",
        "",
        result["decision"],
        "",
        "---",
        "",
        f"Thresholds: bias > {result['bias_threshold']} → P1b/P1d; "
        f"std > {result['std_threshold']} → P1c/P1d.",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--n", type=int, default=None, help="Number of events (default: all)")
    p.add_argument("--seed", type=int, default=_PREREG_SEED)
    p.add_argument("--n-bootstrap", type=int, default=_N_BOOTSTRAP)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    p.add_argument("--dry-run", action="store_true", help="Use fixed appraisal; no LLM call")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.verbose or args.dry_run:
        print(f"Dataset: {args.dataset}")
        print(f"N: {args.n or 'all'} | seed: {args.seed} | dry_run: {args.dry_run}")

    result = run(
        dataset_path=args.dataset,
        n=args.n,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.out_md.write_text(render_md(result), encoding="utf-8")

    if args.verbose or args.dry_run:
        print(f"Written: {args.out_json}, {args.out_md}")
        print(f"Decision: {result['decision']}")


if __name__ == "__main__":
    main()
