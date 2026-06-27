"""Addendum W — affine calibration of direct-VAD arousal (Hw1/Hw2/Gw).

Addendum V found that direct-VAD arousal beats the SEC->projection on *correlation*
(r 0.58 vs 0.23) but trades it for a worse *MAE* (0.193 vs 0.112), driven by a
systematic under-prediction (bias -0.093) and a wider spread than EmoBank's narrow
arousal range. Pearson r is invariant under an affine map, so a fit
``arousal_cal = a * arousal_direct + b`` can correct the scale/offset, keeping the
better r while cutting MAE.

This runner is **deterministic and offline** — it consumes the paired per-item
prediction dump produced once by the Addendum V harness
(``benchmarks.appraisal_vad.runner --dump-predictions``) and performs the affine fit
under two pre-declared protocols (native EmoBank split; 5-fold CV). No LLM calls.

Pre-registration: ``benchmarks/preregistration_addendum_w_arousal_calibration.md``.

Usage::

    # 1. one-time LLM pass (needs API key) to produce the dump:
    uv run python -m benchmarks.appraisal_vad.runner \
        --dump-predictions benchmarks/arousal_calibration/predictions.json
    # 2. deterministic calibration + verdicts:
    make bench-arousal-calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.common.statistics import DEFAULT_N_BOOTSTRAP, paired_bootstrap_diff
from benchmarks.human_gold_appraisal.runner import _pearson

ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).parent
DEFAULT_PREDICTIONS = _HERE / "predictions.json"
DEFAULT_OUT_JSON = _HERE / "results.json"
DEFAULT_OUT_MD = _HERE / "results.md"

DIMENSION = "arousal"
EVAL_SPLITS = ("dev", "test")
GUARD_R_TOL = 0.02


def affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ordinary least squares slope/intercept for ``y ~ a*x + b``."""
    slope, intercept = np.polyfit(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), 1
    )
    return float(slope), float(intercept)


def _metrics(pred: np.ndarray, human: np.ndarray) -> dict[str, Any]:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(human, dtype=np.float64)
    resid = p - y
    return {
        "n": len(y),
        "mae": round(float(np.mean(np.abs(resid))), 4),
        "bias": round(float(np.mean(resid)), 4),
        "pearson_r": round(_pearson(p, y), 4),
    }


def kfold_oof_calibrate(x: np.ndarray, y: np.ndarray, *, k: int, seed: int) -> np.ndarray:
    """Out-of-fold calibrated predictions: each item scored by a model that never saw it."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    n = len(xa)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    oof = np.empty(n, dtype=np.float64)
    for fold in np.array_split(perm, k):
        train_idx = np.setdiff1d(perm, fold, assume_unique=False)
        a, b = affine_fit(xa[train_idx], ya[train_idx])
        oof[fold] = a * xa[fold] + b
    return oof


def _abs_err_reduction(
    raw_err: np.ndarray, cal_err: np.ndarray, *, n_bootstrap: int, seed: int
) -> dict[str, Any]:
    """Paired bootstrap on per-item ``|raw_err| - |cal_err|`` (positive = calibration helps)."""
    diff, lo, hi, p = paired_bootstrap_diff(
        np.abs(raw_err).tolist(),
        np.abs(cal_err).tolist(),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return {
        "mean_abs_err_reduction": round(diff, 4),
        "ci": [round(lo, 4), round(hi, 4)],
        "p": round(p, 4),
        "improves": bool(diff > 0 and lo > 0),
    }


def _evaluate(
    x: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    cal: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Metrics + verdicts for one held-out evaluation set (raw / calibrated / scherer)."""
    raw_m = _metrics(x, y)
    cal_m = _metrics(cal, y)
    sch_m = _metrics(s, y)
    gw_r_ok = cal_m["pearson_r"] >= raw_m["pearson_r"] - GUARD_R_TOL
    return {
        "raw": raw_m,
        "calibrated": cal_m,
        "scherer_m1": sch_m,
        "Hw1_cal_beats_raw": _abs_err_reduction(
            x - y, cal - y, n_bootstrap=n_bootstrap, seed=seed
        ),
        "Hw2_cal_beats_scherer": _abs_err_reduction(
            s - y, cal - y, n_bootstrap=n_bootstrap, seed=seed
        ),
        "Gw_r_preserved": bool(gw_r_ok),
    }


def run(
    dump: dict[str, Any],
    *,
    dimension: str = DIMENSION,
    k: int = 5,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
) -> dict[str, Any]:
    items = dump["items"]
    x = np.array([it["direct_vad"][dimension] for it in items], dtype=np.float64)
    s = np.array([it["scherer_m1"][dimension] for it in items], dtype=np.float64)
    y = np.array([it["human"][dimension] for it in items], dtype=np.float64)
    splits = [it.get("split") for it in items]

    # Deployable coefficients: full-sample affine fit (what a shipped calibration would use).
    a_full, b_full = affine_fit(x, y)

    protocols: dict[str, Any] = {}

    # P1 — native EmoBank split (headline): fit on train, evaluate on dev+test.
    train_idx = np.array([i for i, sp in enumerate(splits) if sp == "train"], dtype=np.int64)
    eval_idx = np.array([i for i, sp in enumerate(splits) if sp in EVAL_SPLITS], dtype=np.int64)
    if len(train_idx) >= 2 and len(eval_idx) >= 2:
        a, b = affine_fit(x[train_idx], y[train_idx])
        cal_eval = a * x[eval_idx] + b
        p1 = _evaluate(
            x[eval_idx], s[eval_idx], y[eval_idx], cal_eval, n_bootstrap=n_bootstrap, seed=seed
        )
        p1["coefficients"] = {"a": round(a, 4), "b": round(b, 4)}
        p1["n_fit"] = len(train_idx)
        p1["n_eval"] = len(eval_idx)
        protocols["native_split"] = p1

    # P2 — 5-fold CV over all items (robustness): out-of-fold calibration.
    oof = kfold_oof_calibrate(x, y, k=k, seed=seed)
    p2 = _evaluate(x, s, y, oof, n_bootstrap=n_bootstrap, seed=seed)
    p2["k"] = k
    p2["n_eval"] = len(y)
    protocols["kfold_cv"] = p2

    headline = protocols.get("native_split", protocols["kfold_cv"])
    gw_ok = bool(a_full > 0 and headline["Gw_r_preserved"])
    adopt = bool(headline["Hw1_cal_beats_raw"]["improves"] and gw_ok)

    return {
        "benchmark": "arousal_affine_calibration_w",
        "dataset": dump.get("dataset"),
        "version": dump.get("version"),
        "dimension": dimension,
        "n": len(items),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "deployable_coefficients": {"a": round(a_full, 4), "b": round(b_full, 4)},
        "protocols": protocols,
        "verdicts": {
            "Hw1_cal_reduces_mae": bool(headline["Hw1_cal_beats_raw"]["improves"]),
            "Hw2_cal_beats_scherer": bool(headline["Hw2_cal_beats_scherer"]["improves"]),
            "Gw_slope_positive_and_r_preserved": gw_ok,
            "adopt_calibration": adopt,
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    co = report["deployable_coefficients"]
    lines = [
        "# Addendum W — affine calibration of direct-VAD arousal",
        "",
        f"Dataset: `{report['dataset']}` v{report['version']} · dimension: "
        f"**{report['dimension']}** · N={report['n']} · bootstrap n={report['n_bootstrap']} · "
        f"seed={report['seed']}.",
        "",
        f"Deployable affine coefficients (full-sample fit): "
        f"`arousal_cal = {co['a']:+.4f}·arousal_direct {co['b']:+.4f}`.",
        "",
    ]
    for name, proto in report["protocols"].items():
        header = f"## Protocol `{name}`"
        if "coefficients" in proto:
            pc = proto["coefficients"]
            header += (
                f" — fit `a={pc['a']:+.4f}, b={pc['b']:+.4f}` "
                f"(n_fit={proto['n_fit']}, n_eval={proto['n_eval']})"
            )
        elif "k" in proto:
            header += f" — {proto['k']}-fold out-of-fold (n_eval={proto['n_eval']})"
        lines += [header, ""]
        lines += [
            "| Estimator | MAE | bias | Pearson r |",
            "|---|---:|---:|---:|",
        ]
        for key, label in (
            ("raw", "direct_vad (raw)"),
            ("calibrated", "direct_vad (calibrated)"),
            ("scherer_m1", "scherer_m1"),
        ):
            m = proto[key]
            lines.append(
                f"| {label} | {m['mae']:.4f} | {m['bias']:+.4f} | {m['pearson_r']:+.4f} |"
            )
        h1 = proto["Hw1_cal_beats_raw"]
        h2 = proto["Hw2_cal_beats_scherer"]
        lines += [
            "",
            f"- **Hw1** (calibrated < raw MAE): ΔMAE={h1['mean_abs_err_reduction']:+.4f} "
            f"CI=[{h1['ci'][0]:+.4f}, {h1['ci'][1]:+.4f}] p={h1['p']:.4f} → "
            f"{'✅ improves' if h1['improves'] else '— no'}",
            f"- **Hw2** (calibrated < scherer MAE): ΔMAE={h2['mean_abs_err_reduction']:+.4f} "
            f"CI=[{h2['ci'][0]:+.4f}, {h2['ci'][1]:+.4f}] p={h2['p']:.4f} → "
            f"{'✅ dominates' if h2['improves'] else '— no'}",
            f"- **Gw** (r preserved): {'✅' if proto['Gw_r_preserved'] else '✗'}",
            "",
        ]
    v = report["verdicts"]
    lines += [
        "## Decision",
        "",
        f"- Hw1 (reduces MAE): {'✅ PASS' if v['Hw1_cal_reduces_mae'] else '✗ FAIL'}",
        f"- Hw2 (beats scherer MAE): {'✅ PASS' if v['Hw2_cal_beats_scherer'] else '✗ FAIL'}",
        f"- Gw (slope>0 & r preserved): "
        f"{'✅ OK' if v['Gw_slope_positive_and_r_preserved'] else '✗'}",
        f"- **Adopt affine calibration:** "
        f"{'✅ YES' if v['adopt_calibration'] else '✗ NO (raw direct-VAD arousal stands)'}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Addendum W — affine arousal calibration.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--dimension", type=str, default=DIMENSION)
    parser.add_argument("--k", type=int, default=5, help="CV folds for the robustness protocol.")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.predictions.exists():
        raise SystemExit(
            f"Predictions dump not found: {args.predictions}\n"
            "Produce it first with one LLM pass (needs EMOTIONAL_MEMORY_LLM_API_KEY):\n"
            "  uv run python -m benchmarks.appraisal_vad.runner "
            f"--dump-predictions {args.predictions}"
        )

    dump = json.loads(args.predictions.read_text(encoding="utf-8"))
    report = run(
        dump,
        dimension=args.dimension,
        k=args.k,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out_md.write_text(_render_markdown(report), encoding="utf-8")
    v = report["verdicts"]
    print(
        f"Addendum W complete: adopt_calibration={v['adopt_calibration']} "
        f"(Hw1={v['Hw1_cal_reduces_mae']} Hw2={v['Hw2_cal_beats_scherer']} "
        f"Gw={v['Gw_slope_positive_and_r_preserved']})"
    )


if __name__ == "__main__":
    main()
