"""Fit + evaluate SEC→valence/arousal recalibration models (Addendum O).

Pure numpy, no LLM, no network. Consumes the SEC dump produced by ``dump_sec.py`` and the
frozen by-scenario split, fits M0/M1/M2 on TRAIN, and evaluates them against the current mapping
on the held-out TEST scenarios. Reports bias / MAE / Pearson r / valence sign-accuracy with
bootstrap CIs, plus 5-fold CV on train for coefficient stability.

Pre-registration: benchmarks/preregistration_addendum_o_mapping_recalibration.md
Inputs:  sec_dump.gpt5mini.jsonl, split.scenarios.seed42.json
Output:  results.recalibration.gpt5mini.{json,md}

Models (valence intercept constrained to 0; arousal intercept free — see protocol G1):
  M0  current weights + offset only
  M1  recalibrated weights on theoretical features (coping_signed, |novelty|, 1-coping)
  M2  free linear on the 5 raw SECs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.common.statistics import bootstrap_ci

_HERE = Path(__file__).resolve().parent
DEFAULT_DUMP = _HERE / "sec_dump.gpt5mini.jsonl"
DEFAULT_SPLIT = _HERE / "split.scenarios.seed42.json"
DEFAULT_OUT_JSON = _HERE / "results.recalibration.gpt5mini.json"
DEFAULT_OUT_MD = _HERE / "results.recalibration.gpt5mini.md"

_SEED = 42
_N_BOOTSTRAP = 10_000

# Current mapping coefficients (appraisal_schema._scherer_project), for the baseline + M0.
_CUR_V = {"goal_relevance": 0.4, "norm_congruence": 0.3, "coping_signed": 0.2, "novelty": 0.1}
_CUR_A = {"abs_novelty": 0.5, "one_minus_coping": 0.3, "self_relevance": 0.2}


# ── feature construction ───────────────────────────────────────────────────────


def _valence_theory_features(rows: list[dict[str, Any]]) -> np.ndarray:
    """[goal_relevance, norm_congruence, coping_signed, novelty] — current valence basis."""
    return np.array(
        [
            [
                r["goal_relevance"],
                r["norm_congruence"],
                2.0 * r["coping_potential"] - 1.0,
                r["novelty"],
            ]
            for r in rows
        ],
        dtype=np.float64,
    )


def _arousal_theory_features(rows: list[dict[str, Any]]) -> np.ndarray:
    """[|novelty|, 1-coping, self_relevance] — current arousal basis."""
    return np.array(
        [[abs(r["novelty"]), 1.0 - r["coping_potential"], r["self_relevance"]] for r in rows],
        dtype=np.float64,
    )


def _raw_features(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [
            [
                r["novelty"],
                r["goal_relevance"],
                r["coping_potential"],
                r["norm_congruence"],
                r["self_relevance"],
            ]
            for r in rows
        ],
        dtype=np.float64,
    )


def _baseline_valence(rows: list[dict[str, Any]]) -> np.ndarray:
    f = _valence_theory_features(rows)
    w = np.array(
        [
            _CUR_V["goal_relevance"],
            _CUR_V["norm_congruence"],
            _CUR_V["coping_signed"],
            _CUR_V["novelty"],
        ]
    )
    return f @ w


def _baseline_arousal(rows: list[dict[str, Any]]) -> np.ndarray:
    f = _arousal_theory_features(rows)
    w = np.array([_CUR_A["abs_novelty"], _CUR_A["one_minus_coping"], _CUR_A["self_relevance"]])
    return f @ w


# ── linear fit helpers (numpy lstsq) ───────────────────────────────────────────


def _fit(x: np.ndarray, y: np.ndarray, *, intercept: bool) -> np.ndarray:
    """Least-squares coefficients; if intercept, last column is the bias term."""
    a = np.hstack([x, np.ones((x.shape[0], 1))]) if intercept else x
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return coef


def _apply(x: np.ndarray, coef: np.ndarray, *, intercept: bool) -> np.ndarray:
    a = np.hstack([x, np.ones((x.shape[0], 1))]) if intercept else x
    return a @ coef


# ── metrics ────────────────────────────────────────────────────────────────────


def _pearson(pred: np.ndarray, oracle: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(pred, oracle)
    return float(c[0, 1])


def _metrics(pred: np.ndarray, oracle: np.ndarray, *, sign: bool) -> dict[str, float]:
    res = pred - oracle
    bias, lo, hi = bootstrap_ci(res.tolist(), n_bootstrap=_N_BOOTSTRAP, seed=_SEED)
    out = {
        "bias": bias,
        "bias_ci_lo": lo,
        "bias_ci_hi": hi,
        "mae": float(np.mean(np.abs(res))),
        "pearson_r": _pearson(pred, oracle),
    }
    if sign:
        out["sign_acc"] = float(np.mean((pred >= 0) == (oracle >= 0)))
    return out


# ── model definitions ──────────────────────────────────────────────────────────


def _predict_axis(
    model: str,
    axis: str,
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> np.ndarray:
    """Return test-set predictions for (model, axis). Valence intercept constrained to 0 (G1)."""
    is_val = axis == "valence"
    feat = _valence_theory_features if is_val else _arousal_theory_features
    y_train = np.array([r[f"oracle_{axis}"] for r in train])
    intercept = not is_val  # arousal: free intercept; valence: constrained to 0

    if model == "M0":  # current weights + offset (offset only if arousal)
        base = _baseline_valence if is_val else _baseline_arousal
        if is_val:
            return base(test)
        offset = float(np.mean(y_train - base(train)))
        return base(test) + offset
    if model == "M1":  # recalibrated theoretical-feature weights
        coef = _fit(feat(train), y_train, intercept=intercept)
        return _apply(feat(test), coef, intercept=intercept)
    if model == "M2":  # free linear on raw SECs
        coef = _fit(_raw_features(train), y_train, intercept=intercept)
        return _apply(_raw_features(test), coef, intercept=intercept)
    raise ValueError(model)


# ── runner ─────────────────────────────────────────────────────────────────────


def run(dump: Path, split: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines() if line]
    sp = json.loads(split.read_text(encoding="utf-8"))
    train_ids, test_ids = set(sp["train"]), set(sp["test"])
    train = [r for r in rows if r["scenario_id"] in train_ids]
    test = [r for r in rows if r["scenario_id"] in test_ids]

    ov = np.array([r["oracle_valence"] for r in test])
    oa = np.array([r["oracle_arousal"] for r in test])

    results: dict[str, Any] = {
        "n_train_events": len(train),
        "n_test_events": len(test),
        "seed": _SEED,
        "axes": {},
    }
    for axis, oracle in (("valence", ov), ("arousal", oa)):
        entry = {
            "baseline": _metrics(
                (_baseline_valence if axis == "valence" else _baseline_arousal)(test),
                oracle,
                sign=(axis == "valence"),
            )
        }
        for model in ("M0", "M1", "M2"):
            pred = _predict_axis(model, axis, train, test)
            entry[model] = _metrics(pred, oracle, sign=(axis == "valence"))
        results["axes"][axis] = entry

    results["m1_coefficients"] = _m1_coefficients(train)
    results["cv_m1"] = _cv_m1(rows, sp["train"])
    return results


def _m1_coefficients(train: list[dict[str, Any]]) -> dict[str, Any]:
    """Fitted M1 weights for promotion. Valence intercept constrained to 0 (G1)."""
    yv = np.array([r["oracle_valence"] for r in train])
    ya = np.array([r["oracle_arousal"] for r in train])
    cv = _fit(_valence_theory_features(train), yv, intercept=False)
    ca = _fit(_arousal_theory_features(train), ya, intercept=True)
    return {
        "valence": {
            "goal_relevance": float(cv[0]),
            "norm_congruence": float(cv[1]),
            "coping_signed": float(cv[2]),
            "novelty": float(cv[3]),
            "intercept": 0.0,
        },
        "arousal": {
            "abs_novelty": float(ca[0]),
            "one_minus_coping": float(ca[1]),
            "self_relevance": float(ca[2]),
            "intercept": float(ca[3]),
        },
    }


def _cv_m1(rows: list[dict[str, Any]], train_ids: list[str]) -> dict[str, Any]:
    """5-fold CV by scenario on the train split: mean±sd of M1 test-fold metrics + coef sd."""
    rng = np.random.default_rng(_SEED)
    ids = sorted(train_ids)
    perm = list(rng.permutation(len(ids)))
    folds = [[ids[i] for i in perm[k::5]] for k in range(5)]
    by_sid: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_sid.setdefault(r["scenario_id"], []).append(r)

    agg: dict[str, list[float]] = {
        f"{ax}_{m}": [] for ax in ("valence", "arousal") for m in ("bias", "mae", "r")
    }
    coef_v: list[np.ndarray] = []
    coef_a: list[np.ndarray] = []
    for k in range(5):
        hold = set(folds[k])
        tr = [r for sid in ids if sid not in hold for r in by_sid[sid]]
        te = [r for sid in folds[k] for r in by_sid[sid]]
        for axis in ("valence", "arousal"):
            pred = _predict_axis("M1", axis, tr, te)
            oracle = np.array([r[f"oracle_{axis}"] for r in te])
            res = pred - oracle
            agg[f"{axis}_bias"].append(float(np.mean(res)))
            agg[f"{axis}_mae"].append(float(np.mean(np.abs(res))))
            agg[f"{axis}_r"].append(_pearson(pred, oracle))
        coef_v.append(
            _fit(
                _valence_theory_features(tr),
                np.array([r["oracle_valence"] for r in tr]),
                intercept=False,
            )
        )
        coef_a.append(
            _fit(
                _arousal_theory_features(tr),
                np.array([r["oracle_arousal"] for r in tr]),
                intercept=True,
            )
        )

    def _ms(xs: list[float]) -> dict[str, float]:
        a = np.array(xs)
        return {"mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1))}

    return {
        "metrics": {k: _ms(v) for k, v in agg.items()},
        "coef_sd_valence": [float(s) for s in np.std(np.array(coef_v), axis=0, ddof=1)],
        "coef_sd_arousal": [float(s) for s in np.std(np.array(coef_a), axis=0, ddof=1)],
    }


def render_md(res: dict[str, Any]) -> str:
    lines = [
        "# Addendum O — SEC→Affect Mapping Recalibration",
        "",
        f"Train events: {res['n_train_events']} | Test events: {res['n_test_events']} | "
        f"seed: {res['seed']}",
        "",
    ]
    for axis, entry in res["axes"].items():
        lines += [
            f"## {axis.capitalize()} (held-out test)",
            "",
            "| Model | bias | MAE | Pearson r |",
            "|---|---|---|---|",
        ]
        for name in ("baseline", "M0", "M1", "M2"):
            m = entry[name]
            lines.append(f"| {name} | {m['bias']:+.3f} | {m['mae']:.3f} | {m['pearson_r']:.3f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)

    res = run(args.dump, args.split)
    args.out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    args.out_md.write_text(render_md(res), encoding="utf-8")
    print(render_md(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
