"""Deterministic unit tests for the Addendum W affine arousal calibration.

No LLM: the calibration consumes a per-item prediction dump and is pure numpy, so
its math is validated here on synthetic data with a known scale/offset mismatch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmarks.arousal_calibration.runner import (
    affine_fit,
    kfold_oof_calibrate,
    run,
)


def _synthetic_dump(n: int = 200, seed: int = 0) -> dict[str, Any]:
    """direct_vad arousal: high r with human but wrong scale/offset (bad MAE).

    scherer arousal: near-constant around the mean (low r, moderate MAE).
    """
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.33, 0.80, n)  # EmoBank-like narrow human arousal
    direct = (y - 0.5) * 2.0 + 0.45 + 0.02 * rng.standard_normal(n)  # widened + shifted
    scherer = 0.5 + 0.05 * rng.standard_normal(n)  # almost constant near the mean
    items = []
    for i in range(n):
        split = "train" if i < int(0.7 * n) else ("dev" if i % 2 == 0 else "test")
        items.append(
            {
                "id": i,
                "split": split,
                "human": {"valence": 0.0, "arousal": float(y[i]), "dominance": 0.5},
                "direct_vad": {"valence": 0.0, "arousal": float(direct[i]), "dominance": 0.5},
                "scherer_m1": {"valence": 0.0, "arousal": float(scherer[i]), "dominance": 0.5},
            }
        )
    return {"dataset": "synthetic", "version": "test", "n": n, "items": items}


def test_affine_fit_recovers_known_line() -> None:
    x = np.linspace(0.0, 1.0, 50)
    y = 0.4 * x + 0.3
    a, b = affine_fit(x, y)
    assert a == np.float64(a)  # returns plain float
    assert abs(a - 0.4) < 1e-9
    assert abs(b - 0.3) < 1e-9


def test_kfold_oof_recovers_perfect_line() -> None:
    # On a noiseless affine relationship, any training subset recovers the exact map,
    # so out-of-fold predictions must reproduce y.
    x = np.linspace(0.0, 1.0, 60)
    y = -0.25 * x + 0.6
    oof = kfold_oof_calibrate(x, y, k=5, seed=42)
    assert oof.shape == y.shape
    assert np.allclose(oof, y, atol=1e-9)


def test_calibration_reduces_mae_and_adopts() -> None:
    report = run(_synthetic_dump(), n_bootstrap=500, seed=42)
    native = report["protocols"]["native_split"]
    # Calibrated MAE strictly below raw MAE on held-out items.
    assert native["calibrated"]["mae"] < native["raw"]["mae"]
    # Pearson r preserved (affine map).
    assert abs(native["calibrated"]["pearson_r"] - native["raw"]["pearson_r"]) < 0.05
    # Pre-registered verdicts.
    assert report["verdicts"]["Hw1_cal_reduces_mae"] is True
    assert report["verdicts"]["Gw_slope_positive_and_r_preserved"] is True
    assert report["verdicts"]["adopt_calibration"] is True
    # Deployable slope is positive (no sign flip).
    assert report["deployable_coefficients"]["a"] > 0


def test_run_is_deterministic() -> None:
    dump = _synthetic_dump()
    r1 = run(dump, n_bootstrap=300, seed=7)
    r2 = run(dump, n_bootstrap=300, seed=7)
    assert r1 == r2


def test_cv_protocol_present_without_split() -> None:
    # A dump lacking split labels still yields the CV protocol (and skips native_split).
    dump = _synthetic_dump()
    for it in dump["items"]:
        it["split"] = None
    report = run(dump, n_bootstrap=300, seed=1)
    assert "native_split" not in report["protocols"]
    assert "kfold_cv" in report["protocols"]
