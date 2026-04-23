"""Bootstrap confidence intervals and significance tests for benchmark metrics.

Provides paired bootstrap CI, McNemar exact test, and formatting utilities.
All functions are pure; use a seeded numpy Generator for reproducibility.
No scipy dependency — only numpy (hard dep) + stdlib math.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

DEFAULT_N_BOOTSTRAP: int = 2000
DEFAULT_CI: float = 0.95


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CI,
    seed: int = 0,
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a scalar statistic of *values*.

    Returns ``(point, lo, hi)`` where ``point = statistic(values)``.
    On empty or singleton input returns ``(point, point, point)`` — no crash.
    Works for both proportions (0/1 Bernoulli) and continuous means.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(arr))
    if n == 1:
        return (point, point, point)

    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_stats = np.array([statistic(arr[idx]) for idx in indices], dtype=np.float64)
    lo = float(np.quantile(boot_stats, alpha))
    hi = float(np.quantile(boot_stats, 1.0 - alpha))
    return (point, lo, hi)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CI,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Paired bootstrap on per-item differences ``a[i] - b[i]``.

    Returns ``(diff_point, lo, hi, p_two_sided)``.
    Two-sided p-value tests H0: mean_diff = 0 via mean-centered bootstrap.
    """
    if len(a) != len(b):
        raise ValueError(f"Sequences must have equal length: {len(a)} != {len(b)}")
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    d = arr_a - arr_b
    n = len(d)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    diff_point = float(np.mean(d))
    if n == 1:
        return (diff_point, diff_point, diff_point, 1.0)

    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_diffs = np.array([float(np.mean(d[idx])) for idx in indices], dtype=np.float64)

    lo = float(np.quantile(boot_diffs, alpha))
    hi = float(np.quantile(boot_diffs, 1.0 - alpha))

    # Two-sided p-value via mean-centering: H0 is mean_diff = 0
    boot_centered = boot_diffs - diff_point
    p = float(np.mean(np.abs(boot_centered) >= abs(diff_point)))
    p = min(max(p, 0.0), 1.0)
    return (diff_point, lo, hi, p)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value on discordant pairs (binomial, p=0.5).

    ``only_a``: pairs where system A is correct and B is wrong.
    ``only_b``: pairs where system B is correct and A is wrong.
    Concordant pairs (both right / both wrong) are irrelevant.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    # Two-sided exact: 2 * P(X <= k) where X ~ Binomial(n, 0.5)
    half = 0.5**n
    p = 2.0 * sum(math.comb(n, i) * half for i in range(k + 1))
    return min(p, 1.0)


def format_point_ci(point: float, lo: float, hi: float, *, dp: int = 2) -> str:
    """Render ``0.60 [0.45, 0.75]`` at *dp* decimal places."""
    fmt = f"{{:.{dp}f}}"
    return f"{fmt.format(point)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def ci_payload(
    point: float,
    lo: float,
    hi: float,
    *,
    n_bootstrap: int,
    method: str = "bootstrap_percentile",
) -> dict[str, Any]:
    """Build JSON sub-object with full-precision CI data."""
    return {
        "point": point,
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_method": method,
        "n_bootstrap": n_bootstrap,
    }
