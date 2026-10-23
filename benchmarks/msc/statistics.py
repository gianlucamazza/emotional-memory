"""Stable exact paired binary inference and prospective McNemar power, NumPy only."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray


@lru_cache(maxsize=8)
def _logfacts(n: int) -> NDArray[np.float64]:
    return np.array([math.lgamma(i + 1) for i in range(n + 1)])


def binomial_pmf(n: int, p: float) -> NDArray[np.float64]:
    if n < 0 or not 0 <= p <= 1:
        raise ValueError("Invalid binomial parameters")
    if p in (0, 1):
        result = np.zeros(n + 1)
        result[0 if p == 0 else n] = 1
        return result
    k = np.arange(n + 1)
    facts = _logfacts(n)
    logp = facts[n] - facts[k] - facts[n - k] + k * math.log(p)
    logp += (n - k) * math.log1p(-p)
    pmf = np.exp(logp)
    return pmf / pmf.sum()


def mcnemar(a_only: int, b_only: int) -> float:
    if min(a_only, b_only) < 0:
        raise ValueError("Negative discordant count")
    return min(1.0, 2 * float(binomial_pmf(a_only + b_only, 0.5)[: min(a_only, b_only) + 1].sum()))


def discordance_upper(k: int, n: int) -> float:
    """One-sided 95% Clopper-Pearson upper limit, not estimated effect size."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("Invalid pilot counts")
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binomial_pmf(n, mid)[: k + 1].sum() > 0.05:
            lo = mid
        else:
            hi = mid
    return hi


def plan_n(discordant: int, pilot_n: int, *, maximum: int) -> dict[str, float | int | str]:
    """Smallest exact unconditional power >= .80 at delta .05 and pilot upper q."""
    if maximum < 1:
        raise ValueError("Positive available-sample cap required")
    q = max(0.05, discordance_upper(discordant, pilot_n))
    p_win = (q + 0.05) / (2 * q)
    conditional = np.zeros(maximum + 1)
    for m in range(1, maximum + 1):
        null = binomial_pmf(m, 0.5)
        cdf = np.cumsum(null)
        lower = np.minimum(np.arange(m + 1), m - np.arange(m + 1))
        reject = 2 * cdf[lower] <= 0.05
        conditional[m] = binomial_pmf(m, p_win)[reject].sum()
        power = float(binomial_pmf(m, q) @ conditional[: m + 1])
        if power >= 0.8:
            return {
                "status": "POWERED",
                "n": m,
                "power": power,
                "discordance_upper": q,
                "delta": 0.05,
                "alpha": 0.05,
                "pilot_n": pilot_n,
                "pilot_discordant": discordant,
                "maximum": maximum,
            }
    return {
        "status": "INFEASIBLE",
        "discordance_upper": q,
        "delta": 0.05,
        "alpha": 0.05,
        "pilot_n": pilot_n,
        "pilot_discordant": discordant,
        "maximum": maximum,
    }


def paired_report(a: list[bool], b: list[bool]) -> dict[str, object]:
    if not a or len(a) != len(b):
        raise ValueError("Need nonempty aligned pairs")
    counts = [
        sum(x and not y for x, y in zip(a, b, strict=True)),
        sum(y and not x for x, y in zip(a, b, strict=True)),
        sum(x and y for x, y in zip(a, b, strict=True)),
        sum(not x and not y for x, y in zip(a, b, strict=True)),
    ]
    n = len(a)
    samples = np.random.default_rng(20261004).multinomial(n, np.array(counts) / n, 10000)
    lo, hi = np.quantile((samples[:, 0] - samples[:, 1]) / n, [0.025, 0.975])
    delta = (counts[0] - counts[1]) / n
    p = mcnemar(counts[0], counts[1])
    return {
        "n": n,
        "a_top1": sum(a) / n,
        "b_top1": sum(b) / n,
        "a_only": counts[0],
        "b_only": counts[1],
        "both": counts[2],
        "neither": counts[3],
        "delta": delta,
        "ci95": [float(lo), float(hi)],
        "p_two_sided": p,
        "ci_method": "paired percentile bootstrap, 10000 samples, seed 20261004",
        "superiority": delta > 0 and p < 0.05,
        "five_point_interpretation": "excluded"
        if hi < 0.05
        else ("supported" if lo >= 0.05 else "compatible_not_established"),
    }
