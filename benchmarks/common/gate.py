"""Query-affect-conditioned gate (Addendum Y).

The gate is a per-query router: for each query, if the appraised query is
affectively **neutral** (``abs(valence) < tau``) the retrieval falls back to the
``naive_cosine`` arm; otherwise it uses the ``aft_query_appraised`` arm. Because
the gate routes each query *wholly* to one arm, and both arms already produce
per-query scores, the gated arm is an **exact per-query selection** between two
already-computed score vectors — no new retrieval, no new LLM calls, and exact
cosine-equivalence on neutral queries (pre-registration Addendum Y §Protocol).

All functions are pure and operate on aligned per-query lists (same query order
for ``cosine``, ``aft``, and ``valences``). Each harness assembles those three
aligned lists in its own way and calls :func:`gate_analysis`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_TAU = 0.2
TAU_GRID = (0.1, 0.2, 0.3)


def gated_scores(
    cosine: Sequence[float],
    aft: Sequence[float],
    valences: Sequence[float],
    tau: float,
) -> list[float]:
    """Per-query gated scores: cosine when ``abs(valence) < tau`` else aft.

    The three inputs must be aligned (same query order and length). ``tau <= 0``
    yields the pure aft arm; ``tau > 1`` yields the pure cosine arm (valence is
    on ``[-1, 1]``).
    """
    if not (len(cosine) == len(aft) == len(valences)):
        raise ValueError(
            f"misaligned lengths: cosine={len(cosine)} aft={len(aft)} valences={len(valences)}"
        )
    return [c if abs(v) < tau else a for c, a, v in zip(cosine, aft, valences, strict=True)]


def routing_rate(valences: Sequence[float], tau: float) -> float:
    """Fraction of queries routed to cosine (``abs(valence) < tau``)."""
    n = len(valences)
    if n == 0:
        return 0.0
    return sum(1 for v in valences if abs(v) < tau) / n


def recovery_fraction(gated_mean: float, aft_mean: float, cosine_mean: float) -> float:
    """Share of the always-on penalty the gate removes.

    On an off-regime corpus the always-on penalty is ``cosine_mean - aft_mean``
    (cosine ahead of AFT). The gate recovers
    ``(gated_mean - aft_mean) / (cosine_mean - aft_mean)``: 0.0 = no recovery
    (gated == aft), 1.0 = full recovery (gated == cosine). Returns ``nan`` when
    there is no penalty to recover (denominator ~ 0) — the ratio is undefined and
    the closure should read the raw gated-vs-cosine gap instead.
    """
    denom = cosine_mean - aft_mean
    if abs(denom) < 1e-12:
        return float("nan")
    return (gated_mean - aft_mean) / denom


@dataclass(frozen=True)
class GateResult:
    """Gated arm summary at one tau, for one corpus."""

    tau: float
    gated_scores: tuple[float, ...]
    routing_rate: float
    gated_mean: float
    cosine_mean: float
    aft_mean: float
    recovery_fraction: float


def gate_analysis(
    cosine: Sequence[float],
    aft: Sequence[float],
    valences: Sequence[float],
    tau: float = DEFAULT_TAU,
) -> GateResult:
    """Compose the gated arm and summarize it (means, routing, recovery)."""
    g = gated_scores(cosine, aft, valences, tau)
    n = len(g)
    gm = sum(g) / n if n else 0.0
    cm = sum(cosine) / n if n else 0.0
    am = sum(aft) / n if n else 0.0
    return GateResult(
        tau=tau,
        gated_scores=tuple(g),
        routing_rate=routing_rate(valences, tau),
        gated_mean=gm,
        cosine_mean=cm,
        aft_mean=am,
        recovery_fraction=recovery_fraction(gm, am, cm),
    )


def _contrast(
    a: Sequence[float], b: Sequence[float], *, n_bootstrap: int, seed: int
) -> dict[str, float]:
    """Paired one-tailed bootstrap contrast a - b (reuses common.statistics)."""
    from benchmarks.common.statistics import cohens_d_paired, paired_bootstrap_diff

    diff, lo, hi, p_two = paired_bootstrap_diff(a, b, n_bootstrap=n_bootstrap, seed=seed)
    p_one = p_two / 2.0 if diff >= 0 else 1.0 - p_two / 2.0
    return {
        "delta": diff,
        "ci_lower": lo,
        "ci_upper": hi,
        "p_bootstrap_onetail": p_one,
        "cohens_d": cohens_d_paired(a, b),
    }


def gate_report(
    cosine: Sequence[float],
    aft: Sequence[float],
    valences: Sequence[float],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    primary_tau: float = DEFAULT_TAU,
    tau_grid: Sequence[float] = TAU_GRID,
) -> dict[str, object]:
    """Full gated-arm report for one corpus: primary-tau contrasts + sensitivity.

    ``gated vs cosine`` and ``gated vs aft`` are paired one-tailed bootstrap
    contrasts (a - b, positive = gated better). ``recovery_fraction`` and
    ``routing_rate`` are the descriptive gate diagnostics. The sensitivity block
    recomputes the same at every tau in ``tau_grid``.
    """

    def _at(tau: float) -> dict[str, object]:
        r = gate_analysis(cosine, aft, valences, tau)
        return {
            "tau": tau,
            "routing_rate": r.routing_rate,
            "gated_mean": r.gated_mean,
            "cosine_mean": r.cosine_mean,
            "aft_mean": r.aft_mean,
            "recovery_fraction": r.recovery_fraction,
            "gated_vs_cosine": _contrast(
                r.gated_scores, cosine, n_bootstrap=n_bootstrap, seed=seed
            ),
            "gated_vs_aft": _contrast(r.gated_scores, aft, n_bootstrap=n_bootstrap, seed=seed),
        }

    return {
        "n_queries": len(cosine),
        "primary_tau": primary_tau,
        "primary": _at(primary_tau),
        "sensitivity": [_at(t) for t in tau_grid],
    }
