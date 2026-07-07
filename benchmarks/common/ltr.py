"""Linear learning-to-rank over the 6 AFT retrieval signals (Addendum Z).

Pairwise logistic learning-to-rank (RankNet-style) with **k-fold cross-fitting**:
a 6-parameter linear weight vector ``w`` is fit on train folds and every query is
scored by a ``w`` that never saw it. Pure numpy, no new dependency.

The ranker is deliberately linear and tiny (6 parameters) so that ``w`` is the
exact, interpretable counterfactual to the fixed retrieval weight vector — in
particular the sign of ``w[1]`` (mood congruence) is a readout: negative means the
corpus rewards *counter-congruent* recall (the Addendum X support-mode residual).

All functions are pure. See ``benchmarks/preregistration_addendum_z_learned_profile.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

N_SIGNALS = (
    6  # s1 semantic, s2 mood-congruence, s3 affect-prox, s4 momentum, s5 recency, s6 resonance
)
DEFAULT_K = 5
DEFAULT_LR = 0.1
DEFAULT_STEPS = 2000
DEFAULT_L2 = 1.0

# Fixed weight vectors for the non-learned arms (applied to RAW signals).
COSINE_WEIGHTS: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
# The engine default base_weights (retrieval.RetrievalConfig.base_weights).
AFT_FIXED_WEIGHTS: tuple[float, ...] = (0.35, 0.25, 0.15, 0.10, 0.10, 0.05)


# A per-query metric: maps a ranking (candidate indices, best-first) to a scalar
# in [0, 1] (e.g. top1 hit, nDCG@k). Constructed by each corpus adapter.
MetricFn = Callable[[Sequence[int]], float]


@dataclass(frozen=True)
class QueryFeatures:
    """One query's ranking problem.

    features: (n_candidates, 6) raw AFT signals per candidate.
    gold:     (n_candidates,) binary relevance in {0.0, 1.0}.
    metric:   ranking (candidate indices, best-first) -> score in [0, 1].
    """

    features: NDArray[np.float64]
    gold: NDArray[np.float64]
    metric: MetricFn


def fit_pairwise(
    features_list: Sequence[NDArray[np.float64]],
    gold_list: Sequence[NDArray[np.float64]],
    *,
    lr: float = DEFAULT_LR,
    steps: int = DEFAULT_STEPS,
    l2: float = DEFAULT_L2,
) -> NDArray[np.float64]:
    """Fit a 6-dim weight vector by pairwise logistic LTR.

    For each query, every (relevant, non-relevant) candidate pair contributes a
    logistic loss ``softplus(-w·(x_pos - x_neg))`` — the pair is satisfied when the
    relevant candidate scores above the non-relevant one. Minimised by full-batch
    gradient descent with L2 regularisation. Deterministic (seed-free).
    """
    diffs: list[NDArray[np.float64]] = []
    for feats, gold in zip(features_list, gold_list, strict=True):
        pos = feats[gold > 0.5]
        neg = feats[gold <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue  # query with no contrast contributes no pairs
        d = pos[:, None, :] - neg[None, :, :]  # (n_pos, n_neg, 6)
        diffs.append(d.reshape(-1, N_SIGNALS))
    if not diffs:
        return np.zeros(N_SIGNALS, dtype=np.float64)
    diff = np.concatenate(diffs, axis=0)  # (n_pairs, 6)

    w = np.zeros(N_SIGNALS, dtype=np.float64)
    for _ in range(steps):
        s = np.clip(diff @ w, -30.0, 30.0)
        sig_neg = 1.0 / (1.0 + np.exp(s))  # sigmoid(-s) = -d(softplus(-s))/ds
        grad = -(diff * sig_neg[:, None]).mean(axis=0) + l2 * w
        w -= lr * grad
    return w


def score_fixed(queries: Sequence[QueryFeatures], weights: Sequence[float]) -> list[float]:
    """Per-query metric under a FIXED weight vector on raw signals (no fit)."""
    w = np.asarray(weights, dtype=np.float64)
    out: list[float] = []
    for q in queries:
        ranking = list(np.argsort(-(q.features @ w), kind="stable"))
        out.append(q.metric(ranking))
    return out


def cross_fit(
    queries: Sequence[QueryFeatures],
    *,
    k: int = DEFAULT_K,
    lr: float = DEFAULT_LR,
    steps: int = DEFAULT_STEPS,
    l2: float = DEFAULT_L2,
) -> tuple[list[float], NDArray[np.float64], float]:
    """Held-out learning-to-rank via k-fold cross-fitting.

    Deterministic seed-free fold assignment ``fold(i) = i % k``. For each fold:
    fit a z-score scaler + ``w`` on the *other* folds and score the held-out fold
    with that unseen ``w`` (features standardised by train-fold statistics). No
    query is ever scored by a ``w`` that saw it.

    Returns ``(heldout_metric_per_query, mean_weight_vector, in_sample_metric_mean)``.
    ``mean_weight_vector`` is the fold-averaged ``w`` (in standardised space; the
    sign of each component is meaningful). ``in_sample_metric_mean`` scores every
    query with the full-data ``w`` — the train→test generalization-gap reference.
    """
    n = len(queries)
    if n == 0:
        return [], np.zeros(N_SIGNALS, dtype=np.float64), float("nan")
    if k < 2:
        raise ValueError(f"k must be >= 2 for held-out cross-fitting, got {k}")

    folds = [i % k for i in range(n)]
    heldout = [0.0] * n
    fold_ws: list[NDArray[np.float64]] = []
    for f in range(k):
        train_idx = [i for i in range(n) if folds[i] != f]
        test_idx = [i for i in range(n) if folds[i] == f]
        if not test_idx or not train_idx:
            continue
        train_stack = np.concatenate([queries[i].features for i in train_idx], axis=0)
        mu = train_stack.mean(axis=0)
        sd = train_stack.std(axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        feats = [(queries[i].features - mu) / sd for i in train_idx]
        golds = [queries[i].gold for i in train_idx]
        w = fit_pairwise(feats, golds, lr=lr, steps=steps, l2=l2)
        fold_ws.append(w)
        for i in test_idx:
            x = (queries[i].features - mu) / sd
            ranking = list(np.argsort(-(x @ w), kind="stable"))
            heldout[i] = queries[i].metric(ranking)

    w_mean = np.mean(fold_ws, axis=0) if fold_ws else np.zeros(N_SIGNALS, dtype=np.float64)

    # In-sample reference: fit on ALL data (standardised globally) and score every query.
    all_stack = np.concatenate([q.features for q in queries], axis=0)
    mu_all = all_stack.mean(axis=0)
    sd_all = np.where(all_stack.std(axis=0) > 1e-9, all_stack.std(axis=0), 1.0)
    feats_all = [(q.features - mu_all) / sd_all for q in queries]
    golds_all = [q.gold for q in queries]
    w_all = fit_pairwise(feats_all, golds_all, lr=lr, steps=steps, l2=l2)
    in_sample = [
        q.metric(list(np.argsort(-((q.features - mu_all) / sd_all @ w_all), kind="stable")))
        for q in queries
    ]
    in_sample_mean = float(np.mean(in_sample)) if in_sample else float("nan")
    return heldout, w_mean, in_sample_mean
