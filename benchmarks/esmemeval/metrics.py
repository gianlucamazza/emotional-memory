"""Retrieval metrics for Addendum X2 (ES-MemEval).

Two families, both over binary session-level relevance:

- **Upstream-verbatim** ``upstream_recall_at_k`` / ``upstream_ndcg_at_k``:
  formula-identical to ``QaRetrievalExperiment.recall`` / ``.ndcg`` in the
  ES-MemEval repo, so our numbers are comparable to the published baselines.
  Their nDCG carries a non-standard rank offset — DCG enumerates ranks from
  ``log2(4)`` while IDCG starts at ``log2(2)`` — so attainable nDCG < 1. The
  distortion is identical for both arms and preserves ordering (pre-registration
  §Hypotheses). The primary metric Hx2 is ``upstream_ndcg_at_k(..., k=4)``.
- **Standard** MAP/MRR/nDCG/Recall/Precision (same implementations as
  ``benchmarks/madialbench/metrics.py``, generalized to hashable ids), reported
  as a secondary check.

Unit-tested against hand-computed examples in ``tests/test_esmemeval.py``.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from collections.abc import Set as AbstractSet

UPSTREAM_K_GRID = (2, 4, 6)  # the grid upstream reports (their Table 4 headline: k=4)
STD_K_GRID = (1, 3, 4, 5, 10)
PRIMARY_K = 4


def upstream_recall_at_k(
    gold: AbstractSet[Hashable], retrieved: Sequence[Hashable], k: int
) -> float:
    """Verbatim ``QaRetrievalExperiment.recall`` (identical to standard recall)."""
    prediction = list(retrieved[:k])
    hit = set(prediction) & set(gold)
    return len(hit) / len(gold) if len(gold) != 0 else 0


def upstream_ndcg_at_k(
    gold: AbstractSet[Hashable], retrieved: Sequence[Hashable], k: int
) -> float:
    """Verbatim ``QaRetrievalExperiment.ndcg``, non-standard rank offset included."""
    prediction = list(retrieved[:k])
    dcg = 0.0
    for index, item in enumerate(prediction, 2):
        if item in gold:
            dcg += 1 / math.log2(index + 2)
    idcg = sum(1 / math.log2(i) for i in range(2, min(len(gold), k) + 2))
    return dcg / idcg if idcg > 0 else 0


def average_precision_at_k(
    gold: AbstractSet[Hashable], retrieved: Sequence[Hashable], k: int
) -> float:
    if not gold:
        return 0.0
    retrieved = list(retrieved[:k])
    score = 0.0
    num_hits = 0.0
    for i, result in enumerate(retrieved):
        if result in gold and result not in retrieved[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(gold), k)


def reciprocal_rank_at_k(
    gold: AbstractSet[Hashable], retrieved: Sequence[Hashable], k: int
) -> float:
    for i, item in enumerate(retrieved[:k]):
        if item in gold:
            return 1.0 / (i + 1.0)
    return 0.0


def _dcg(relevances: Sequence[int], k: int) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances[:k]))


def ndcg_at_k(gold: AbstractSet[Hashable], retrieved: Sequence[Hashable], k: int) -> float:
    retrieved = list(retrieved[:k])
    relevances = [1 if item in gold else 0 for item in retrieved]
    idcg = _dcg(sorted(relevances, reverse=True), k)
    if not idcg:
        return 0.0
    return _dcg(relevances, k) / idcg


def recall_at_k(gold: AbstractSet[Hashable], retrieved: Sequence[Hashable], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def precision_at_k(gold: AbstractSet[Hashable], retrieved: Sequence[Hashable], k: int) -> float:
    return len(set(retrieved[:k]) & gold) / k


UPSTREAM_METRICS = {
    "u_recall": upstream_recall_at_k,
    "u_ndcg": upstream_ndcg_at_k,
}

STD_METRICS = {
    "map": average_precision_at_k,
    "mrr": reciprocal_rank_at_k,
    "ndcg": ndcg_at_k,
    "recall": recall_at_k,
    "precision": precision_at_k,
}
