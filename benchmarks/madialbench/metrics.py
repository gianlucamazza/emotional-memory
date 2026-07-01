"""Retrieval metrics replicating MADial-Bench's `embedding_score_new.py` verbatim.

Binary relevance against the gold ``relevant-id`` set; AP normalized by
``min(len(gold), k)``; nDCG with binary gains. Kept formula-identical to the
benchmark's own implementation so our numbers are comparable to the published
baselines (unit-tested against hand-computed examples in
``tests/test_madialbench.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from collections.abc import Set as AbstractSet

K_GRID = (1, 3, 5, 10)


def average_precision_at_k(gold: AbstractSet[int], retrieved: Sequence[int], k: int) -> float:
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


def reciprocal_rank_at_k(gold: AbstractSet[int], retrieved: Sequence[int], k: int) -> float:
    for i, item in enumerate(retrieved[:k]):
        if item in gold:
            return 1.0 / (i + 1.0)
    return 0.0


def _dcg(relevances: Sequence[int], k: int) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances[:k]))


def ndcg_at_k(gold: AbstractSet[int], retrieved: Sequence[int], k: int) -> float:
    retrieved = list(retrieved[:k])
    relevances = [1 if item in gold else 0 for item in retrieved]
    idcg = _dcg(sorted(relevances, reverse=True), k)
    if not idcg:
        return 0.0
    return _dcg(relevances, k) / idcg


def recall_at_k(gold: AbstractSet[int], retrieved: Sequence[int], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def precision_at_k(gold: AbstractSet[int], retrieved: Sequence[int], k: int) -> float:
    return len(set(retrieved[:k]) & gold) / k


METRICS = {
    "map": average_precision_at_k,
    "mrr": reciprocal_rank_at_k,
    "ndcg": ndcg_at_k,
    "recall": recall_at_k,
    "precision": precision_at_k,
}
