"""Tests for the Addendum X MADial-Bench harness (loader + replicated metrics)."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.madialbench.dataset import (
    DIALOGUE_FILE,
    MEMORY_FILE,
    TIME_HEADER,
    load_dataset,
)
from benchmarks.madialbench.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

# ─── Metrics: hand-computed examples (formula-identical to embedding_score_new.py) ──

GOLD = {1, 78, 105}


def test_ap_at_k_hand_computed() -> None:
    # Hits at ranks 1 and 3: AP@5 = (1/1 + 2/3) / min(3, 5) = 5/9
    assert average_precision_at_k(GOLD, [1, 2, 78, 3, 4], 5) == pytest.approx(5 / 9)
    # No hits.
    assert average_precision_at_k(GOLD, [2, 3, 4], 5) == 0.0
    # AP normalizes by min(len(gold), k): single-slot cutoff with a hit at rank 1.
    assert average_precision_at_k(GOLD, [1, 78], 1) == pytest.approx(1.0)
    # Empty gold guard.
    assert average_precision_at_k(set(), [1], 5) == 0.0


def test_mrr_at_k_hand_computed() -> None:
    assert reciprocal_rank_at_k(GOLD, [2, 3, 78], 5) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k(GOLD, [2, 3], 5) == 0.0
    # Cutoff: hit exists but beyond k.
    assert reciprocal_rank_at_k(GOLD, [2, 3, 78], 2) == 0.0


def test_ndcg_at_k_hand_computed() -> None:
    # Hits at ranks 2 and 3 of 3 retrieved: DCG = 1/log2(3) + 1/log2(4),
    # IDCG = 1/log2(2) + 1/log2(3).
    import math

    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(GOLD, [2, 1, 78], 3) == pytest.approx(dcg / idcg)
    # Perfect ranking → 1.0; no hits → 0.0.
    assert ndcg_at_k(GOLD, [1, 78, 105], 3) == pytest.approx(1.0)
    assert ndcg_at_k(GOLD, [2, 3, 4], 3) == 0.0


def test_recall_precision_at_k_hand_computed() -> None:
    assert recall_at_k(GOLD, [1, 2, 78], 3) == pytest.approx(2 / 3)
    assert precision_at_k(GOLD, [1, 2, 78], 3) == pytest.approx(2 / 3)
    assert precision_at_k(GOLD, [1], 5) == pytest.approx(1 / 5)


# ─── Loader: pinned data integrity and protocol-exact query construction ──────────


def test_load_dataset_counts_and_integrity() -> None:
    ds = load_dataset()
    assert len(ds.memories) == 160
    assert len(ds.queries) == 160
    bank = {m.memory_id for m in ds.memories}
    assert bank == set(range(1, 161))
    for q in ds.queries:
        assert q.gold_ids <= bank
        assert 1 <= len(q.gold_ids) <= 7
        assert q.user_id in (1, 2)
    # Gold sets are user-disjoint (pre-registration §Protocol).
    gold_u1 = set().union(*(q.gold_ids for q in ds.queries if q.user_id == 1))
    gold_u2 = set().union(*(q.gold_ids for q in ds.queries if q.user_id == 2))
    assert not gold_u1 & gold_u2


def test_query_text_replicates_embedding_py() -> None:
    ds = load_dataset()
    q0 = ds.queries[0]
    assert q0.text.startswith(TIME_HEADER)
    # Context strictly before the first test turn: the assistant's test reply
    # must not leak into the query.
    assert q0.text.count("<Bart>") + q0.text.count("<Lisa>") > 0
    # First dialogue: test-turn [8] → exactly turns 0..7 concatenated.
    import json

    with Path(DIALOGUE_FILE).open(encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    expected = TIME_HEADER + "".join(first["dialogue"][: first["test-turn"][0]])
    assert q0.text == expected


def test_loader_rejects_tampered_data(tmp_path: Path) -> None:
    tampered = tmp_path / "MADial-Bench-en-dialogue.jsonl"
    tampered.write_text(Path(DIALOGUE_FILE).read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        load_dataset(dialogue_file=tampered, memory_file=MEMORY_FILE)


def test_loader_hash_check_can_be_bypassed_only_explicitly(tmp_path: Path) -> None:
    # verify_hashes=False exists for unit tests only; the runner never sets it.
    ds = load_dataset(verify_hashes=False)
    assert len(ds.queries) == 160
