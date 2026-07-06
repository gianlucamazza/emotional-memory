"""Tests for the Addendum X2 ES-MemEval harness (loader + pools + metrics)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.esmemeval.dataset import (
    EVO_EMO_FILE,
    N_IN_FAMILY,
    N_QUESTIONS,
    N_SEEKERS,
    N_SESSIONS,
    POOL_SIZE,
    build_pools,
    emotion_class,
    load_dataset,
)
from benchmarks.esmemeval.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    upstream_ndcg_at_k,
    upstream_recall_at_k,
)

# ─── Metrics: hand-computed examples ──────────────────────────────────────────

GOLD = {"a", "b", "c"}


def test_upstream_ndcg_hand_computed() -> None:
    # Verbatim quirk: DCG enumerates from log2(4), IDCG from log2(2) — a hit at
    # rank 1 with a single gold scores (1/log2(4)) / (1/log2(2)) = 0.5, not 1.0.
    assert upstream_ndcg_at_k({"a"}, ["a"], 4) == pytest.approx(0.5)
    # Hits at ranks 1 and 3: DCG = 1/log2(4) + 1/log2(6);
    # IDCG (|gold|=3, k=4) = 1/log2(2) + 1/log2(3) + 1/log2(4).
    dcg = 1 / math.log2(4) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert upstream_ndcg_at_k(GOLD, ["a", "x", "b", "y"], 4) == pytest.approx(dcg / idcg)
    # No hits / empty gold.
    assert upstream_ndcg_at_k(GOLD, ["x", "y"], 4) == 0.0
    assert upstream_ndcg_at_k(set(), ["a"], 4) == 0


def test_upstream_recall_hand_computed() -> None:
    assert upstream_recall_at_k(GOLD, ["a", "x", "b"], 3) == pytest.approx(2 / 3)
    assert upstream_recall_at_k(GOLD, ["a", "x", "b"], 2) == pytest.approx(1 / 3)
    assert upstream_recall_at_k(set(), ["a"], 3) == 0


def test_std_metrics_hand_computed() -> None:
    # Same implementations as the madialbench module (generalized ids).
    assert average_precision_at_k(GOLD, ["a", "x", "b", "y", "z"], 5) == pytest.approx(5 / 9)
    assert reciprocal_rank_at_k(GOLD, ["x", "y", "b"], 5) == pytest.approx(1 / 3)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(GOLD, ["x", "a", "b"], 3) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(GOLD, ["a", "b", "c"], 3) == pytest.approx(1.0)
    assert recall_at_k(GOLD, ["a", "x", "b"], 3) == pytest.approx(2 / 3)
    assert precision_at_k(GOLD, ["a"], 5) == pytest.approx(1 / 5)


# ─── Loader: pinned data integrity and protocol-exact document construction ──


def test_load_dataset_counts_and_integrity() -> None:
    ds = load_dataset()
    assert len(ds.sessions) == N_SESSIONS
    assert len(ds.queries) == N_IN_FAMILY
    assert ds.n_zero_gold == N_QUESTIONS - N_IN_FAMILY
    bank = {s.key for s in ds.sessions}
    assert len(bank) == N_SESSIONS  # composite keys are globally unique
    assert len({s.seeker_id for s in ds.sessions}) == N_SEEKERS
    for q in ds.queries:
        assert q.gold_keys  # in-family by construction
        assert q.gold_keys <= bank
        # Gold is same-seeker only (upstream message_indices semantics).
        assert all(k.split("/", 1)[0] == q.seeker_id for k in q.gold_keys)


def test_transcript_replicates_upstream_rendering() -> None:
    ds = load_dataset()
    with Path(EVO_EMO_FILE).open(encoding="utf-8") as fh:
        raw = json.load(fh)
    first_seeker = raw[0]
    first_sess = first_seeker["dialog_history"][0]
    expected = "\n".join(
        f"{first_seeker['basic_info']['name']}: {t['content']}"
        if t["role"] == "seeker"
        else f"supporter: {t['content']}"
        for t in first_sess["dialogue"]
    )
    key = f"{first_seeker['id']}/{first_sess['id']}"
    session = next(s for s in ds.sessions if s.key == key)
    assert session.text == expected
    assert session.timestamp == first_sess["timestamp"]
    # Emotion/topic/summary labels must NOT leak into the document text.
    assert str(first_sess["emotion"]) not in ("", session.text.split("\n")[0])


def test_loader_rejects_tampered_data(tmp_path: Path) -> None:
    tampered = tmp_path / "evo_emo.json"
    tampered.write_text(Path(EVO_EMO_FILE).read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        load_dataset(data_file=tampered)


def test_loader_hash_check_can_be_bypassed_only_explicitly() -> None:
    # verify_hashes=False exists for unit tests only; the runner never sets it.
    ds = load_dataset(verify_hashes=False)
    assert len(ds.queries) == N_IN_FAMILY


# ─── Candidate pools: upstream 50-candidate design, deterministic ────────────


def test_pools_deterministic_and_well_formed() -> None:
    ds = load_dataset()
    pools_a = build_pools(ds)
    pools_b = build_pools(ds)
    assert pools_a == pools_b  # same master seed → identical pools
    by_seeker: dict[str, set[str]] = {}
    for s in ds.sessions:
        by_seeker.setdefault(s.seeker_id, set()).add(s.key)
    for q in ds.queries:
        pool = pools_a[q.query_id]
        assert len(pool) == POOL_SIZE
        assert len(set(pool)) == POOL_SIZE  # no duplicates
        # All gold and ALL same-seeker sessions are in the pool (upstream:
        # correct + incorrect same-user, cross-user only as fill).
        assert q.gold_keys <= set(pool)
        assert by_seeker[q.seeker_id] <= set(pool)


def test_pools_differ_with_seed() -> None:
    ds = load_dataset()
    pools_a = build_pools(ds, seed=0)
    pools_b = build_pools(ds, seed=1)
    assert pools_a != pools_b


# ─── D1 label mapping (fixed ex-ante) ─────────────────────────────────────────


def test_emotion_class_mapping() -> None:
    assert emotion_class("relief") == "positive"
    assert emotion_class("Hope") == "positive"
    assert emotion_class("anxiety") == "negative"
    assert emotion_class("depression") == "negative"
    assert emotion_class("nostalgia") == "excluded"
    assert emotion_class("mixed emotions") == "excluded"
    assert emotion_class("guilt, worry") == "excluded"  # compound


def test_d1_classes_match_preregistered_prior() -> None:
    ds = load_dataset()
    classes = [emotion_class(s.emotion) for s in ds.sessions]
    # Ex-ante prior in the pre-registration (Amendment A1 exact counts under
    # the D1 mapping): near-uniformly negative bank, tiny positive class.
    assert classes.count("positive") == 7
    assert classes.count("negative") == 376
    assert classes.count("excluded") == 18
    assert len(classes) == N_SESSIONS
