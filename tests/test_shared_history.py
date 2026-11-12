"""Behavioral checks for the offline longitudinal replay contract."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from benchmarks.realistic.adapters.base import TokenHashEmbedder
from benchmarks.realistic.runner import (
    DEFAULT_OUT_JSON,
    DEFAULT_OUT_MD,
    DEFAULT_PROTOCOL,
    load_dataset,
    shared_history_output_paths,
)
from benchmarks.realistic.shared_history import (
    ARMS,
    DEFAULT_SEMANTIC_MODEL,
    DEFAULT_SHARED_HISTORY_EMBEDDER,
    EPOCH,
    HASH_EMBEDDER_VERSION,
    SharedHistoryAdapter,
    build_shared_history_embedder,
    controlled_runtime,
    describe_embedder,
    resolve_hash_salts,
    run_shared_history,
)
from emotional_memory import EmotionalMemoryConfig


def small_dataset():
    dataset = load_dataset()
    return dataset.model_copy(update={"scenarios": dataset.scenarios[:1]})


def test_shared_replay_exact_repeat_and_common_inputs():
    dataset = small_dataset()
    before = dataset.model_dump(mode="json")
    first = run_shared_history(dataset, seed=7, n_bootstrap=20)
    assert first == run_shared_history(dataset, seed=7, n_bootstrap=20)
    assert dataset.model_dump(mode="json") == before
    reports = [system["scenarios"][0] for system in first["systems"]]
    assert all(r["checkpoint"] == reports[0]["checkpoint"] for r in reports)
    inputs = [
        [(t["index"], t["timestamp"], t["input_sha256"]) for t in r["trace"]] for r in reports
    ]
    assert all(events == inputs[0] for events in inputs)
    for report in reports:
        queries = [t for t in report["trace"] if t["kind"] == "retrieve"]
        assert queries
        assert any(
            m["tag"]["retrieval_count"] > 0 for t in queries for m in t["snapshot"]["memories"]
        )


def test_shared_arms_do_not_share_store_config_or_tags():
    arms = [SharedHistoryAdapter(arm, "scenario", 0) for arm in ARMS]
    try:
        for arm in arms:
            arm.begin_session("one")
            arm.encode(
                memory_alias="a",
                content="same text",
                valence=0.8,
                arousal=0.8,
                metadata={"nested": [1]},
            )
        before = arms[1].snapshot()
        assert arms[0].config is not arms[1].config
        arms[0].retrieve("same text", top_k=1, valence=0.8, arousal=0.8)
        arms[0].engine.list_all()[0].metadata["nested"].append(2)
        assert arms[1].snapshot() == before
        assert arms[2].snapshot() == before
    finally:
        for arm in arms:
            arm.close()


def test_runtime_restored_on_exception():
    import emotional_memory.engine as engine
    import emotional_memory.models as models

    old_datetime, old_uuid = engine.datetime, models.uuid
    with pytest.raises(RuntimeError), controlled_runtime(EPOCH, "fixture-id"):
        assert engine.datetime.now(tz=UTC) == EPOCH
        raise RuntimeError("failure")
    assert engine.datetime is old_datetime
    assert models.uuid is old_uuid
    assert datetime.now(tz=UTC) != EPOCH


def test_shared_history_requires_recorded_query_affect():
    dataset = small_dataset().model_copy(deep=True)
    dataset.scenarios[0].sessions[-1].queries[0].state = None
    with pytest.raises(ValueError, match="recorded query affect"):
        run_shared_history(dataset, n_bootstrap=20)


def test_shared_history_resets_each_scenario():
    dataset = small_dataset()
    duplicate = dataset.scenarios[0].model_copy(deep=True)
    duplicate.scenario_id = "independent"
    dataset.scenarios.append(duplicate)
    result = run_shared_history(dataset, n_bootstrap=20)
    query_count = sum(
        len(session["queries"])
        for scenario in result["systems"][0]["scenarios"]
        for session in scenario["sessions"]
    )
    assert result["pairwise_comparisons"]
    assert all(row["n_queries"] == query_count for row in result["pairwise_comparisons"])
    for system in result["systems"]:
        first, second = system["scenarios"]
        assert first["checkpoint"] == second["checkpoint"]
        assert second["sessions"][0]["memory_count_start"] == 0
        assert not second["sessions"][0]["state_loaded_from_store"]


def test_shared_history_records_selected_dataset_path():
    path = Path("fixtures/custom-realistic.json")
    result = run_shared_history(small_dataset(), n_bootstrap=20, dataset_path=path)
    assert result["protocol"]["dataset"]["path"] == str(path)


def test_gated_arm_uses_configured_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, float] = {}

    def record(valence: float, tau: float) -> bool:
        seen["tau"] = tau
        return True

    monkeypatch.setattr("benchmarks.realistic.shared_history.is_query_affect_neutral", record)
    arm = SharedHistoryAdapter(
        "gated", "scenario", 0, EmotionalMemoryConfig(query_affect_gate_tau=0.8)
    )
    try:
        arm.retrieve("same text", top_k=1, valence=0.5, arousal=0.2)
    finally:
        arm.close()
    assert seen["tau"] == 0.8


def test_shared_history_defaults_do_not_replace_published_results():
    out_json, out_md, out_protocol = shared_history_output_paths(
        DEFAULT_OUT_JSON, DEFAULT_OUT_MD, DEFAULT_PROTOCOL
    )
    assert out_json.name == "results.shared_history.json"
    assert out_md.name == "results.shared_history.md"
    assert out_protocol.name == "results.shared_history.protocol.json"
    explicit = Path("explicit.json")
    assert shared_history_output_paths(explicit, DEFAULT_OUT_MD, DEFAULT_PROTOCOL)[0] == explicit


def _ranking_payload(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Comparable ranking fields; ignores receipt metadata that may grow."""
    payload: list[dict[str, Any]] = []
    for system in result["systems"]:
        queries = [
            {
                "query_id": query["query_id"],
                "retrieved_memory_ids": query["retrieved_memory_ids"],
                "top1_hit": query["top1_hit"],
                "hit": query["hit"],
            }
            for scenario in system["scenarios"]
            for session in scenario["sessions"]
            for query in session["queries"]
        ]
        payload.append(
            {
                "system": system["system"],
                "metrics": {
                    key: system["aggregate_metrics"][key]
                    for key in ("top1_accuracy", "hit_at_k", "query_count")
                },
                "queries": queries,
            }
        )
    return payload


def test_token_hash_embedder_empty_salt_matches_unsalted() -> None:
    texts = ["hello world", "same text", "Project Alpha shipped"]
    unsalted = TokenHashEmbedder()
    empty_salt = TokenHashEmbedder(salt="")
    for text in texts:
        assert unsalted.embed(text) == empty_salt.embed(text)


def test_token_hash_embedder_salt_changes_embedding_deterministically() -> None:
    text = "hello world"
    baseline = TokenHashEmbedder().embed(text)
    salted = TokenHashEmbedder(salt="alpha")
    other = TokenHashEmbedder(salt="beta")
    assert salted.embed(text) == salted.embed(text)
    assert salted.embed(text) != baseline
    assert salted.embed(text) != other.embed(text)


def test_resolve_hash_salts_default_and_k() -> None:
    assert resolve_hash_salts() == [""]
    assert resolve_hash_salts(hash_salt=["alpha", "beta"]) == ["alpha", "beta"]
    assert resolve_hash_salts(hash_salts=3) == ["0", "1", "2"]
    with pytest.raises(ValueError, match="not both"):
        resolve_hash_salts(hash_salt=["a"], hash_salts=2)
    with pytest.raises(ValueError, match=">= 1"):
        resolve_hash_salts(hash_salts=0)


def test_shared_history_default_path_matches_explicit_hash() -> None:
    dataset = small_dataset()
    implicit = run_shared_history(dataset, seed=7, n_bootstrap=20)
    explicit = run_shared_history(
        dataset,
        seed=7,
        n_bootstrap=20,
        embedder=TokenHashEmbedder(),
        embedder_name=DEFAULT_SHARED_HISTORY_EMBEDDER,
        hash_salts=[""],
    )
    assert _ranking_payload(implicit) == _ranking_payload(explicit)
    assert implicit["protocol"]["read_only_retrieves"] is False
    assert implicit["protocol"]["hash_salts"] == [""]
    assert implicit["protocol"]["embedder"]["name"] == "hash"
    assert implicit["protocol"]["embedder"]["class"] == "TokenHashEmbedder"
    assert implicit["protocol"]["embedder"]["version"] == HASH_EMBEDDER_VERSION
    assert implicit["manifest"]["embedder"]["name"] == "hash"
    assert implicit["manifest"]["embedder"]["hash_salts"] == [""]
    assert "salt_runs" not in implicit


def test_shared_history_hash_salt_is_recorded_and_changes_ranking_inputs() -> None:
    dataset = small_dataset()
    plain = run_shared_history(dataset, seed=7, n_bootstrap=8)
    salted = run_shared_history(dataset, seed=7, n_bootstrap=8, hash_salts=["alpha"])
    assert salted["protocol"]["hash_salts"] == ["alpha"]
    assert salted["protocol"]["embedder"]["hash_salts"] == ["alpha"]
    assert salted["manifest"]["embedder"]["hash_salts"] == ["alpha"]
    assert salted["protocol"]["read_only_retrieves"] is False
    assert TokenHashEmbedder().embed("hello world") != TokenHashEmbedder(salt="alpha").embed(
        "hello world"
    )
    # Different embeddings change stored vectors; ranking payload may or may not
    # move on this tiny fixture, but traces must differ.
    assert (
        plain["systems"][0]["scenarios"][0]["trace"]
        != salted["systems"][0]["scenarios"][0]["trace"]
    )


def test_shared_history_multiple_hash_salts_emit_salt_runs() -> None:
    dataset = small_dataset()
    result = run_shared_history(dataset, seed=7, n_bootstrap=8, hash_salts=["0", "1"])
    assert result["protocol"]["hash_salts"] == ["0", "1"]
    assert "salt_runs" in result
    assert len(result["salt_runs"]) == 2
    assert result["salt_runs"][0]["protocol"]["hash_salts"] == ["0"]
    assert result["salt_runs"][1]["protocol"]["hash_salts"] == ["1"]
    assert (
        result["salt_runs"][0]["systems"][0]["scenarios"][0]["trace"]
        != result["salt_runs"][1]["systems"][0]["scenarios"][0]["trace"]
    )


def test_shared_history_read_only_retrieves_always_written() -> None:
    dataset = small_dataset()
    writable = run_shared_history(dataset, seed=7, n_bootstrap=8)
    readonly = run_shared_history(dataset, seed=7, n_bootstrap=8, read_only_retrieves=True)
    assert writable["protocol"]["read_only_retrieves"] is False
    assert readonly["protocol"]["read_only_retrieves"] is True


def test_semantic_embedder_missing_extra_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from unittest.mock import patch

    with (
        patch.dict(sys.modules, {"sentence_transformers": None}),
        pytest.raises(ImportError, match="sentence-transformers"),
    ):
        build_shared_history_embedder("sbert-mini")


@pytest.mark.sentence_transformers
def test_shared_history_semantic_embedder_records_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sentence_transformers")
    from unittest.mock import MagicMock

    import numpy as np

    fake_model = MagicMock()
    fake_model.encode.side_effect = lambda input, **_: (
        np.random.default_rng(0).random((len(input), 384)).astype("float32")
        if isinstance(input, list)
        else np.random.default_rng(0).random(384).astype("float32")
    )
    mock_cls = MagicMock(return_value=fake_model)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", mock_cls)

    embedder = build_shared_history_embedder(
        "sentence-transformers", embedder_model=DEFAULT_SEMANTIC_MODEL
    )
    mock_cls.assert_called_once_with(DEFAULT_SEMANTIC_MODEL, device="cpu")
    dataset = small_dataset()
    result = run_shared_history(
        dataset,
        seed=7,
        n_bootstrap=8,
        embedder=embedder,
        embedder_name="sentence-transformers",
        embedder_model=DEFAULT_SEMANTIC_MODEL,
    )
    info = result["protocol"]["embedder"]
    assert info["name"] == "sentence-transformers"
    assert info["class"] == "SentenceTransformerEmbedder"
    assert info["model"] == DEFAULT_SEMANTIC_MODEL
    assert info["device"] == "cpu"
    assert info["version"]
    assert result["protocol"]["read_only_retrieves"] is False
    assert (
        describe_embedder(
            embedder,
            embedder_name="sentence-transformers",
            embedder_model=DEFAULT_SEMANTIC_MODEL,
            hash_salts=[""],
        )["device"]
        == "cpu"
    )
