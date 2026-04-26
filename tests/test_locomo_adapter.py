"""Smoke tests for the LoCoMo benchmark adapter — no real LLM, no download."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from benchmarks.locomo.dataset import (
    LoCoMoDataset,
    _parse_conversation,
    load_dataset,
)
from benchmarks.locomo.scoring import (
    bleu1,
    build_judge_prompt,
    is_adversarial_correct,
    parse_judge_response,
    score_predictions,
    token_f1,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SYNTHETIC_RAW: dict[str, Any] = {
    "sample_id": "test_conv_0",
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1_date_time": "2023-01-01",
        "session_1": [
            {"dia_id": "d1", "speaker": "Alice", "text": "I love hiking in the mountains."},
            {"dia_id": "d2", "speaker": "Bob", "text": "I prefer the beach."},
        ],
        "session_2_date_time": "2023-02-01",
        "session_2": [
            {"dia_id": "d3", "speaker": "Alice", "text": "I went hiking last weekend."},
        ],
    },
    "qa": [
        {
            "question": "What does Alice love to do?",
            "answer": "hiking",
            "category": 4,
            "evidence": ["d1"],
        },
        {
            "question": "What does Bob prefer?",
            "answer": "the beach",
            "category": 1,
            "evidence": ["d2"],
        },
        {
            "question": "What does Alice prefer — hiking or skiing?",
            "answer": "Not mentioned in the conversation",
            "category": 5,
            "evidence": [],
        },
    ],
}


def _make_synthetic_dataset() -> LoCoMoDataset:
    return LoCoMoDataset(conversations=[_parse_conversation(_SYNTHETIC_RAW)])


def _stub_embedder() -> MagicMock:
    """Mock embedder returning a fixed 4-dim unit vector."""
    emb = MagicMock()
    emb.embed.return_value = [1.0, 0.0, 0.0, 0.0]
    emb.embed_batch.return_value = [[1.0, 0.0, 0.0, 0.0]]
    return emb


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_parse_synthetic_conversation() -> None:
    conv = _parse_conversation(_SYNTHETIC_RAW)

    assert conv.sample_id == "test_conv_0"
    assert conv.speaker_a == "Alice"
    assert len(conv.sessions) == 2
    assert conv.sessions[0].session_num == 1
    assert len(conv.sessions[0].turns) == 2
    assert conv.sessions[0].turns[0].speaker == "Alice"
    assert len(conv.qa_pairs) == 3
    assert conv.qa_pairs[0].category == 4
    assert conv.qa_pairs[2].is_adversarial is True
    assert conv.qa_pairs[0].is_adversarial is False


def test_load_dataset_from_local_file(tmp_path: Path) -> None:
    f = tmp_path / "locomo10.json"
    f.write_text(json.dumps([_SYNTHETIC_RAW]), encoding="utf-8")

    dataset = load_dataset(f)

    assert len(dataset.conversations) == 1
    assert dataset.total_qa == 3


def test_load_dataset_limit_conversations(tmp_path: Path) -> None:
    f = tmp_path / "locomo10.json"
    f.write_text(json.dumps([_SYNTHETIC_RAW, _SYNTHETIC_RAW]), encoding="utf-8")

    dataset = load_dataset(f, limit_conversations=1)

    assert len(dataset.conversations) == 1


def test_load_dataset_limit_qa(tmp_path: Path) -> None:
    f = tmp_path / "locomo10.json"
    f.write_text(json.dumps([_SYNTHETIC_RAW]), encoding="utf-8")

    dataset = load_dataset(f, limit_qa_per_conversation=1)

    assert len(dataset.conversations[0].qa_pairs) == 1
    assert dataset.conversations[0].qa_pairs[0].question == "What does Alice love to do?"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_token_f1_exact_match() -> None:
    assert token_f1("hiking", "hiking") == pytest.approx(1.0)


def test_token_f1_partial_overlap() -> None:
    f1 = token_f1("hiking in mountains", "hiking")
    assert 0.0 < f1 < 1.0


def test_token_f1_no_overlap() -> None:
    assert token_f1("beach", "hiking") == pytest.approx(0.0)


def test_bleu1_exact_match() -> None:
    assert bleu1("hiking", "hiking") == pytest.approx(1.0)


def test_bleu1_zero_on_mismatch() -> None:
    assert bleu1("beach", "hiking") == pytest.approx(0.0)


def test_build_judge_prompt_contains_fields() -> None:
    prompt = build_judge_prompt("What is X?", "gold answer", "generated answer")

    assert "What is X?" in prompt
    assert "gold answer" in prompt
    assert "generated answer" in prompt


def test_parse_judge_response_json_correct() -> None:
    assert parse_judge_response('{"label": "CORRECT"}') is True


def test_parse_judge_response_json_wrong() -> None:
    assert parse_judge_response('{"label": "WRONG"}') is False


def test_parse_judge_response_fallback_correct() -> None:
    assert parse_judge_response("The answer is CORRECT.") is True


def test_parse_judge_response_fallback_wrong() -> None:
    assert parse_judge_response("This is WRONG.") is False


def test_parse_judge_response_invalid_json_falls_back() -> None:
    assert parse_judge_response("not json at all, CORRECT") is True


def test_is_adversarial_correct_refusal() -> None:
    assert is_adversarial_correct("Not mentioned in the conversation.") is True
    assert is_adversarial_correct("no information available") is True
    assert is_adversarial_correct("Alice loves hiking.") is False


def test_score_predictions_aggregate() -> None:
    preds = [
        {
            "question": "q1",
            "gold": "hiking",
            "prediction": "hiking",
            "category": 4,
            "category_name": "single_hop",
            "is_adversarial": False,
            "judge_correct": True,
        },
        {
            "question": "q2",
            "gold": "beach",
            "prediction": "hiking",
            "category": 1,
            "category_name": "multi_hop",
            "is_adversarial": False,
            "judge_correct": False,
        },
    ]
    scores = score_predictions(preds)

    assert scores["aggregate"]["n"] == 2
    assert 0.0 < scores["aggregate"]["f1"] <= 1.0
    assert scores["aggregate"]["judge_accuracy"] == pytest.approx(0.5)
    assert "single_hop" in scores["by_category"]
    assert "multi_hop" in scores["by_category"]


def test_score_predictions_excludes_adversarial_by_default() -> None:
    preds = [
        {
            "question": "q1",
            "gold": "hiking",
            "prediction": "hiking",
            "category": 4,
            "is_adversarial": False,
        },
        {
            "question": "q2",
            "gold": "Not mentioned",
            "prediction": "Not mentioned in the conversation",
            "category": 5,
            "is_adversarial": True,
        },
    ]
    scores = score_predictions(preds)

    assert scores["aggregate"]["n"] == 1
    assert "adversarial" not in scores["by_category"]


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def test_naive_rag_adapter_runs() -> None:
    from benchmarks.locomo.adapters.naive_rag import NaiveRAGLoCoMoAdapter

    dataset = _make_synthetic_dataset()
    conv = dataset.conversations[0]

    with (
        patch(
            "benchmarks.locomo.adapters.naive_rag.SentenceTransformerEmbedder.make_bge_small",
            return_value=_stub_embedder(),
        ),
        patch("benchmarks.locomo.adapters.naive_rag.call_llm", return_value="hiking"),
    ):
        adapter = NaiveRAGLoCoMoAdapter(top_k=2)
        preds = adapter.run_conversation(conv)

    assert len(preds) == 3
    assert all("prediction" in p and "gold" in p and "category" in p for p in preds)
    assert all(p["sample_id"] == "test_conv_0" for p in preds)


def test_aft_adapter_runs() -> None:
    from benchmarks.locomo.adapters.aft import AFTLoCoMoAdapter

    dataset = _make_synthetic_dataset()
    conv = dataset.conversations[0]

    with (
        patch(
            "benchmarks.locomo.adapters.aft.SentenceTransformerEmbedder.make_bge_small",
            return_value=_stub_embedder(),
        ),
        patch("benchmarks.locomo.adapters.aft.call_llm", return_value="hiking"),
    ):
        adapter = AFTLoCoMoAdapter(top_k=2)
        preds = adapter.run_conversation(conv)

    assert len(preds) == 3
    assert all("prediction" in p and "gold" in p for p in preds)
    assert all(p["sample_id"] == "test_conv_0" for p in preds)


# ---------------------------------------------------------------------------
# call_llm retry behaviour
# ---------------------------------------------------------------------------


def _ok_response() -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.raise_for_status.return_value = None
    r.json.return_value = {"choices": [{"message": {"content": "hiking"}}]}
    return r


def _err_response(status: int) -> MagicMock:
    import httpx

    r = MagicMock()
    r.status_code = status
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    r.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"{status} error", request=req, response=r
    )
    return r


def test_call_llm_retries_on_transient_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """call_llm should retry on 520 and return the response when a later attempt succeeds."""
    from benchmarks.locomo.adapters.base import call_llm

    responses = [_err_response(520), _err_response(520), _ok_response()]
    call_count = 0

    def fake_post(*_args: object, **_kwargs: object) -> MagicMock:
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return resp

    monkeypatch.setenv("EMOTIONAL_MEMORY_LLM_API_KEY", "test-key")
    with (
        patch("httpx.post", side_effect=fake_post),
        patch("benchmarks.locomo.adapters.base.time.sleep"),
    ):
        result = call_llm("hello")

    assert result == "hiking"
    assert call_count == 3  # 1 initial + 2 retries


def test_call_llm_raises_after_all_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """call_llm should raise HTTPStatusError once _MAX_RETRIES are exhausted."""
    import httpx

    from benchmarks.locomo.adapters import base as base_mod
    from benchmarks.locomo.adapters.base import _MAX_RETRIES, call_llm

    call_count = 0

    def fake_post(*_args: object, **_kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        return _err_response(520)

    monkeypatch.setenv("EMOTIONAL_MEMORY_LLM_API_KEY", "test-key")
    with (
        patch("httpx.post", side_effect=fake_post),
        patch.object(base_mod, "time", wraps=base_mod.time) as mock_time,
    ):
        mock_time.sleep = MagicMock()
        with pytest.raises(httpx.HTTPStatusError):
            call_llm("hello")

    # 1 initial + _MAX_RETRIES retries
    assert call_count == 1 + _MAX_RETRIES


def test_run_benchmark_end_to_end(tmp_path: Path) -> None:
    from benchmarks.locomo.runner import run_benchmark, write_results

    dataset = _make_synthetic_dataset()

    with (
        patch(
            "benchmarks.locomo.adapters.aft.SentenceTransformerEmbedder.make_bge_small",
            return_value=_stub_embedder(),
        ),
        patch(
            "benchmarks.locomo.adapters.naive_rag.SentenceTransformerEmbedder.make_bge_small",
            return_value=_stub_embedder(),
        ),
        patch("benchmarks.locomo.adapters.aft.call_llm", return_value="hiking"),
        patch("benchmarks.locomo.adapters.naive_rag.call_llm", return_value="hiking"),
    ):
        results = run_benchmark(dataset, systems=["aft", "naive_rag"], run_judge=False)

    assert results["n_conversations"] == 1
    assert results["n_qa_total"] == 3
    assert [s["system"] for s in results["systems"]] == ["aft", "naive_rag"]
    for sys in results["systems"]:
        assert "scores" in sys
        assert sys["scores"]["aggregate"]["n"] >= 1

    out_json = tmp_path / "results.json"
    out_md = tmp_path / "results.md"
    write_results(results, out_json=out_json, out_md=out_md)

    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "locomo_v1"
    md = out_md.read_text(encoding="utf-8")
    assert "# LoCoMo Benchmark Results" in md
    assert "## Aggregate Scores" in md
