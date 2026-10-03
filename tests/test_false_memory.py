"""Invariant tests for the wrong-person decoy study."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from benchmarks.esmemeval.dataset import EsmemQuery, EsmemSession, load_dataset
from benchmarks.false_memory.appraisal_cache import CheckpointedAppraiser
from benchmarks.false_memory.runner import ablation_weights, make_case, select_queries, transplant
from emotional_memory.appraisal import GenericAppraisalVector
from emotional_memory.appraisal_schema import DIRECT_VAD_SCHEMA

if TYPE_CHECKING:
    from emotional_memory.appraisal_llm import LLMAppraisalEngine


def _session(key: str, seeker: str, name: str) -> EsmemSession:
    return EsmemSession(
        key=key,
        seeker_id=seeker,
        session_id=key.split("/")[-1],
        timestamp="2025-01-01",
        emotion="anxiety",
        text=f"{name}: I lost the job.\nsupporter: That sounds difficult.\n{name}: I felt scared.",
    )


def test_balanced_sample_is_deterministic() -> None:
    dataset = load_dataset()
    first = select_queries(dataset)
    second = select_queries(dataset)
    assert [q.query_id for q in first] == [q.query_id for q in second]
    assert len(first) == 200
    assert sorted({q.capability for q in first}) == [
        "conflict detection",
        "information extraction",
        "temporal reasoning",
        "user modeling",
    ]
    assert all(
        sum(q.capability == cap for q in first) == 50 for cap in {q.capability for q in first}
    )


def test_transplant_only_rewrites_seeker_prefix() -> None:
    donor = _session("other/1", "other", "Bob")
    target = _session("target/1", "target", "Alice")
    assert transplant(donor, target) == (
        "Alice: I lost the job.\nsupporter: That sounds difficult.\nAlice: I felt scared."
    )


def test_case_keeps_gold_and_chooses_semantic_donor_outside_pool() -> None:
    target = _session("target/1", "target", "Alice")
    in_pool = _session("other/1", "other", "Bob")
    close = _session("other/2", "other", "Carol")
    far = _session("other/3", "other", "Dana")
    sessions = {s.key: s for s in (target, in_pool, close, far)}
    query = EsmemQuery(7, "target", "g", "information extraction", "job", frozenset({target.key}))
    vectors = {
        target.key: [0.9, 0.1],
        in_pool.key: [0.0, 1.0],
        close.key: [1.0, 0.0],
        far.key: [0.0, 1.0],
    }
    case = make_case(query, sessions, (target.key, in_pool.key), [1.0, 0.0], vectors)
    assert case.donor.key == close.key
    assert case.replaced_key == in_pool.key
    assert case.clean_pool == (target.key, in_pool.key)
    assert case.altered_pool == (target.key, "decoy:7")
    assert target.key in case.altered_pool
    assert "Alice: I lost the job." in case.decoy_text


def test_proximity_ablation_zeroes_only_its_weight() -> None:
    weights = ablation_weights()
    assert len(weights) == 6
    assert weights[2] == 0.0
    assert all(weight > 0 for i, weight in enumerate(weights) if i != 2)
    assert abs(sum(weights) - 1.0) < 1e-12


def test_appraisal_checkpoint_reuses_successful_calls(tmp_path: Path) -> None:
    class Stub:
        calls = 0

        def appraise(
            self, text: str, context: dict[str, Any] | None = None
        ) -> GenericAppraisalVector:
            self.calls += 1
            return GenericAppraisalVector(
                {"valence": -0.4, "arousal": 0.7, "dominance": 0.3}, DIRECT_VAD_SCHEMA
            )

        def close(self) -> None:
            pass

    stub = Stub()
    checkpoint = tmp_path / "appraisals.jsonl"
    first = CheckpointedAppraiser(cast("LLMAppraisalEngine", stub), model="test", path=checkpoint)
    assert first.appraise("episode", {"key": "one"}).to_core_affect().valence == -0.4
    assert stub.calls == 1
    second = CheckpointedAppraiser(cast("LLMAppraisalEngine", stub), model="test", path=checkpoint)
    assert second.appraise("episode", {"key": "one"}).to_core_affect().valence == -0.4
    assert stub.calls == 1
    second.appraise("episode", {"key": "two"})
    assert stub.calls == 2
    other_model = CheckpointedAppraiser(
        cast("LLMAppraisalEngine", stub), model="other", path=checkpoint
    )
    other_model.appraise("episode", {"key": "one"})
    assert stub.calls == 3
