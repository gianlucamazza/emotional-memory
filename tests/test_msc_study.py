"""Scientific invariants for the external study, using explicitly synthetic fixtures."""

from __future__ import annotations

import copy
import itertools
import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.msc.dataset import (
    Candidate,
    Case,
    build_cases,
    digest,
    load_cases,
    prepare,
    write_json,
)
from benchmarks.msc.evaluation import (
    collect_gold,
    evaluate,
    packets,
    response_packets,
    summarize_responses,
)
from benchmarks.msc.runner import CachedEmbedder, mem0_rank, rank_features, run_case, tune
from benchmarks.msc.statistics import (
    binomial_pmf,
    discordance_upper,
    mcnemar,
    paired_report,
    plan_n,
)
from emotional_memory.appraisal import AppraisalVector


def case(index: int = 0) -> Case:
    return Case(
        case_id=f"case-{index}",
        group_id=f"group-{index}",
        query="How did it go?",
        context="Speaker 1: I had an interview.",
        query_session=1,
        candidates=(
            Candidate(memory_id=f"a-{index}", text="I got the job!", session=0, order=0),
            Candidate(memory_id=f"b-{index}", text="I bought apples.", session=0, order=1),
        ),
    )


def annotated(cases: list[Case]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    public, key = packets(cases, ["g1", "g2", "g3"])
    submissions = list(public.values())
    for submission in submissions:
        for item in submission["items"]:
            item["relevant"] = int(item["memory"] == "I got the job!")
            item["affect_required"] = 0
    return key, submissions, collect_gold(cases, key, submissions)


def test_grouping_removes_shared_histories_and_future_labels() -> None:
    previous = {"dialog": [{"text": "old question"}, {"text": "old reply"}]}
    records = [
        {
            "metadata": {"initial_data_id": key},
            "personas": ["SECRET PERSONA"],
            "previous_dialogs": [previous],
            "dialog": [{"text": "human query"}, {"text": "FUTURE ANSWER"}],
        }
        for key in ("root-a", "root-b")
    ]
    result = build_cases(records)
    assert len(result) == 1
    assert result == build_cases(list(reversed(records)))
    serialized = json.dumps(result[0].model_dump())
    assert "SECRET PERSONA" not in serialized
    assert "FUTURE ANSWER" not in serialized
    assert "human query" in result[0].query


def test_transitive_history_grouping() -> None:
    def dialogue(text: str) -> list[dict[str, str]]:
        return [{"text": text}, {"text": text + " reply"}]

    a = {
        "metadata": {"initial_data_id": "a"},
        "previous_dialogs": [{"dialog": dialogue("start")}],
        "dialog": dialogue("middle"),
    }
    b = {
        "metadata": {"initial_data_id": "b"},
        "previous_dialogs": [{"dialog": dialogue("middle")}],
        "dialog": dialogue("end"),
    }
    assert len(build_cases([a, b])) == 1


def test_case_schema_rejects_oracle_and_future() -> None:
    raw = case().model_dump()
    raw["valence"] = 1
    with pytest.raises(ValidationError):
        Case.model_validate(raw)
    raw.pop("valence")
    raw["candidates"][0]["session"] = 1
    with pytest.raises(ValidationError, match="leakage"):
        Case.model_validate(raw)


def test_loading_duplicate_histories_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    a, b = case(0).model_dump(mode="json"), case(1).model_dump(mode="json")
    b["group_id"] = a["group_id"]
    write_json(path, [a, b])
    with pytest.raises(ValueError, match="independent"):
        load_cases(path)
    with pytest.raises(FileExistsError):
        write_json(path, [])


def test_wrong_source_checksum_is_not_accepted(tmp_path: Path) -> None:
    archive = tmp_path / "wrong.tar.gz"
    archive.write_bytes(b"not the source corpus")
    with pytest.raises(ValueError, match="checksum"):
        prepare(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_packets_hide_system_and_source_ids() -> None:
    public, _ = packets([case()], ["r1", "r2", "r3"])
    for payload in public.values():
        card = payload["items"][0]
        assert not {"rank", "system", "memory_id", "group_id", "valence"} & card.keys()
        assert card["relevant"] is None
    assert len({p["items"][0]["item_id"] for p in public.values()}) == 3


@pytest.mark.parametrize(
    "defect", ["empty", "duplicate", "wrong_rater", "wrong_source", "missing"]
)
def test_gold_rejects_bad_submissions(defect: str) -> None:
    key, submissions, _ = annotated([case()])
    if defect == "empty":
        submissions[0]["items"][0]["relevant"] = None
    elif defect == "duplicate":
        submissions[0]["items"].append(submissions[0]["items"][0])
    elif defect == "wrong_rater":
        submissions[0]["items"], submissions[1]["items"] = (
            submissions[1]["items"],
            submissions[0]["items"],
        )
    elif defect == "wrong_source":
        submissions[0]["source"] = "LLM"
    else:
        submissions[0]["items"].pop()
    with pytest.raises(ValueError):
        collect_gold([case()], key, submissions)


def test_majority_gold_and_zero_relevance_exclusion() -> None:
    key, submissions, gold = annotated([case()])
    assert gold["cases"][0]["relevant_ids"] == ["a-0"]
    for submission in submissions:
        for item in submission["items"]:
            item["relevant"] = 0
    gold = collect_gold([case()], key, submissions)
    assert gold["cases"][0]["eligible"] is False
    assert gold["alpha_nominal"] is None


def test_hybrid_tuning_ties_prefer_semantics() -> None:
    _, _, gold = annotated([case()])
    features = {"case-0": {"a-0": [1, 1, 1], "b-0": [0, 0, 0]}}
    assert tune(features, gold) == [1, 0, 0]
    assert rank_features({"b": [1, 1, 1], "a": [1, 1, 1]}, [1, 0, 0]) == ["a", "b"]
    with pytest.raises(ValueError):
        rank_features(features["case-0"], [1, -1, 1])


def test_paired_analysis_and_foreign_memory_rejection() -> None:
    cases = [case(i) for i in range(8)]
    _, _, gold = annotated(cases)
    results = {
        "stage": "pilot",
        "cases_digest": digest([c.model_dump() for c in cases]),
        "rankings": {
            c.case_id: {
                "aft": [c.candidates[0].memory_id],
                "hybrid": [c.candidates[1].memory_id],
                "cosine": [],
                "mem0": [],
            }
            for c in cases
        },
    }
    report = evaluate(cases, gold, results)
    assert report["primary"]["a_only"] == 8
    assert report["primary"]["superiority"]
    results["rankings"]["case-0"]["mem0"] = ["from-another-user"]
    with pytest.raises(ValueError, match="foreign"):
        evaluate(cases, gold, results)
    results["rankings"]["case-0"].pop("mem0")
    with pytest.raises(ValueError, match="Missing system"):
        evaluate(cases, gold, results)


def test_mcnemar_matches_independent_enumeration() -> None:
    for n in range(1, 10):
        outcomes = list(itertools.product([0, 1], repeat=n))
        for wins in range(n + 1):
            tail = sum(sum(row) <= min(wins, n - wins) for row in outcomes) / len(outcomes)
            assert mcnemar(wins, n - wins) == pytest.approx(min(1, 2 * tail))
    assert mcnemar(1000, 1000) == 1
    assert 0 < mcnemar(1100, 900) < 0.001
    assert mcnemar(0, 0) == 1


def test_binomial_and_upper_limit_against_closed_forms() -> None:
    assert binomial_pmf(2, 0.3).tolist() == pytest.approx([0.49, 0.42, 0.09])
    assert sum(binomial_pmf(4000, 0.9)) == pytest.approx(1)
    assert discordance_upper(0, 100) == pytest.approx(1 - 0.05**0.01)
    assert discordance_upper(100, 100) == 1
    assert math.isfinite(discordance_upper(50, 100))


def test_power_infeasible_and_boundary() -> None:
    assert plan_n(10, 100, maximum=20)["status"] == "INFEASIBLE"
    result = plan_n(0, 100, maximum=250)
    assert result["status"] == "POWERED"
    n = int(result["n"])
    # With q=.05 all discordances favor A; exact rejection requires >=6.
    exact_power = 1 - sum(math.comb(n, i) * 0.05**i * 0.95 ** (n - i) for i in range(6))
    assert result["power"] == pytest.approx(exact_power)
    assert result["power"] >= 0.8
    assert plan_n(0, 100, maximum=n - 1)["status"] == "INFEASIBLE"


def test_no_significance_is_not_equivalence() -> None:
    report = paired_report([True, False], [False, True])
    assert report["superiority"] is False
    assert report["five_point_interpretation"] == "compatible_not_established"


class FakeMem0:
    """Explicitly synthetic in-memory contract double, never used for scored runs."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, messages: Any, *, user_id: str, metadata: Any, infer: bool) -> dict[str, Any]:
        assert infer is False
        row = {"id": str(len(self.rows)), "metadata": metadata, "user": user_id}
        self.rows.append(row)
        return {"results": [row]}

    def search(
        self, query: str, *, filters: dict[str, str], top_k: int, threshold: float, rerank: bool
    ) -> dict[str, Any]:
        assert all(r["user"] == filters["user_id"] for r in self.rows)
        assert threshold == 0.0 and rerank is False
        return {"results": self.rows[:top_k]}


def test_mem0_preserves_source_identity() -> None:
    assert mem0_rank(FakeMem0(), case()) == ["a-0", "b-0"]


def test_mem0_cannot_turn_unknown_ids_into_misses() -> None:
    class ForeignMem0(FakeMem0):
        def search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"results": [{"id": "foreign"}]}

    with pytest.raises(ValueError, match="outside"):
        mem0_rank(ForeignMem0(), case())


def test_run_case_exercises_real_aft_without_gold() -> None:
    class Embedder:
        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0] if "apples" not in text else [0.0, 1.0]

    class Appraiser:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def appraise(self, text: str) -> AppraisalVector:
            self.seen.append(text)
            return AppraisalVector.neutral()

    appraiser = Appraiser()
    result = run_case(case(), CachedEmbedder(Embedder()), appraiser, FakeMem0(), [1, 0, 0])
    assert set(result["rankings"]) == {"aft", "hybrid", "cosine", "mem0"}
    assert result["rankings"]["cosine"][0] == "a-0"
    assert appraiser.seen == [c.text for c in case().candidates] + [case().retrieval_text]
    assert all(set(ids) <= {"a-0", "b-0"} for ids in result["rankings"].values())


def test_appraisal_failure_aborts_run() -> None:
    class BrokenAppraiser:
        def appraise(self, text: str) -> Any:
            raise RuntimeError("synthetic transport failure")

    with pytest.raises(RuntimeError, match="synthetic"):
        run_case(case(), CachedEmbedder(None), BrokenAppraiser(), FakeMem0(), [1, 0, 0])


def test_response_evaluation_keeps_raters_out_of_sample_size() -> None:
    cases = [case(i) for i in range(100)]
    responses = {
        c.case_id: dict.fromkeys(["aft", "hybrid", "cosine", "mem0"], "An answer") for c in cases
    }
    public, key = response_packets(cases, responses, ["e1", "e2", "e3"], ["g1", "g2", "g3"])
    submissions = list(public.values())
    for submission in submissions:
        for item in submission["items"]:
            assert "arm" not in item and "system" not in item
            item["usefulness"] = 4
            item["correctness"] = 3
    summary = summarize_responses(key, submissions)
    assert summary["n_cases"] == 100
    assert summary["usefulness"]["delta_aft_hybrid"] == 0
    assert summary["status"] == "SECONDARY_EXPLORATORY"
    with pytest.raises(ValueError, match="distinct"):
        response_packets(cases, responses, ["g1", "e2", "e3"], ["g1", "g2", "g3"])
    bad = copy.deepcopy(submissions)
    bad[0]["items"].pop()
    with pytest.raises(ValueError, match="Missing"):
        summarize_responses(key, bad)


def test_generation_uses_identical_prompt_budget_and_records_calls() -> None:
    from benchmarks.msc.runner import generate_responses

    prompts: list[str] = []
    events: list[dict[str, Any]] = []

    def llm(prompt: str, schema: dict[str, Any]) -> str:
        prompts.append(prompt)
        return '{"answer":"Congratulations."}'

    rankings = {
        "rankings": {"case-0": {arm: ["a-0"] for arm in ("aft", "hybrid", "cosine", "mem0")}}
    }
    output = generate_responses([case()], rankings, llm, events.append)
    assert len(prompts) == 4 and len(set(prompts)) == 1
    assert len(events) == 8
    assert output["case-0"]["aft"] == "Congratulations."


def test_freeze_binds_power_config_and_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from benchmarks.msc import cli
    from benchmarks.msc.dataset import file_sha

    # Explicitly synthetic power/annotation fixtures; no live registration is created.
    cases = [case(i) for i in range(200)]
    _, _, gold = annotated(cases)
    config = {"hybrid_weights": [1, 0, 0], "tuning_provenance": {"synthetic": True}}
    write_json(tmp_path / "reserve.json", [c.model_dump(mode="json") for c in cases])
    write_json(tmp_path / "gold.json", gold)
    write_json(tmp_path / "config.json", config)
    manifest = {"splits": {"reserve": {"sha256": file_sha(tmp_path / "reserve.json")}}}
    write_json(tmp_path / "manifest.json", manifest)
    power = plan_n(0, 100, maximum=200)
    power.update(
        {
            "config_digest": digest(config),
            "manifest_sha256": file_sha(tmp_path / "manifest.json"),
            "pilot_alpha_nominal": 1.0,
        }
    )
    write_json(tmp_path / "power.json", power)
    readiness = dict.fromkeys(
        ["license_evidence", "recruitment_record", "resource_estimate", "model_snapshot_evidence"],
        "SYNTHETIC TEST ONLY",
    )
    readiness.update({"gold_raters": ["g1", "g2", "g3"], "response_raters": ["e1", "e2", "e3"]})
    write_json(tmp_path / "readiness.json", readiness)
    args = argparse.Namespace(
        cases=tmp_path / "reserve.json",
        gold=tmp_path / "gold.json",
        config=tmp_path / "config.json",
        manifest=tmp_path / "manifest.json",
        power=tmp_path / "power.json",
        readiness=tmp_path / "readiness.json",
        out=tmp_path / "frozen",
    )
    monkeypatch.setattr(cli, "validate_config", lambda _: None)
    cli.freeze(args)
    registration = cli.read(args.out / "registration.json")
    assert registration["n"] == power["n"]
    assert len(registration["end_to_end_case_ids"]) == 100
    assert registration["cases_sha256"] == file_sha(args.out / "cases.json")
    # A later config mutation cannot silently inherit the pilot's sample sizing.
    config["hybrid_weights"] = [0, 1, 0]
    args.config.write_text(json.dumps(config))
    args.out = tmp_path / "changed"
    with pytest.raises(ValueError, match="changed since"):
        cli.freeze(args)
    assert not args.out.exists()


def test_registration_checks_committed_bytes_and_live_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from benchmarks.msc import cli
    from benchmarks.msc.dataset import file_sha

    cases_path = tmp_path / "cases.json"
    config_path = tmp_path / "config.json"
    registration_path = tmp_path / "registration.json"
    write_json(cases_path, [case().model_dump(mode="json")])
    write_json(config_path, {"synthetic": True})
    payload = {
        "status": "FROZEN_PENDING_COMMIT",
        "n": 1,
        "code_digest": "code",
        "cases_sha256": file_sha(cases_path),
        "config_sha256": file_sha(config_path),
    }
    write_json(registration_path, payload)
    committed = registration_path.read_bytes()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "code_digest", lambda: "code")
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=committed))
    assert cli.check_registration(registration_path, cases_path, config_path)["n"] == 1
    config_path.write_text('{"synthetic":false}')
    with pytest.raises(ValueError, match="binding"):
        cli.check_registration(registration_path, cases_path, config_path)
    registration_path.write_text(json.dumps({**payload, "n": 2}))
    with pytest.raises(ValueError, match="committed"):
        cli.check_registration(registration_path, cases_path, config_path)


def test_credentials_cannot_be_persisted_in_config() -> None:
    from benchmarks.msc.cli import validate_config

    with pytest.raises(ValueError, match="Credentials"):
        validate_config({"mem0_config": {"api_key": "synthetic-not-a-secret"}})


def test_cli_failed_attempt_preserves_trace_and_blocks_same_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from benchmarks.msc import cli
    from benchmarks.msc.dataset import file_sha

    cases_path = tmp_path / "cases.json"
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.json"
    out = tmp_path / "results.json"
    write_json(cases_path, [case().model_dump(mode="json")])
    write_json(config_path, {"synthetic": True})
    write_json(manifest_path, {"splits": {"pilot": {"sha256": file_sha(cases_path)}}})
    monkeypatch.setattr(cli, "validate_config", lambda _: None)

    def failing_run(*args: Any, **kwargs: Any) -> Any:
        kwargs["trace"]({"event": "synthetic_attempt"})
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(cli, "run", failing_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "msc",
            "run",
            "--stage",
            "pilot",
            "--cases",
            str(cases_path),
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--out",
            str(out),
        ],
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        cli.main()
    assert not out.exists()
    assert out.with_suffix(".started.json").exists()
    assert cli.read(out.with_suffix(".failure.json"))["error_type"] == "RuntimeError"
    assert "synthetic_attempt" in out.with_suffix(".trace.jsonl").read_text()
    with pytest.raises(FileExistsError):
        cli.main()


def test_cli_cannot_run_reserve_as_pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from benchmarks.msc import cli

    cases_path, config_path, manifest_path = (
        tmp_path / f"{name}.json" for name in ("cases", "config", "manifest")
    )
    write_json(cases_path, [case().model_dump(mode="json")])
    write_json(config_path, {"synthetic": True})
    write_json(manifest_path, {"splits": {"pilot": {"sha256": "a-different-split"}}})
    monkeypatch.setattr(cli, "validate_config", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "msc",
            "run",
            "--stage",
            "pilot",
            "--cases",
            str(cases_path),
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--out",
            str(tmp_path / "result.json"),
        ],
    )
    with pytest.raises(ValueError, match="Stage/split"):
        cli.main()
    assert not (tmp_path / "result.started.json").exists()
