"""Preparation, annotation, freezing and analysis CLI; live calls are explicit."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from benchmarks.msc.dataset import SEED, digest, file_sha, load_cases, prepare, write_json
from benchmarks.msc.evaluation import (
    collect_gold,
    evaluate,
    packets,
    response_packets,
    summarize_responses,
)
from benchmarks.msc.runner import generate_responses, run, tune
from benchmarks.msc.statistics import plan_n
from emotional_memory.llm_http import OpenAICompatibleLLMConfig, make_httpx_llm

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("emotional_memory", "numpy", "pydantic", "sentence-transformers", "mem0ai", "httpx")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def code_digest() -> str:
    paths = sorted((ROOT / "src/emotional_memory").rglob("*.py"))
    paths += sorted((ROOT / "benchmarks/msc").glob("*.py"))
    paths += [
        ROOT / "benchmarks/msc/protocol.md",
        ROOT / "uv.lock",
        ROOT / "benchmarks/common/similarity.py",
        ROOT / "benchmarks/human_eval/pipeline.py",
    ]
    return digest({str(p.relative_to(ROOT)): file_sha(p) for p in paths})


def validate_config(config: dict[str, Any]) -> None:
    def reject_secrets(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in (
                    "api_key",
                    "password",
                    "secret",
                    "token",
                    "access_token",
                ) or key.lower().endswith("_api_key"):
                    raise ValueError(
                        "Credentials must come from environment, not persisted config"
                    )
                reject_secrets(child)
        elif isinstance(value, list):
            for child in value:
                reject_secrets(child)

    reject_secrets(config)
    if config["embedder_model"] != "BAAI/bge-small-en-v1.5":
        raise ValueError("Study fixes bge-small-en-v1.5")
    if not re.fullmatch(r"[0-9a-f]{40}", config.get("embedder_revision") or ""):
        raise ValueError("Pin the embedder to a commit revision")
    for name in (
        "appraisal_model",
        "appraisal_base_url",
        "generation_model",
        "generation_base_url",
    ):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f"Missing model/endpoint pin: {name}")
    # This protocol uses a controlled Mem0 memory-unit reference, not fact extraction.
    if config.get("mem0_infer") is not False:
        raise ValueError("Mem0 infer=False is required to preserve annotated units")
    from benchmarks.msc.runner import rank_features

    rank_features({}, config["hybrid_weights"])
    for package in PACKAGES:
        if config["package_versions"].get(package) != importlib.metadata.version(package):
            raise ValueError(f"Package version mismatch: {package}")
    if not config.get("mem0_config"):
        raise ValueError("Explicit Mem0 provider configuration required")


def check_registration(path: Path, cases_path: Path, config_path: Path) -> dict[str, Any]:
    registration = read(path)
    if registration["status"] != "FROZEN_PENDING_COMMIT":
        raise ValueError("Invalid registration status")
    relative = path.resolve().relative_to(ROOT)
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required for registration verification")
    committed = subprocess.run(  # noqa: S603 - fixed read-only argv, no shell
        [git, "show", f"HEAD:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    if committed != path.read_bytes():
        raise ValueError("Registration must be committed before confirmation")
    expected = {
        "cases_sha256": file_sha(cases_path),
        "config_sha256": file_sha(config_path),
        "code_digest": code_digest(),
    }
    if any(registration[k] != v for k, v in expected.items()):
        raise ValueError("Frozen registration binding changed")
    if len(load_cases(cases_path)) != registration["n"]:
        raise ValueError("Sample size differs from frozen N")
    return registration


def freeze(args: argparse.Namespace) -> None:
    config = read(args.config)
    validate_config(config)
    approvals = read(args.readiness)
    required = (
        "license_evidence",
        "recruitment_record",
        "resource_estimate",
        "model_snapshot_evidence",
    )
    if any(not isinstance(approvals.get(k), str) or not approvals[k].strip() for k in required):
        raise ValueError("Source license, recruitment, resource and model evidence required")
    if set(approvals["gold_raters"]) & set(approvals["response_raters"]):
        raise ValueError("End-to-end raters must be independent of gold annotators")
    if any(len(set(approvals[k])) != 3 for k in ("gold_raters", "response_raters")):
        raise ValueError("Need three distinct raters per panel")
    manifest = read(args.manifest)
    if manifest["splits"]["reserve"]["sha256"] != file_sha(args.cases):
        raise ValueError("Reserve differs from source manifest")
    cases = load_cases(args.cases)
    gold = read(args.gold)
    if gold["cases_digest"] != digest([c.model_dump() for c in cases]):
        raise ValueError("Gold not bound to reserve")
    if sorted(gold["raters"]) != sorted(approvals["gold_raters"]):
        raise ValueError("Gold roster differs from recruitment record")
    eligible = {r["case_id"] for r in gold["cases"] if r["eligible"]}
    power = read(args.power)
    if power["status"] != "POWERED":
        raise ValueError("No powered sample size available")
    if not config.get("tuning_provenance"):
        raise ValueError("Development tuning provenance is required")
    if power.get("config_digest") != digest(config):
        raise ValueError("Configuration changed since the sizing pilot")
    if power.get("manifest_sha256") != file_sha(args.manifest):
        raise ValueError("Pilot and reserve must share the same source/split manifest")
    if power.get("pilot_alpha_nominal") is None or power["pilot_alpha_nominal"] < 0.67:
        raise ValueError("Pilot annotation agreement is insufficient")
    if gold.get("source") != "human" or len(gold["cases"]) != len(cases):
        raise ValueError("Complete human reserve gold is required")
    if {r["case_id"] for r in gold["cases"]} != {c.case_id for c in cases}:
        raise ValueError("Reserve gold coverage mismatch")
    recalculated = plan_n(power["pilot_discordant"], power["pilot_n"], maximum=power["maximum"])
    if recalculated["status"] != "POWERED" or recalculated["n"] != power["n"]:
        raise ValueError("Powered N does not reproduce from the pilot counts")
    n = power["n"]
    chosen = [c for c in cases if c.case_id in eligible][:n]
    if len(chosen) != n or n < 100:
        raise ValueError("Insufficient eligible independent histories (including 100 E2E cases)")
    chosen_ids = {c.case_id for c in chosen}
    filtered_gold = dict(gold)
    filtered_gold["cases"] = [r for r in gold["cases"] if r["case_id"] in chosen_ids]
    filtered_gold["cases_digest"] = digest([c.model_dump() for c in chosen])
    filtered_gold["agreement_scope"] = "all independently annotated reserve candidates"
    filtered_gold["reserve_excluded_no_relevant_memory"] = len(cases) - len(eligible)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "cases.json", [c.model_dump(mode="json") for c in chosen])
    write_json(out / "gold.json", filtered_gold)
    write_json(out / "config.json", config)
    end_to_end = sorted(chosen_ids, key=lambda k: digest([SEED, k, "e2e"]))[:100]
    registration = {
        "status": "FROZEN_PENDING_COMMIT",
        "n": n,
        "end_to_end_case_ids": end_to_end,
        "cases_sha256": file_sha(out / "cases.json"),
        "gold_sha256": file_sha(out / "gold.json"),
        "config_sha256": file_sha(out / "config.json"),
        "code_digest": code_digest(),
        "manifest_sha256": file_sha(args.manifest),
        "power_sha256": file_sha(args.power),
        "readiness_sha256": file_sha(args.readiness),
        "power": power,
        "primary": "AFT minus hybrid top-1; exact two-sided McNemar alpha .05",
        "seed": SEED,
        "readiness": approvals,
    }
    write_json(out / "registration.json", registration)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, paths: tuple[str, ...]) -> argparse.ArgumentParser:
        child = sub.add_parser(name)
        for option in paths:
            child.add_argument(f"--{option}", type=Path, required=True)
        return child

    command("prepare", ("archive", "out"))
    p = command("packets", ("cases", "out"))
    p.add_argument("--raters", nargs=3, required=True)
    p = command("gold", ("cases", "key", "out"))
    p.add_argument("--ratings", nargs=3, type=Path, required=True)
    p = command("run", ("cases", "config", "out"))
    p.add_argument("--stage", choices=("development", "pilot", "confirmatory"), required=True)
    p.add_argument("--registration", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    command("tune", ("cases", "gold", "results", "config", "out"))
    p = command("analyze", ("cases", "gold", "results", "out"))
    p.add_argument("--registration", type=Path)
    p = command("power", ("pilot_report", "out"))
    p.add_argument("--maximum", type=int, required=True)
    command("freeze", ("cases", "gold", "config", "manifest", "power", "readiness", "out"))
    command("generate", ("cases", "results", "config", "registration", "out"))
    p = command("response-packets", ("cases", "responses", "registration", "out"))
    p = command("response-summary", ("key", "out"))
    p.add_argument("--ratings", nargs=3, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.archive, args.out), indent=2))
    elif args.command == "packets":
        public, private = packets(load_cases(args.cases), args.raters)
        # Rater strings never become file paths.
        for i, payload in enumerate(public.values()):
            write_json(args.out / f"rater-{i + 1}.json", payload)
        write_json(args.out / "private-key.json", private)
    elif args.command == "gold":
        gold = collect_gold(
            load_cases(args.cases), read(args.key), [read(p) for p in args.ratings]
        )
        gold["provenance"] = {
            "packet_key_sha256": file_sha(args.key),
            "submission_sha256": [file_sha(p) for p in args.ratings],
        }
        write_json(args.out, gold)
    elif args.command == "run":
        config = read(args.config)
        validate_config(config)
        if args.stage == "confirmatory":
            if args.registration is None:
                parser.error("confirmation requires --registration")
            registration = check_registration(args.registration, args.cases, args.config)
            if registration["manifest_sha256"] != file_sha(args.manifest):
                raise ValueError("Frozen source manifest changed")
        elif read(args.manifest)["splits"][args.stage]["sha256"] != file_sha(args.cases):
            raise ValueError("Stage/split mismatch; reserve cannot run as pilot")
        # Exclusive creation prevents unnoticed reruns of an already started trial.
        started = {
            "stage": args.stage,
            "cases_sha256": file_sha(args.cases),
            "config_sha256": file_sha(args.config),
            "code_digest": code_digest(),
        }
        write_json(args.out.with_suffix(".started.json"), started)
        try:
            with args.out.with_suffix(".trace.jsonl").open("x", encoding="utf-8") as trace_file:

                def record(event: dict[str, Any]) -> None:
                    trace_file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                    trace_file.flush()

                result = run(load_cases(args.cases), config, args.stage, trace=record)
        except Exception as exc:
            write_json(
                args.out.with_suffix(".failure.json"),
                {
                    **started,
                    "error_type": type(exc).__name__,
                    "status": "ABORTED_NO_RESULT: retain marker; investigate before any new run",
                },
            )
            raise
        result["manifest_sha256"] = file_sha(args.manifest)
        write_json(args.out, result)
    elif args.command == "tune":
        results = read(args.results)
        gold = read(args.gold)
        if results["stage"] != "development":
            raise ValueError("Tune only on development")
        evaluate(load_cases(args.cases), gold, results)
        config = read(args.config)
        config["hybrid_weights"] = tune(
            {k: v["features"] for k, v in results["traces"].items()}, gold
        )
        config["tuning_provenance"] = {
            "results_sha256": file_sha(args.results),
            "gold_sha256": file_sha(args.gold),
        }
        write_json(args.out, config)
    elif args.command == "analyze":
        results = read(args.results)
        if results["stage"] == "confirmatory":
            if args.registration is None:
                parser.error("confirmation requires --registration")
            registration = check_registration(
                args.registration, args.cases, args.registration.parent / "config.json"
            )
            if registration["gold_sha256"] != file_sha(args.gold):
                raise ValueError("Frozen human gold changed")
            if registration["cases_sha256"] != file_sha(args.cases):
                raise ValueError("Frozen cases changed")
            if digest(read(args.registration.parent / "config.json")) != results["config_digest"]:
                raise ValueError("Results/config mismatch")
        report = evaluate(load_cases(args.cases), read(args.gold), results)
        report["results_sha256"] = file_sha(args.results)
        report["gold_sha256"] = file_sha(args.gold)
        write_json(args.out, report)
    elif args.command == "power":
        report = read(args.pilot_report)
        if report["status"] != "pilot":
            raise ValueError("Power may only use the independent pilot")
        primary = report["primary"]
        power = plan_n(primary["a_only"] + primary["b_only"], primary["n"], maximum=args.maximum)
        power["pilot_report_sha256"] = file_sha(args.pilot_report)
        if primary["n"] > 100:
            raise ValueError("Pilot exceeds the reserved 100 histories")
        power["pilot_alpha_nominal"] = report["alpha_nominal"]
        power["config_digest"] = report["config_digest"]
        power["manifest_sha256"] = report["manifest_sha256"]
        write_json(args.out, power)
    elif args.command == "freeze":
        freeze(args)
    elif args.command == "generate":
        registration = check_registration(args.registration, args.cases, args.config)
        config = read(args.config)
        env = OpenAICompatibleLLMConfig.from_env()
        if (
            env is None
            or env.model != config["generation_model"]
            or env.base_url != config["generation_base_url"]
        ):
            raise ValueError("Generation model/endpoint mismatch")
        results = read(args.results)
        if results["stage"] != "confirmatory" or results["config_digest"] != digest(config):
            raise ValueError("End-to-end requires the frozen confirmatory run")
        if results["cases_digest"] != digest([c.model_dump() for c in load_cases(args.cases)]):
            raise ValueError("Response generation case/result mismatch")
        write_json(
            args.out.with_suffix(".started.json"),
            {
                "registration_sha256": file_sha(args.registration),
                "results_sha256": file_sha(args.results),
                "status": "GENERATION_STARTED",
            },
        )
        llm = make_httpx_llm(env)
        try:
            chosen = [
                c
                for c in load_cases(args.cases)
                if c.case_id in registration["end_to_end_case_ids"]
            ]
            with args.out.with_suffix(".trace.jsonl").open("x", encoding="utf-8") as trace_file:

                def record_response(event: dict[str, Any]) -> None:
                    trace_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                    trace_file.flush()

                write_json(
                    args.out, generate_responses(chosen, results, llm, trace=record_response)
                )
        except Exception as exc:
            write_json(
                args.out.with_suffix(".failure.json"),
                {
                    "status": "GENERATION_ABORTED",
                    "error_type": type(exc).__name__,
                },
            )
            raise
        finally:
            llm.close()  # type: ignore[attr-defined]
    elif args.command == "response-packets":
        registration = read(args.registration)
        chosen = [
            c for c in load_cases(args.cases) if c.case_id in registration["end_to_end_case_ids"]
        ]
        roster = registration["readiness"]
        public, private = response_packets(
            chosen, read(args.responses), roster["response_raters"], roster["gold_raters"]
        )
        for i, payload in enumerate(public.values()):
            write_json(args.out / f"rater-{i + 1}.json", payload)
        write_json(args.out / "private-key.json", private)
    elif args.command == "response-summary":
        write_json(args.out, summarize_responses(read(args.key), [read(p) for p in args.ratings]))


if __name__ == "__main__":
    main()
