"""Binary-relevance ranking metrics and auditable longitudinal report metadata."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarks.realistic.runner import ReplayDataset
    from emotional_memory import EmotionalMemoryConfig


def ranking_metrics(retrieved: list[str], relevant: set[str], k: int) -> dict[str, float]:
    """Binary relevance; repeated IDs never earn additional gain. MRR is truncated at k."""
    if k < 1 or not relevant:
        raise ValueError("Ranking metrics require positive k and nonempty relevance judgments")
    seen: set[str] = set()
    gains = []
    for item in retrieved[:k]:
        gains.append(float(item in relevant and item not in seen))
        seen.add(item)
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return {
        "recall_at_k": sum(gains) / len(relevant),
        "ndcg_at_k": dcg / ideal,
        "reciprocal_rank_at_k": next(
            (1 / rank for rank, gain in enumerate(gains, start=1) if gain), 0.0
        ),
    }


def reproducibility_manifest(
    dataset: ReplayDataset,
    config: EmotionalMemoryConfig,
    *,
    seed: int,
    top_k: int,
    embedder: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Identify effective inputs, implementation and environment, not run wall time."""
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=root,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in (root / "src", root / "benchmarks")
        for path in sorted(directory.rglob("*.py"))
    }
    return {
        "schema_version": 1,
        "package_version": version("emotional-memory"),
        "git_revision": revision,
        "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "source_files": hashes,
        "dataset_sha256": hashlib.sha256(
            json.dumps(dataset.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest(),
        "seed": seed,
        "seed_usage": "UUID5 namespace and bootstrap resampling",
        "top_k": top_k,
        "engine_config": config.model_dump(mode="json"),
        "embedder": embedder
        if embedder is not None
        else {"class": "TokenHashEmbedder", "dimension": 64},
        "appraisal": "recorded dataset query affect; no model inference",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {name: version(name) for name in ("numpy", "pydantic")},
        "clock": {"epoch": "2026-01-01T00:00:00+00:00", "step_seconds": 1},
        "checkpoint": "empty store and default affective state, reset for each scenario",
        "execution": "dedicated synchronous single-threaded process",
        "reproducibility_scope": "same source hashes, inputs, configuration and environment",
    }


def state_divergence(systems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare paired event snapshots to cosine; hashes indicate difference, not quality."""
    baseline = next(system for system in systems if system["system"] == "cosine")
    rows = []
    for system in systems:
        if system is baseline:
            continue
        for scenario, reference in zip(system["scenarios"], baseline["scenarios"], strict=True):
            if scenario["scenario_id"] != reference["scenario_id"]:
                raise ValueError("Scenario alignment mismatch")
            for event, other in zip(scenario["trace"], reference["trace"], strict=True):
                if (event["index"], event["kind"], event["input_sha256"]) != (
                    other["index"],
                    other["kind"],
                    other["input_sha256"],
                ):
                    raise ValueError("Shared-history input alignment mismatch")
                memories = {m["id"]: m for m in event["snapshot"]["memories"]}
                refs = {m["id"]: m for m in other["snapshot"]["memories"]}
                rows.append(
                    {
                        "system": system["system"],
                        "baseline": "cosine",
                        "scenario_id": scenario["scenario_id"],
                        "event_index": event["index"],
                        "kind": event["kind"],
                        "state_equal": event["state_sha256"] == other["state_sha256"],
                        "affective_state_equal": event["snapshot"]["state"]
                        == other["snapshot"]["state"],
                        "different_memory_count": sum(
                            memories.get(key) != refs.get(key)
                            for key in memories.keys() | refs.keys()
                        ),
                    }
                )
    return rows
