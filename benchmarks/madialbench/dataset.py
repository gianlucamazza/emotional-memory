"""MADial-Bench EN dataset loader (Addendum X).

Loads the two JSONL files released in the MADial-Bench repo (NAACL 2025, MIT
license; He et al., arXiv:2409.15240), pinned by sha256 per the pre-registration
`benchmarks/preregistration_addendum_x_madialbench_third_party.md`. The loader
FAILS on any hash mismatch — the study is defined on these exact bytes.

Query construction replicates the benchmark's own `Embedding.py`:
``"<Time>: 2024-06-16\\n"`` + all dialogue turns with index ``i < test_turn[0]``
(context up to the FIRST test turn, exclusive), verbatim concatenation.
Gold = the dialogue's ``relevant-id`` set, binary relevance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "madialbench"
DIALOGUE_FILE = DATA_DIR / "MADial-Bench-en-dialogue.jsonl"
MEMORY_FILE = DATA_DIR / "MADial-Bench-en-memory.jsonl"

# Pinned in the pre-registration; repo commit 572e3a10d6d01852a65e4508e0b3ab2a00d0710c.
DIALOGUE_SHA256 = "ba987172b4a720dd108c9a3b04855b8489ecde96fdf5d484d6522602ec6f4a31"
MEMORY_SHA256 = "d384b2d35e01ed364165001911af7f8193c808d168b106dea7b23435c7717aa5"

TIME_HEADER = "<Time>: 2024-06-16\n"

NEGATIVE_EMOTIONS = frozenset(
    {"Anxious", "Anxiety", "Sad", "Angry", "Disappointed", "Fear", "Frustrated"}
)


@dataclass(frozen=True)
class MadialMemory:
    """One memory-bank entry (global id space 1..160)."""

    memory_id: int
    time: str
    scene: str
    emotion: str
    event: str


@dataclass(frozen=True)
class MadialQuery:
    """One retrieval query (one per dialogue)."""

    query_id: int  # 0-based dialogue index (file order)
    user_id: int
    text: str  # TIME_HEADER + turns before the first test turn
    gold_ids: frozenset[int]  # relevant-id set, binary relevance


@dataclass(frozen=True)
class MadialDataset:
    memories: tuple[MadialMemory, ...]
    queries: tuple[MadialQuery, ...]


def _verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"{path.name}: sha256 mismatch — got {digest}, pinned {expected}. "
            "Refusing to run on unpinned data (see Addendum X pre-registration)."
        )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_dataset(
    dialogue_file: Path = DIALOGUE_FILE,
    memory_file: Path = MEMORY_FILE,
    *,
    verify_hashes: bool = True,
) -> MadialDataset:
    """Load and validate the pinned MADial-Bench EN split."""
    if verify_hashes:
        _verify_sha256(dialogue_file, DIALOGUE_SHA256)
        _verify_sha256(memory_file, MEMORY_SHA256)

    memories: list[MadialMemory] = []
    for row in _load_jsonl(memory_file):
        for mem_id, payload in row.items():
            assert isinstance(payload, dict)
            memories.append(
                MadialMemory(
                    memory_id=int(mem_id),
                    time=str(payload["time"]),
                    scene=str(payload["scene"]),
                    emotion=str(payload["emotion"]),
                    event=str(payload["event"]),
                )
            )
    memories.sort(key=lambda m: m.memory_id)

    bank_ids = {m.memory_id for m in memories}
    queries: list[MadialQuery] = []
    for idx, row in enumerate(_load_jsonl(dialogue_file)):
        turns = row["dialogue"]
        test_turns = row["test-turn"]
        gold = row["relevant-id"]
        assert isinstance(turns, list) and isinstance(test_turns, list) and isinstance(gold, list)
        first_test = int(test_turns[0])
        text = TIME_HEADER + "".join(str(t) for t in turns[:first_test])
        gold_ids = frozenset(int(g) for g in gold)
        if not gold_ids <= bank_ids:
            raise RuntimeError(f"dialogue {idx}: gold ids {gold_ids - bank_ids} not in bank")
        queries.append(
            MadialQuery(
                query_id=idx,
                user_id=int(row["user-id"]),  # type: ignore[call-overload]
                text=text,
                gold_ids=gold_ids,
            )
        )

    if len(memories) != 160 or len(queries) != 160:
        raise RuntimeError(
            f"expected 160 memories / 160 queries, got {len(memories)} / {len(queries)}"
        )
    return MadialDataset(memories=tuple(memories), queries=tuple(queries))
