"""Read the pinned upstream archive without extracting files or using persona labels."""

from __future__ import annotations

import hashlib
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

ARCHIVE_SHA256 = "e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60"
SOURCE_URL = "https://parl.ai/downloads/msc/msc_v0.1.tar.gz"
SEED = 20261004


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Never silently replace an earlier study artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    memory_id: str
    text: str = Field(min_length=1)
    session: int = Field(ge=0)
    order: int = Field(ge=0)


class Case(BaseModel):
    """The entire system-visible input. No gold, persona summaries or future turns."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    group_id: str
    query: str = Field(min_length=1)
    context: str
    query_session: int = Field(ge=1)
    candidates: tuple[Candidate, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def check_history(self) -> Case:
        if any(c.session >= self.query_session for c in self.candidates):
            raise ValueError("Future/current-session memory leakage")
        if len({c.memory_id for c in self.candidates}) != len(self.candidates):
            raise ValueError("Duplicate memory ID")
        if [c.order for c in self.candidates] != list(range(len(self.candidates))):
            raise ValueError("Memory order must be chronological and contiguous")
        return self

    @property
    def retrieval_text(self) -> str:
        return f"{self.context}\n{self.query}".strip()


def load_cases(path: Path) -> list[Case]:
    cases = [Case.model_validate(row) for row in json.loads(path.read_text())]
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("Duplicate case ID")
    if len({c.group_id for c in cases}) != len(cases):
        raise ValueError("More than one query per independent history")
    return cases


def _dialog_key(dialog: list[dict[str, Any]]) -> str:
    return digest([" ".join(t["text"].split()).casefold() for t in dialog])


def build_cases(records: list[dict[str, Any]], *, seed: int = SEED) -> list[Case]:
    """Union shared antecedents, choose longest history, then one seeded human turn."""
    parent: dict[str, str] = {}

    def root(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners: dict[str, str] = {}
    for row in records:
        key = str(row["metadata"]["initial_data_id"])
        root(key)
        # Include current dialogs too: descendants may reuse them as antecedents.
        for dialog in [r["dialog"] for r in row["previous_dialogs"]] + [row["dialog"]]:
            fingerprint = _dialog_key(dialog)
            if fingerprint in owners:
                a, b = root(key), root(owners[fingerprint])
                parent[max(a, b)] = min(a, b)
            owners[fingerprint] = key
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[root(str(row["metadata"]["initial_data_id"]))].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        eligible = [r for r in rows if r["previous_dialogs"] and len(r["dialog"]) >= 2]
        if not eligible:
            continue
        row = min(eligible, key=lambda r: (-len(r["previous_dialogs"]), digest(r)))
        rng = random.Random(f"{seed}:{key}")
        # Select a turn with a following response, but never export that response.
        turn = rng.randrange(len(row["dialog"]) - 1)
        candidates = []
        for session, previous in enumerate(row["previous_dialogs"]):
            dialog = previous["dialog"]
            for start in range(0, len(dialog), 2):
                text = "\n".join(
                    f"Speaker {i % 2 + 1}: {dialog[i]['text']}"
                    for i in range(start, min(start + 2, len(dialog)))
                )
                candidates.append(
                    Candidate(
                        memory_id=digest([key, session, start])[:24],
                        text=text,
                        session=session,
                        order=len(candidates),
                    )
                )
        context = "\n".join(
            f"Speaker {i % 2 + 1}: {row['dialog'][i]['text']}" for i in range(turn)
        )
        result.append(
            Case(
                case_id=digest([key, len(row["previous_dialogs"]), turn])[:24],
                group_id=digest(key),
                query=f"Speaker {turn % 2 + 1}: {row['dialog'][turn]['text']}",
                context=context,
                query_session=len(row["previous_dialogs"]),
                candidates=tuple(candidates),
            )
        )
    random.Random(seed).shuffle(result)
    return result


def prepare(archive: Path, out: Path) -> dict[str, Any]:
    if file_sha(archive) != ARCHIVE_SHA256:
        raise ValueError("Archive does not match the upstream MSC v0.1 checksum")
    records = []
    members = {}
    with tarfile.open(archive) as stream:
        for member in sorted(stream.getmembers(), key=lambda m: m.name):
            if "/msc_dialogue/session_" not in member.name or not member.name.endswith(
                ("train.txt", "valid.txt", "test.txt")
            ):
                continue
            source = stream.extractfile(member)
            if source is None:
                raise ValueError("Missing archive member")
            data = source.read()
            members[member.name] = hashlib.sha256(data).hexdigest()
            records.extend(json.loads(line) for line in data.splitlines() if line.strip())
    cases = build_cases(records)
    splits = {"development": cases[:200], "pilot": cases[200:300], "reserve": cases[300:]}
    manifest: dict[str, Any] = {
        "status": "PREPARATORY_NOT_EXECUTED",
        "source_url": SOURCE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "member_sha256": members,
        "seed": SEED,
        "source_records": len(records),
        "independent_histories": len(cases),
        "license_status": "REVIEW_REQUIRED: archive contains no license file",
        "splits": {},
    }
    for name, items in splits.items():
        path = out / f"{name}.json"
        write_json(path, [c.model_dump(mode="json") for c in items])
        manifest["splits"][name] = {
            "sha256": file_sha(path),
            "count": len(items),
            "candidate_count": sum(len(c.candidates) for c in items),
        }
    write_json(out / "manifest.json", manifest)
    return manifest
