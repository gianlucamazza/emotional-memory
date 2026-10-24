"""Guard the owner-only publishable list for the planned HF dataset.

``benchmarks/datasets/PUBLISHABLE.txt`` is the allow-list for
``emotional-memory-benchmarks``. Third-party or third-party-derived sets
must never appear there. This module does not change benchmark scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "datasets"
PUBLISHABLE_PATH = DATASETS_DIR / "PUBLISHABLE.txt"

# Paths relative to benchmarks/datasets/. These stay in git and must not
# be listed as publishable (licenses recorded in sibling ATTRIBUTION files
# and benchmarks/datasets/README.md).
FORBIDDEN_RELATIVE_PATHS: frozenset[str] = frozenset(
    {
        "emobank_v1.json",
        "dailydialog_personas_v1.json",
        "madialbench/MADial-Bench-en-dialogue.jsonl",
        "madialbench/MADial-Bench-en-memory.jsonl",
        "esmemeval/evo_emo.json",
    }
)

FORBIDDEN_IDS: frozenset[str] = frozenset(
    {
        "emobank_v1",
        "madialbench",
        "MADial-Bench-en-dialogue",
        "MADial-Bench-en-memory",
        "esmemeval",
        "evo_emo",
        "dailydialog_personas_v1",
        "dailydialog_personas",
    }
)

# Owner-authored retrieval / affect sets that the Hub dataset may include.
OWNER_RELATIVE_PATHS: dict[str, str] = {
    "affect_reference_v1": "affect_reference_v1.jsonl",
    "realistic_recall_v1": "realistic_recall_v1.json",
    "realistic_recall_v2": "realistic_recall_v2.json",
    "realistic_recall_v2_it": "realistic_recall_v2_it.json",
    "realistic_recall_v2_es": "realistic_recall_v2_es.json",
    "realistic_recall_v2_fr": "realistic_recall_v2_fr.json",
    "realistic_recall_v3": "realistic_recall_v3.json",
    "realistic_recall_v3_noAF": "realistic_recall_v3_noAF.json",
    "realistic_recall_v4_noAF": "realistic_recall_v4_noAF.json",
    "realistic_recall_v5_gate": "realistic_recall_v5_gate.json",
    "realistic_recall_shared_history_v2": "realistic_recall_shared_history_v2.json",
}


def _parse_publishable(text: str) -> list[str]:
    ids: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def _entry_tokens(entry: str) -> set[str]:
    path = Path(entry)
    parts = {path.name, path.stem, *path.parts}
    return {part for part in parts if part not in {".", "/", ""}}


def test_publishable_list_exists_and_is_owner_only() -> None:
    assert PUBLISHABLE_PATH.is_file(), f"missing {PUBLISHABLE_PATH}"
    ids = _parse_publishable(PUBLISHABLE_PATH.read_text(encoding="utf-8"))
    assert ids, "PUBLISHABLE.txt has no dataset ids"

    forbidden_hits = [entry for entry in ids if _entry_tokens(entry) & FORBIDDEN_IDS]
    assert forbidden_hits == [], (
        f"third-party or third-party-derived set(s) listed as publishable: {forbidden_hits}"
    )

    assert set(ids) == set(OWNER_RELATIVE_PATHS), (
        "PUBLISHABLE.txt ids must match the owner-authored allow-list exactly: "
        f"extra={set(ids) - set(OWNER_RELATIVE_PATHS)}, "
        f"missing={set(OWNER_RELATIVE_PATHS) - set(ids)}"
    )

    for dataset_id in ids:
        relative = OWNER_RELATIVE_PATHS[dataset_id]
        path = DATASETS_DIR / relative
        assert path.is_file(), f"publishable id {dataset_id!r} missing file {path}"
        assert relative not in FORBIDDEN_RELATIVE_PATHS


def test_third_party_sets_remain_in_repo_and_stay_off_the_list() -> None:
    listed = set(_parse_publishable(PUBLISHABLE_PATH.read_text(encoding="utf-8")))
    for relative in FORBIDDEN_RELATIVE_PATHS:
        path = DATASETS_DIR / relative
        assert path.is_file(), f"third-party set missing from repo: {path}"
        assert relative not in listed
        assert Path(relative).stem not in listed
        assert Path(relative).parts[0] not in listed


def test_realistic_recall_publishable_json_declares_cc_by_4_0() -> None:
    for dataset_id, relative in OWNER_RELATIVE_PATHS.items():
        if not dataset_id.startswith("realistic_recall_"):
            continue
        payload = json.loads((DATASETS_DIR / relative).read_text(encoding="utf-8"))
        assert payload.get("license") == "CC-BY-4.0", (
            f"{relative} must carry license CC-BY-4.0 in top-level metadata "
            f"(got {payload.get('license')!r})"
        )


def test_attribution_notes_exist_for_third_party_sets() -> None:
    required = (
        DATASETS_DIR / "madialbench" / "ATTRIBUTION.md",
        DATASETS_DIR / "esmemeval" / "ATTRIBUTION.md",
        DATASETS_DIR / "emobank_v1.ATTRIBUTION.md",
        DATASETS_DIR / "dailydialog_personas_v1.ATTRIBUTION.md",
    )
    for path in required:
        assert path.is_file(), f"missing attribution note: {path}"
        text = path.read_text(encoding="utf-8")
        assert "PUBLISHABLE.txt" in text
        assert "not" in text.lower()
