#!/usr/bin/env python3
"""Verify that all paths cited in claim_validation_matrix.json exist on disk.

Exit 0 if all refs are present; exit 1 with a summary of missing paths.
Run from the repo root:
    python tools/audit_claim_refs.py
    uv run python tools/audit_claim_refs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO_ROOT / "docs" / "research" / "claim_validation_matrix.json"

REF_KEYS = ("evidence_refs", "benchmark_refs", "protocol_refs", "limitations_refs")


def main() -> int:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    missing: list[tuple[str, str, str]] = []

    for claim in matrix.get("claims", []):
        claim_id: str = claim.get("claim_id", "<unknown>")
        for key in REF_KEYS:
            for ref_path in claim.get(key, []):
                full = REPO_ROOT / ref_path
                if not full.exists():
                    missing.append((claim_id, key, ref_path))

    if missing:
        print(f"FAIL — {len(missing)} missing ref(s) in claim_validation_matrix.json:\n")
        for claim_id, key, path in missing:
            print(f"  [{claim_id}] {key}: {path}")
        print()
        return 1

    n_refs = sum(len(claim.get(key, [])) for claim in matrix.get("claims", []) for key in REF_KEYS)
    print(f"OK — all {n_refs} refs in claim_validation_matrix.json exist on disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
