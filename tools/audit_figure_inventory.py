#!/usr/bin/env python3
"""Verify figure_inventory.json integrity from the command line.

Checks:
  - Every declared output exists on disk.
  - No orphan files in tracked asset directories.
  - Every source_benchmark path exists.
  - Every referenced_in file contains the figure stem.
  - Every asset cited in README.md / docs/research/*.md / paper/main.tex
    is declared in the inventory.
  - Every non-null claim_id exists in claim_validation_matrix.json.

Exit 0 if all checks pass; exit 1 with a failure summary.

Run from the repo root:
    python tools/audit_figure_inventory.py
    uv run python tools/audit_figure_inventory.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / "docs" / "research" / "figure_inventory.json"
MATRIX_PATH = REPO_ROOT / "docs" / "research" / "claim_validation_matrix.json"

TRACKED_DIRS: list[tuple[Path, str]] = [
    (REPO_ROOT / "docs" / "images", "*.png"),
    (REPO_ROOT / "docs" / "images" / "research", "*.png"),
    (REPO_ROOT / "docs" / "images" / "research", "*.pdf"),
    (REPO_ROOT / "paper" / "figures", "*.pdf"),
]
PUBLIC_ASSET_PREFIXES = ("docs/images/", "paper/figures/")


def _load() -> tuple[list[dict], set[str]]:  # type: ignore[type-arg]
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    figures: list[dict] = inventory["figures"]  # type: ignore[type-arg]
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    valid_claim_ids = {c["claim_id"] for c in matrix.get("claims", [])}
    return figures, valid_claim_ids


def _collect_cited_assets() -> set[str]:
    cited: set[str] = set()
    md_files = [REPO_ROOT / "README.md", *(REPO_ROOT / "docs" / "research").glob("*.md")]
    md_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    for md_file in md_files:
        if not md_file.exists():
            continue
        for m in md_pattern.finditer(md_file.read_text(encoding="utf-8")):
            raw = m.group(1)
            if md_file.parent != REPO_ROOT:
                path = (md_file.parent / raw).resolve().relative_to(REPO_ROOT).as_posix()
            else:
                path = raw
            if any(path.startswith(pfx) for pfx in PUBLIC_ASSET_PREFIXES):
                cited.add(path)
    tex = REPO_ROOT / "paper" / "main.tex"
    if tex.exists():
        tex_pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
        for m in tex_pattern.finditer(tex.read_text(encoding="utf-8")):
            raw = m.group(1)
            path = (REPO_ROOT / "paper" / raw).resolve().relative_to(REPO_ROOT).as_posix()
            if any(path.startswith(pfx) for pfx in PUBLIC_ASSET_PREFIXES):
                cited.add(path)
    return cited


def main() -> int:
    figures, valid_claim_ids = _load()
    failures: list[str] = []

    declared: set[str] = set()
    for fig in figures:
        for out in fig["outputs"]:
            declared.add(out)

    # Forward: outputs exist
    failures.extend(
        f"[{fig['stem']}] missing output: {out}"
        for fig in figures
        for out in fig["outputs"]
        if not (REPO_ROOT / out).exists()
    )

    # Backward: no orphans
    for directory, pattern in TRACKED_DIRS:
        if not directory.exists():
            continue
        for path in directory.glob(pattern):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel not in declared:
                failures.append(f"orphan (not in inventory): {rel}")

    # Source liveness
    failures.extend(
        f"[{fig['stem']}] missing source_benchmark: {src}"
        for fig in figures
        for src in fig["source_benchmarks"]
        if not (REPO_ROOT / src).exists()
    )

    # Reference liveness
    for fig in figures:
        for ref in fig["referenced_in"]:
            ref_path = REPO_ROOT / ref
            if not ref_path.exists():
                failures.append(f"[{fig['stem']}] referenced_in file missing: {ref}")
            elif fig["stem"] not in ref_path.read_text(encoding="utf-8"):
                failures.append(f"[{fig['stem']}] stem not found in {ref}")

    # Cross-check public surfaces
    failures.extend(
        f"cited but not in inventory: {asset}"
        for asset in sorted(_collect_cited_assets() - declared)
    )

    # Claim sanity
    for fig in figures:
        cid = fig.get("claim_id")
        if cid is not None and cid not in valid_claim_ids:
            failures.append(f"[{fig['stem']}] unknown claim_id: {cid}")

    if failures:
        print(f"FAIL — {len(failures)} issue(s) in figure_inventory.json:\n")
        for f in failures:
            print(f"  {f}")
        print()
        return 1

    print(f"OK — {len(figures)} figures, all inventory checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
