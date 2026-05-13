"""Integrity tests for docs/research/figure_inventory.json.

Checks:
1. Schema valid (all required fields, known category/plot_kind values).
2. Forward: every declared output file exists on disk.
3. Backward: every file in docs/images/, docs/images/research/, paper/figures/
   is declared in the inventory (no orphans).
4. Source liveness: every source_benchmark path exists on disk.
5. Reference liveness: for each referenced_in path, at least one occurrence of
   the figure stem appears in that file.
6. Cross-check public surfaces: every docs/images/* and paper/figures/* asset
   cited via ![...](...) in README.md / docs/research/*.md or via
   \\includegraphics in paper/main.tex is declared in the inventory.
7. Claim sanity: if claim_id is set, it must exist in claim_validation_matrix.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "docs" / "research" / "figure_inventory.json"
MATRIX_PATH = ROOT / "docs" / "research" / "claim_validation_matrix.json"

KNOWN_CATEGORIES = {"didactic", "evidence", "paper"}
KNOWN_PLOT_KINDS = {
    "scatter",
    "line",
    "line_multipanel",
    "radar",
    "heatmap",
    "network_graph",
    "bar_with_ci",
    "grouped_bar",
    "grouped_bar_with_ci",
    "bar_forest",
}
TRACKED_DIRS: list[tuple[Path, str]] = [
    (ROOT / "docs" / "images", "*.png"),
    (ROOT / "docs" / "images" / "research", "*.png"),
    (ROOT / "docs" / "images" / "research", "*.pdf"),
    (ROOT / "paper" / "figures", "*.pdf"),
]
PUBLIC_MD_DIRS = [
    ROOT / "docs" / "research",
]
PUBLIC_ASSET_PREFIXES = ("docs/images/", "paper/figures/")


@pytest.fixture(scope="module")
def inventory() -> dict:  # type: ignore[type-arg]
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def figures(inventory: dict) -> list[dict]:  # type: ignore[type-arg]
    return inventory["figures"]  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def valid_claim_ids() -> set[str]:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    return {c["claim_id"] for c in matrix.get("claims", [])}


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------


def test_inventory_version(inventory: dict) -> None:
    assert inventory.get("inventory_version") == "1.0"


def test_required_fields(figures: list[dict]) -> None:  # type: ignore[type-arg]
    required = {
        "stem",
        "category",
        "generator",
        "outputs",
        "source_benchmarks",
        "referenced_in",
        "claim_id",
        "plot_kind",
        "description",
    }
    for fig in figures:
        missing = required - fig.keys()
        assert not missing, f"Figure '{fig.get('stem')}' missing fields: {missing}"


def test_known_categories(figures: list[dict]) -> None:  # type: ignore[type-arg]
    for fig in figures:
        assert fig["category"] in KNOWN_CATEGORIES, (
            f"'{fig['stem']}' has unknown category '{fig['category']}'"
        )


def test_known_plot_kinds(figures: list[dict]) -> None:  # type: ignore[type-arg]
    for fig in figures:
        assert fig["plot_kind"] in KNOWN_PLOT_KINDS, (
            f"'{fig['stem']}' has unknown plot_kind '{fig['plot_kind']}'"
        )


def test_outputs_nonempty(figures: list[dict]) -> None:  # type: ignore[type-arg]
    for fig in figures:
        assert fig["outputs"], f"'{fig['stem']}' has empty outputs list"


def test_unique_stems(figures: list[dict]) -> None:  # type: ignore[type-arg]
    stems = [fig["stem"] for fig in figures]
    dupes = {s for s in stems if stems.count(s) > 1}
    assert not dupes, f"Duplicate stems in inventory: {dupes}"


# ---------------------------------------------------------------------------
# 2. Forward: declared outputs must exist on disk
# ---------------------------------------------------------------------------


def test_outputs_exist_on_disk(figures: list[dict]) -> None:  # type: ignore[type-arg]
    missing = []
    for fig in figures:
        for out in fig["outputs"]:
            path = ROOT / out
            if not path.exists():
                missing.append(f"[{fig['stem']}] {out}")
    assert not missing, "Declared outputs missing from disk:\n" + "\n".join(missing)


# ---------------------------------------------------------------------------
# 3. Backward: no orphan files in tracked directories
# ---------------------------------------------------------------------------


def test_no_orphan_files(figures: list[dict]) -> None:  # type: ignore[type-arg]
    declared: set[str] = set()
    for fig in figures:
        for out in fig["outputs"]:
            declared.add(out)

    orphans = []
    for directory, pattern in TRACKED_DIRS:
        if not directory.exists():
            continue
        for path in directory.glob(pattern):
            rel = path.relative_to(ROOT).as_posix()
            if rel not in declared:
                orphans.append(rel)

    assert not orphans, "Files on disk not declared in inventory:\n" + "\n".join(sorted(orphans))


# ---------------------------------------------------------------------------
# 4. Source liveness
# ---------------------------------------------------------------------------


def test_source_benchmarks_exist(figures: list[dict]) -> None:  # type: ignore[type-arg]
    missing = [
        f"[{fig['stem']}] {src}"
        for fig in figures
        for src in fig["source_benchmarks"]
        if not (ROOT / src).exists()
    ]
    assert not missing, "Declared source benchmarks missing:\n" + "\n".join(missing)


# ---------------------------------------------------------------------------
# 5. Reference liveness: stem appears in each referenced_in file
# ---------------------------------------------------------------------------


def test_reference_liveness(figures: list[dict]) -> None:  # type: ignore[type-arg]
    broken = []
    for fig in figures:
        stem = fig["stem"]
        for ref in fig["referenced_in"]:
            ref_path = ROOT / ref
            if not ref_path.exists():
                broken.append(f"[{stem}] referenced_in file missing: {ref}")
                continue
            content = ref_path.read_text(encoding="utf-8")
            if stem not in content:
                broken.append(f"[{stem}] stem not found in {ref}")
    assert not broken, "Reference liveness failures:\n" + "\n".join(broken)


# ---------------------------------------------------------------------------
# 6. Cross-check public surfaces
# ---------------------------------------------------------------------------


def _collect_cited_assets() -> set[str]:
    """Return relative paths (repo-root-relative) of all images cited in
    README.md, docs/research/*.md, and paper/main.tex that fall under
    docs/images/ or paper/figures/."""
    cited: set[str] = set()

    # Markdown: ![alt](path)
    md_files = [ROOT / "README.md", *(ROOT / "docs" / "research").glob("*.md")]
    md_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    for md_file in md_files:
        if not md_file.exists():
            continue
        for m in md_pattern.finditer(md_file.read_text(encoding="utf-8")):
            raw = m.group(1)
            # Normalise ../images/research/foo.png relative to docs/research/
            if md_file.parent != ROOT:
                path = (md_file.parent / raw).resolve().relative_to(ROOT).as_posix()
            else:
                path = raw
            if any(path.startswith(pfx) for pfx in PUBLIC_ASSET_PREFIXES):
                cited.add(path)

    # LaTeX: \includegraphics[...]{path} — relative to paper/
    tex = ROOT / "paper" / "main.tex"
    if tex.exists():
        tex_pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
        for m in tex_pattern.finditer(tex.read_text(encoding="utf-8")):
            raw = m.group(1)
            path = (ROOT / "paper" / raw).resolve().relative_to(ROOT).as_posix()
            if any(path.startswith(pfx) for pfx in PUBLIC_ASSET_PREFIXES):
                cited.add(path)

    return cited


def test_cited_assets_in_inventory(figures: list[dict]) -> None:  # type: ignore[type-arg]
    declared: set[str] = set()
    for fig in figures:
        for out in fig["outputs"]:
            declared.add(out)

    cited = _collect_cited_assets()
    undeclared = cited - declared
    assert not undeclared, "Assets cited in public surfaces but not in inventory:\n" + "\n".join(
        sorted(undeclared)
    )


# ---------------------------------------------------------------------------
# 7. Claim sanity
# ---------------------------------------------------------------------------


def test_claim_ids_valid(figures: list[dict], valid_claim_ids: set[str]) -> None:  # type: ignore[type-arg]
    bad = []
    for fig in figures:
        cid = fig.get("claim_id")
        if cid is not None and cid not in valid_claim_ids:
            bad.append(f"[{fig['stem']}] unknown claim_id '{cid}'")
    assert not bad, "Unknown claim IDs in inventory:\n" + "\n".join(bad)
