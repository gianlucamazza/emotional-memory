"""Check that the Python minimum version is consistent across the repository.

Single source of truth: `requires-python` in `pyproject.toml`.
All other references (ruff target-version, mypy python_version, basedpyright
pythonVersion, CI matrix, etc.) must match the floor declared there.

This catches drift like bumping `requires-python` to ">=3.12" but forgetting
to update CI matrix or mypy config.

Usage:
    uv run python scripts/check_python_version_consistency.py
    uv run python scripts/check_python_version_consistency.py --fix-suggest

Exit codes:
    0 — all references consistent
    1 — drift detected (printed to stderr)
    2 — script error (e.g. could not parse pyproject.toml)
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Finding:
    file: Path
    field: str
    expected: str
    actual: str

    def render(self) -> str:
        rel = self.file.relative_to(ROOT)
        return f"  {rel}: {self.field} = {self.actual!r}, expected {self.expected!r}"


def _read_pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _floor_from_requires_python(spec: str) -> tuple[int, int]:
    """Extract the minimum minor version from a `requires-python` specifier.

    Supports the common forms ">=3.11", ">=3.11,<4", "~=3.11". Raises if the
    floor is not expressible as Python (3, minor).
    """
    match = re.search(r">=\s*3\.(\d+)", spec)
    if match is None:
        match = re.search(r"~=\s*3\.(\d+)", spec)
    if match is None:
        raise SystemExit(f"Cannot extract minor-version floor from {spec!r}")
    return (3, int(match.group(1)))


def _check_ruff_target(data: dict, floor: tuple[int, int]) -> Finding | None:
    target = data.get("tool", {}).get("ruff", {}).get("target-version")
    if target is None:
        return None
    expected = f"py3{floor[1]}"
    if target != expected:
        return Finding(
            file=ROOT / "pyproject.toml",
            field="tool.ruff.target-version",
            expected=expected,
            actual=target,
        )
    return None


def _check_mypy_python_version(data: dict, floor: tuple[int, int]) -> Finding | None:
    pv = data.get("tool", {}).get("mypy", {}).get("python_version")
    if pv is None:
        return None
    expected = f"3.{floor[1]}"
    if pv != expected:
        return Finding(
            file=ROOT / "pyproject.toml",
            field="tool.mypy.python_version",
            expected=expected,
            actual=pv,
        )
    return None


def _check_basedpyright_python_version(data: dict, floor: tuple[int, int]) -> Finding | None:
    pv = data.get("tool", {}).get("basedpyright", {}).get("pythonVersion")
    if pv is None:
        return None
    expected = f"3.{floor[1]}"
    if pv != expected:
        return Finding(
            file=ROOT / "pyproject.toml",
            field="tool.basedpyright.pythonVersion",
            expected=expected,
            actual=pv,
        )
    return None


def _check_classifiers(data: dict, floor: tuple[int, int]) -> list[Finding]:
    """Every supported version (floor up to declared max) must have a classifier."""
    classifiers: list[str] = data.get("project", {}).get("classifiers", [])
    seen_minors: set[int] = set()
    for c in classifiers:
        m = re.match(r"Programming Language :: Python :: 3\.(\d+)$", c)
        if m:
            seen_minors.add(int(m.group(1)))

    if not seen_minors:
        return []  # no classifiers — out of scope

    # The floor minor must be present as a classifier.
    findings: list[Finding] = []
    if floor[1] not in seen_minors:
        findings.append(
            Finding(
                file=ROOT / "pyproject.toml",
                field="project.classifiers",
                expected=f"includes 'Programming Language :: Python :: 3.{floor[1]}'",
                actual=f"missing 3.{floor[1]} classifier (have {sorted(seen_minors)})",
            )
        )
    # Every minor below the floor must NOT be classified as supported.
    findings.extend(
        Finding(
            file=ROOT / "pyproject.toml",
            field="project.classifiers",
            expected=f"no classifier below 3.{floor[1]}",
            actual=f"has obsolete classifier 'Python :: 3.{m}'",
        )
        for m in seen_minors
        if m < floor[1]
    )
    return findings


def _check_ci_matrix(floor: tuple[int, int]) -> list[Finding]:
    ci = ROOT / ".github" / "workflows" / "ci.yml"
    if not ci.exists():
        return []
    text = ci.read_text(encoding="utf-8")
    # Match `python-version: ["3.11", "3.12", ...]`
    m = re.search(r"python-version:\s*\[([^\]]+)\]", text)
    if not m:
        return []
    versions = re.findall(r'"3\.(\d+)"', m.group(1))
    minors = {int(v) for v in versions}
    findings: list[Finding] = []
    if floor[1] not in minors:
        findings.append(
            Finding(
                file=ci,
                field="jobs.test.strategy.matrix.python-version",
                expected=f'includes "3.{floor[1]}"',
                actual=f"missing 3.{floor[1]} (have {sorted(minors)})",
            )
        )
    findings.extend(
        Finding(
            file=ci,
            field="jobs.test.strategy.matrix.python-version",
            expected=f"no version below 3.{floor[1]}",
            actual=f'has obsolete "3.{m_}"',
        )
        for m_ in minors
        if m_ < floor[1]
    )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix-suggest",
        action="store_true",
        help="Print suggested fixes (does not modify files).",
    )
    args = parser.parse_args()

    data = _read_pyproject()
    requires_python = data.get("project", {}).get("requires-python")
    if not requires_python:
        print("ERROR: project.requires-python missing from pyproject.toml", file=sys.stderr)
        return 2

    floor = _floor_from_requires_python(requires_python)
    print(f"SSOT: requires-python = {requires_python!r} → floor = 3.{floor[1]}")

    findings: list[Finding] = [
        f
        for f in (
            _check_ruff_target(data, floor),
            _check_mypy_python_version(data, floor),
            _check_basedpyright_python_version(data, floor),
        )
        if f is not None
    ]
    findings.extend(_check_classifiers(data, floor))
    findings.extend(_check_ci_matrix(floor))

    if not findings:
        print("OK: all Python-version references consistent with floor.")
        return 0

    print(f"\nFAIL: {len(findings)} inconsistencies found:", file=sys.stderr)
    for f in findings:
        print(f.render(), file=sys.stderr)

    if args.fix_suggest:
        print("\nSuggested fixes:", file=sys.stderr)
        print(
            f"  Edit pyproject.toml: align ruff/mypy/basedpyright to py3{floor[1]} / 3.{floor[1]}",
            file=sys.stderr,
        )
        print(
            f"  Edit .github/workflows/ci.yml matrix: drop versions < 3.{floor[1]}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
