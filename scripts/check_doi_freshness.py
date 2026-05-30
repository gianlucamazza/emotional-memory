"""Anti-recurrence gate: a release tag must carry a fresh Zenodo version DOI.

Background
----------
v0.11.2 and v0.11.3 shipped via the on-tag GitHub Actions workflow, which
publishes to PyPI + GitHub but does NOT run the Zenodo flow (``make release``).
As a result ``release.toml`` ``version_doi`` stayed at v0.11.1's value
(``10.5281/zenodo.20440996``) and the per-version metadata (CITATION.cff,
.zenodo.json, README BibTeX) shipped a DOI pointing at the wrong release.

Under Zenodo's model every version gets its own DOI, so a new release must have a
``version_doi`` different from the previous release's. This gate compares
``version_doi`` in ``release.toml`` at a *current* ref against the value at the
*previous* git tag and fails when they are equal. It is a pure-git, offline
check (no network) so it never flakes, and is meant to run in the on-tag release
workflow — the exact path the buggy releases took — before anything is published.

Design: the verdict logic (:func:`evaluate`) is pure and side-effect-free so it
is unit-tested directly; git/filesystem access lives only in the thin I/O helpers
and :func:`main`.

Usage::

    # Default: compare the working tree against the most recent tag before HEAD
    uv run python scripts/check_doi_freshness.py

    # Explicit refs:
    uv run python scripts/check_doi_freshness.py --current v0.11.2 --previous v0.11.1
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_VERSION_DOI_RE = re.compile(r'^version_doi\s*=\s*"([^"]*)"', re.MULTILINE)

# Exit codes
_PASS = 0
_FAIL = 1


@dataclass(frozen=True)
class Verdict:
    """Outcome of the freshness comparison."""

    code: int
    message: str


def parse_version_doi(release_toml_text: str) -> str | None:
    """Extract ``version_doi`` from release.toml *text*, or None if absent.

    Pure: a tolerant regex first (works on partial/edited files), then a full
    TOML parse as a fallback. No I/O.
    """
    match = _VERSION_DOI_RE.search(release_toml_text)
    if match:
        return match.group(1)
    try:
        data = tomllib.loads(release_toml_text)
    except tomllib.TOMLDecodeError:
        return None
    value = data.get("release", {}).get("version_doi")
    return str(value) if value is not None else None


def evaluate(
    current_doi: str | None,
    previous_doi: str | None,
    *,
    current_label: str = "current",
    previous_label: str = "previous",
) -> Verdict:
    """Decide whether the current DOI is fresh relative to the previous one.

    Pure and fully unit-testable. Semantics:
    - current_doi missing -> FAIL (can't validate a release without a DOI)
    - previous_doi missing (no prior release / unreadable) -> PASS (skip)
    - equal DOIs -> FAIL (stale: a new release must reserve a new DOI)
    - different DOIs -> PASS
    """
    if current_doi is None:
        return Verdict(
            _FAIL,
            f"ERROR: could not read version_doi from release.toml at {current_label!r}",
        )
    if previous_doi is None:
        return Verdict(
            _PASS,
            f"OK: no readable previous version_doi before {current_label!r} — "
            "check skipped (first release or shallow history).",
        )
    if current_doi == previous_doi:
        return Verdict(
            _FAIL,
            f"FAIL: version_doi at {current_label} == version_doi at {previous_label} "
            f"({current_doi}).\n"
            "  A new release must reserve a fresh Zenodo version DOI.\n"
            "  This tag was likely pushed without running 'make release' (which\n"
            "  reserves the DOI before tagging). Run 'make release VERSION=X.Y.Z'\n"
            "  to reserve + deposit on Zenodo, or fix release.toml version_doi.",
        )
    return Verdict(
        _PASS,
        f"OK: version_doi changed {previous_doi} ({previous_label}) -> "
        f"{current_doi} ({current_label}).",
    )


# ── thin git/filesystem I/O (kept out of the pure layer) ───────────────────────


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    # scripts/** waive S603/S607 project-wide: release tooling shells out to git
    # by design with fixed, non-user argv.
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _release_toml_at(ref: str) -> str | None:
    """Return release.toml contents at *ref* ('WORKTREE' for the working copy)."""
    if ref == "WORKTREE":
        return (ROOT / "release.toml").read_text(encoding="utf-8")
    result = _git(["show", f"{ref}:release.toml"])
    return result.stdout if result.returncode == 0 else None


def _doi_at(ref: str) -> str | None:
    text = _release_toml_at(ref)
    return parse_version_doi(text) if text is not None else None


def _previous_tag(before_ref: str) -> str | None:
    """Most recent ``v*`` tag strictly before *before_ref* (uses ``<ref>^``)."""
    result = _git(["describe", "--tags", "--abbrev=0", "--match", "v*", f"{before_ref}^"])
    out = result.stdout.strip()
    return out if result.returncode == 0 and out else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        default="WORKTREE",
        help="git ref (or WORKTREE) whose release.toml is the new release. Default: WORKTREE.",
    )
    parser.add_argument(
        "--previous",
        default=None,
        help="git ref of the previous release. Default: most recent v* tag before --current.",
    )
    args = parser.parse_args(argv)

    # The ref we derive the previous tag from: HEAD when comparing the worktree.
    anchor = "HEAD" if args.current == "WORKTREE" else args.current
    previous_ref = args.previous or _previous_tag(anchor)

    verdict = evaluate(
        _doi_at(args.current),
        _doi_at(previous_ref) if previous_ref else None,
        current_label=args.current,
        previous_label=previous_ref or "previous",
    )
    stream = sys.stdout if verdict.code == _PASS else sys.stderr
    print(verdict.message, file=stream)
    return verdict.code


if __name__ == "__main__":
    raise SystemExit(main())
