"""Atomically bump version across upstream SSOT files, then propagate via sync-metadata.

Usage:
    uv run python scripts/bump_version.py X.Y.Z [--date YYYY-MM-DD] [--dry-run]

Steps:
    1. Update pyproject.toml [project] version
    2. Update CITATION.cff version + date-released
    3. Promote CHANGELOG.md [Unreleased] heading → [X.Y.Z] - DATE; update compare links
    4. Run scripts/sync_release_metadata.py --from-toml  (propagates to ~9 dependent files)
    5. Run scripts/preflight.py --fast X.Y.Z  (report; non-fatal: G5/G6 expected to fail)
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _replace(pattern: str, repl: str, text: str, description: str) -> str:
    updated, count = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Could not update {description}")
    return updated


def _repo_url() -> str:
    data = tomllib.loads((ROOT / "release.toml").read_text(encoding="utf-8"))
    return str(data["release"]["repo_url"]).rstrip("/")


def _bump_pyproject(text: str, version: str) -> str:
    return _replace(
        r'^version = "[^"]+"$',
        f'version = "{version}"',
        text,
        "pyproject.toml version",
    )


def _bump_citation(text: str, version: str, date: str) -> str:
    text = _replace(
        r'^version:\s*"[^"]+"$',
        f'version: "{version}"',
        text,
        "CITATION.cff version",
    )
    return _replace(
        r'^date-released:\s*"[^"]+"$',
        f'date-released: "{date}"',
        text,
        "CITATION.cff date-released",
    )


def _extract_prev_version(changelog: str) -> str:
    """Read the current version from the [Unreleased] compare link."""
    m = re.search(r"^\[Unreleased\]:\s+\S+/compare/v(\S+)\.\.\.HEAD", changelog, re.MULTILINE)
    if m is None:
        raise SystemExit("Could not extract previous version from CHANGELOG [Unreleased] link")
    return m.group(1)


def _bump_changelog(text: str, version: str, date: str, repo_url: str) -> str:
    prev = _extract_prev_version(text)

    # Promote [Unreleased] heading; keep empty Unreleased section above new version
    text = _replace(
        r"^## \[Unreleased\]$",
        f"## [Unreleased]\n\n## [{version}] - {date}",
        text,
        "CHANGELOG version heading",
    )

    # Update [Unreleased] compare link to point at the new version
    text = _replace(
        r"^\[Unreleased\]:\s+\S+/compare/v\S+\.\.\.HEAD$",
        f"[Unreleased]: {repo_url}/compare/v{version}...HEAD",
        text,
        "CHANGELOG [Unreleased] compare link",
    )

    # Insert new version compare link immediately after [Unreleased] link
    text = re.sub(
        r"^(\[Unreleased\]:.+)$",
        rf"\1\n[{version}]: {repo_url}/compare/v{prev}...v{version}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("version", metavar="VERSION", help="New version string (e.g. 0.9.1)")
    parser.add_argument(
        "--date",
        default=datetime.date.today().isoformat(),
        help="Release date ISO 8601 (default: today)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print diff without writing files")
    args = parser.parse_args(argv)

    version: str = args.version
    date: str = args.date
    dry_run: bool = args.dry_run

    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise SystemExit(f"Invalid version: {version!r} — expected X.Y.Z")

    repo_url = _repo_url()

    pyproject_path = ROOT / "pyproject.toml"
    citation_path = ROOT / "CITATION.cff"
    changelog_path = ROOT / "CHANGELOG.md"

    pyproject_new = _bump_pyproject(pyproject_path.read_text(encoding="utf-8"), version)
    citation_new = _bump_citation(citation_path.read_text(encoding="utf-8"), version, date)
    changelog_new = _bump_changelog(
        changelog_path.read_text(encoding="utf-8"), version, date, repo_url
    )

    changes = [
        (pyproject_path, pyproject_new),
        (citation_path, citation_new),
        (changelog_path, changelog_new),
    ]

    print(f"Bumping to v{version} ({date})")
    for path, new_text in changes:
        old_text = path.read_text(encoding="utf-8")
        if new_text == old_text:
            print(f"  (unchanged) {path.relative_to(ROOT)}")
            continue
        label = "(dry) " if dry_run else ""
        print(f"  {label}patched  {path.relative_to(ROOT)}")
        if dry_run:
            import difflib

            diff = list(
                difflib.unified_diff(
                    old_text.splitlines(),
                    new_text.splitlines(),
                    fromfile=f"a/{path.relative_to(ROOT)}",
                    tofile=f"b/{path.relative_to(ROOT)}",
                    lineterm="",
                )
            )
            for line in diff[:60]:
                print(f"    {line}")
            if len(diff) > 60:
                print(f"    ... ({len(diff) - 60} more lines)")
        else:
            path.write_text(new_text, encoding="utf-8")

    if dry_run:
        print("\nDry run — no files written and no downstream scripts run.")
        return 0

    # Propagate to all dependent files
    print("\nRunning sync_release_metadata --from-toml …")
    rc_sync = subprocess.run(
        ["uv", "run", "python", "scripts/sync_release_metadata.py", "--from-toml"],
        cwd=ROOT,
    )
    if rc_sync.returncode != 0:
        print("ERROR: sync_release_metadata failed — check output above", file=sys.stderr)
        return rc_sync.returncode

    # Preflight sanity check (non-fatal: G5 clean-tree + G6 on-main expected to fail)
    print(f"\nRunning preflight --fast {version} …")
    rc_pre = subprocess.run(
        ["uv", "run", "python", "scripts/preflight.py", "--fast", version],
        cwd=ROOT,
    )
    if rc_pre.returncode != 0:
        print(
            "\nNOTE: preflight --fast reported failures (see above).\n"
            "      G5 (clean tree) and G6 (on main) are expected until you commit\n"
            "      and are on main. Fix any other failures before 'make release'.",
            file=sys.stderr,
        )
    else:
        print(f"\nAll preflight --fast gates passed for v{version}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
