"""Validate that author, license, and keywords are consistent across
package-metadata files.

Single source of truth: `pyproject.toml`. The following derived files must
agree with what `[project]` declares:

    * CITATION.cff      — author family-names/given-names/email, license
    * codemeta.json     — author givenName/familyName/email, license, keywords
    * .zenodo.json      — creators.name, keywords

This script reports drift but does NOT modify files. It is meant to be a
fast guard in CI: when a contributor edits one of the duplicated fields,
they get an immediate red signal pointing to which file to fix.

Usage:
    uv run python scripts/check_metadata_ssot.py

Exit codes:
    0 — consistent
    1 — drift detected (reported to stderr)
    2 — script error (e.g. malformed file)
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_cff_field(text: str, field: str) -> str | None:
    """Extract a top-level scalar field from a CITATION.cff. Naive but enough
    for `license`, since we only need exact equality."""
    m = re.search(rf"^{field}:\s*(.+?)$", text, re.MULTILINE)
    if not m:
        return None
    val = m.group(1).strip()
    if val.startswith(('"', "'")) and val.endswith(('"', "'")):
        val = val[1:-1]
    return val


def _cff_first_author(text: str) -> dict[str, str]:
    """Parse the first author block from CITATION.cff (top-level `authors:`)."""
    # Match the first `- family-names: ...` block under `authors:` (one indent level).
    block = re.search(
        r"^authors:\s*\n((?:  -[\s\S]*?)+?)(?=^[^\s]|\Z)",
        text,
        re.MULTILINE,
    )
    if not block:
        return {}
    first_entry = re.search(
        r"-\s*family-names:\s*(.+?)\s*\n"
        r"\s+given-names:\s*(.+?)\s*\n"
        r"(?:\s+email:\s*(.+?)\s*\n)?",
        block.group(1),
    )
    if not first_entry:
        return {}
    return {
        "family-names": first_entry.group(1).strip(),
        "given-names": first_entry.group(2).strip(),
        "email": (first_entry.group(3) or "").strip(),
    }


def _expected_author(pyproject: dict) -> dict[str, str]:
    """Return canonical {given, family, email} from pyproject [project].authors[0]."""
    authors = pyproject.get("project", {}).get("authors") or []
    if not authors:
        raise SystemExit("pyproject.toml [project].authors is empty")
    first = authors[0]
    name = first.get("name", "")
    email = first.get("email", "")
    # Split "Given Family" or "Given Middle Family" → last token is family.
    parts = name.split()
    if len(parts) < 2:
        raise SystemExit(f"Cannot split author name into given/family: {name!r}")
    return {
        "given-names": " ".join(parts[:-1]),
        "family-names": parts[-1],
        "email": email,
    }


def check_author(pyproject: dict, findings: list[str]) -> None:
    expected = _expected_author(pyproject)

    # CITATION.cff
    cff_path = ROOT / "CITATION.cff"
    if cff_path.exists():
        cff_text = cff_path.read_text(encoding="utf-8")
        cff_author = _cff_first_author(cff_text)
        findings.extend(
            f"CITATION.cff: author.{key} = {cff_author.get(key)!r}, "
            f"expected {expected[key]!r} (from pyproject.toml)"
            for key in ("given-names", "family-names", "email")
            if cff_author.get(key, "") != expected[key]
        )

    # codemeta.json
    cm_path = ROOT / "codemeta.json"
    if cm_path.exists():
        cm = _read_json(cm_path)
        cm_authors = cm.get("author") or []
        if cm_authors:
            first = cm_authors[0]
            for src_key, cm_key in (
                ("given-names", "givenName"),
                ("family-names", "familyName"),
                ("email", "email"),
            ):
                if first.get(cm_key, "") != expected[src_key]:
                    findings.append(
                        f"codemeta.json: author[0].{cm_key} = {first.get(cm_key)!r}, "
                        f"expected {expected[src_key]!r} (from pyproject.toml)"
                    )

    # .zenodo.json — "Family, Given" format
    zen_path = ROOT / ".zenodo.json"
    if zen_path.exists():
        zen = _read_json(zen_path)
        zen_creators = zen.get("creators") or []
        if zen_creators:
            expected_zen = f"{expected['family-names']}, {expected['given-names']}"
            actual_zen = zen_creators[0].get("name", "")
            if actual_zen != expected_zen:
                findings.append(
                    f".zenodo.json: creators[0].name = {actual_zen!r}, "
                    f"expected {expected_zen!r} (from pyproject.toml)"
                )


def check_license(pyproject: dict, findings: list[str]) -> None:
    expected = pyproject.get("project", {}).get("license", "")
    if not expected:
        return

    cff_path = ROOT / "CITATION.cff"
    if cff_path.exists():
        actual = _read_cff_field(cff_path.read_text(encoding="utf-8"), "license") or ""
        if actual != expected:
            findings.append(
                f"CITATION.cff: license = {actual!r}, expected {expected!r} (from pyproject.toml)"
            )

    cm_path = ROOT / "codemeta.json"
    if cm_path.exists():
        cm = _read_json(cm_path)
        cm_license = cm.get("license", "")
        expected_url = f"https://spdx.org/licenses/{expected}.html"
        # codemeta normally stores license as SPDX URL; accept either form.
        if cm_license not in (expected, expected_url):
            findings.append(
                f"codemeta.json: license = {cm_license!r}, "
                f"expected {expected_url!r} or {expected!r} (from pyproject.toml)"
            )


def check_keywords(pyproject: dict, findings: list[str]) -> None:
    """Pyproject keywords must be a SUBSET of codemeta + zenodo keywords.

    Codemeta and Zenodo can have richer taxonomies (research keywords beyond
    PyPI search). The minimal package keywords from pyproject must always be
    present there.
    """
    pp_keywords = set(pyproject.get("project", {}).get("keywords") or [])
    if not pp_keywords:
        return

    cm_path = ROOT / "codemeta.json"
    if cm_path.exists():
        cm_keywords = set(_read_json(cm_path).get("keywords") or [])
        missing = pp_keywords - cm_keywords
        if missing:
            findings.append(
                f"codemeta.json: keywords missing {sorted(missing)!r} "
                f"(present in pyproject.toml [project].keywords)"
            )

    zen_path = ROOT / ".zenodo.json"
    if zen_path.exists():
        zen_keywords = set(_read_json(zen_path).get("keywords") or [])
        missing = pp_keywords - zen_keywords
        if missing:
            findings.append(
                f".zenodo.json: keywords missing {sorted(missing)!r} "
                f"(present in pyproject.toml [project].keywords)"
            )


def main() -> int:
    try:
        pyproject = _read_pyproject()
    except Exception as exc:
        print(f"ERROR: cannot parse pyproject.toml: {exc}", file=sys.stderr)
        return 2

    findings: list[str] = []
    check_author(pyproject, findings)
    check_license(pyproject, findings)
    check_keywords(pyproject, findings)

    if not findings:
        print("OK: metadata SSOT (author, license, keywords) consistent.")
        return 0

    print(f"\nFAIL: {len(findings)} metadata drift(s) detected:", file=sys.stderr)
    for f in findings:
        print(f"  • {f}", file=sys.stderr)
    print(
        "\nFix: update the drifted file(s) to match pyproject.toml, "
        "OR change pyproject.toml [project] and re-run.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
