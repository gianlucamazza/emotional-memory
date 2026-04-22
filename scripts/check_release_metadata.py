"""Validate release-facing metadata consistency for the current repo state."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _pyproject_version() -> str:
    return str(_pyproject()["project"]["version"])


def _effective_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line and not line.startswith("#")]


def _check_current_version_heading(changelog: str, version: str) -> bool:
    pattern = rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}"
    return re.search(pattern, changelog, re.MULTILINE) is not None


def _extract(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        return None
    return str(match.group(1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-local-doi",
        action="store_true",
        help="Also require local .zenodo_doi to match the version DOI in CITATION.cff",
    )
    args = parser.parse_args(argv)

    version = _pyproject_version()
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    demo_readme = (ROOT / "demo" / "README.md").read_text(encoding="utf-8")
    demo_app = (ROOT / "demo" / "app.py").read_text(encoding="utf-8")
    paper_main = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    paper_submission = (ROOT / "paper" / "SUBMISSION.md").read_text(encoding="utf-8")

    errors: list[str] = []
    version_doi = _extract(r'^doi:\s*"([^"]+)"$', citation)
    concept_doi = _extract(r"\[!\[DOI\].*?\]\(https://doi\.org/([^)]+)\)", readme)

    if f'version: "{version}"' not in citation:
        errors.append("CITATION.cff version does not match pyproject.toml")

    demo_requirements = _effective_lines(ROOT / "demo" / "requirements.txt")
    expected_pin = f"emotional-memory=={version}"
    if not demo_requirements or demo_requirements[0] != expected_pin:
        errors.append("demo/requirements.txt does not pin the current package version")

    if "## [Unreleased]" not in changelog:
        errors.append("CHANGELOG.md is missing the Unreleased heading")

    if not _check_current_version_heading(changelog, version):
        errors.append("CHANGELOG.md is missing the current release heading")

    unreleased_link = f"[Unreleased]: https://github.com/gianlucamazza/emotional-memory/compare/v{version}...HEAD"
    if unreleased_link not in changelog:
        errors.append(
            "CHANGELOG.md Unreleased compare link is not anchored to the current version"
        )

    if "https://pypi.org/project/emotional-memory/" not in demo_readme:
        errors.append("demo/README.md is missing the PyPI project link")

    if "[![DOI](" not in readme:
        errors.append("README.md is missing the Zenodo DOI badge")

    if version_doi is None:
        errors.append("CITATION.cff is missing the version DOI")

    if concept_doi is None:
        errors.append("README.md DOI badge does not expose a concept DOI")

    if version_doi is not None:
        if version_doi not in demo_readme:
            errors.append("demo/README.md is not aligned to the CITATION version DOI")
        if version_doi not in paper_submission:
            errors.append("paper/SUBMISSION.md is not aligned to the CITATION version DOI")

    if concept_doi is not None:
        concept_url = f"https://doi.org/{concept_doi}"
        if concept_url not in demo_readme:
            errors.append("demo/README.md is not aligned to the README concept DOI")
        if concept_url not in demo_app:
            errors.append("demo/app.py is not aligned to the README concept DOI")
        if concept_url not in paper_main:
            errors.append("paper/main.tex is not aligned to the README concept DOI")

    if args.require_local_doi:
        local_doi = ROOT / ".zenodo_doi"
        if not local_doi.exists():
            errors.append(".zenodo_doi is required but missing")
        else:
            local_version_doi = local_doi.read_text(encoding="utf-8").strip()
            if version_doi is not None and local_version_doi != version_doi:
                errors.append("CITATION.cff does not match the local .zenodo_doi version DOI")

    if errors:
        print("release metadata check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"release metadata OK for {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
