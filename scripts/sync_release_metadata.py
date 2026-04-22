"""Sync public Zenodo metadata files from a version DOI and concept DOI."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _project_version() -> str:
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise SystemExit("Could not determine project version from pyproject.toml")
    return match.group(1)


def _load_version_doi(explicit: str | None) -> str:
    if explicit:
        return explicit
    doi_file = ROOT / ".zenodo_doi"
    if doi_file.exists():
        return doi_file.read_text(encoding="utf-8").strip()
    raise SystemExit("Version DOI not provided and .zenodo_doi is missing")


def _record_id_from_doi(version_doi: str) -> str:
    return version_doi.rsplit(".", maxsplit=1)[-1]


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _fetch_concept_doi(base_url: str, version_doi: str) -> str:
    record_id = _record_id_from_doi(version_doi)
    with urllib.request.urlopen(  # noqa: S310
        f"{_normalize_base_url(base_url)}/api/records/{record_id}", timeout=20
    ) as response:
        payload = json.load(response)
    concept_doi = str(payload.get("conceptdoi", "")).strip()
    if not concept_doi:
        raise SystemExit(f"Zenodo record {record_id} does not expose a concept DOI")
    return concept_doi


def _replace(pattern: str, repl: str, text: str, description: str) -> str:
    updated, count = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Could not update {description}")
    return updated


def sync_release_metadata(version_doi: str, concept_doi: str, dry_run: bool) -> list[Path]:
    version = _project_version()
    changes: list[tuple[Path, str]] = []
    paper_doi_label = (
        f"{version_doi} (Zenodo concept record for emotional-memory)"
        if version_doi == concept_doi
        else f"{version_doi} (Zenodo software record for v{version})"
    )

    readme = ROOT / "README.md"
    readme_text = readme.read_text(encoding="utf-8")
    updated_readme = _replace(
        r"\[!\[DOI\]\(https://zenodo\.org/badge/DOI/[^\)]+\.svg\)\]\(https://doi\.org/[^\)]+\)",
        f"[![DOI](https://zenodo.org/badge/DOI/{concept_doi}.svg)](https://doi.org/{concept_doi})",
        readme_text,
        "README Zenodo badge",
    )
    if updated_readme != readme_text:
        changes.append((readme, updated_readme))

    citation = ROOT / "CITATION.cff"
    citation_text = citation.read_text(encoding="utf-8")
    updated_citation = _replace(
        r'^doi:\s*"[^"]+"$',
        f'doi: "{version_doi}"',
        citation_text,
        "CITATION DOI",
    )
    if updated_citation != citation_text:
        changes.append((citation, updated_citation))

    demo_readme = ROOT / "demo" / "README.md"
    demo_text = demo_readme.read_text(encoding="utf-8")
    updated_demo = _replace(
        r"- \*\*Zenodo Concept DOI\*\*: \[[^\]]+\]\(https://doi\.org/[^\)]+\)",
        f"- **Zenodo Concept DOI**: [{concept_doi}](https://doi.org/{concept_doi})",
        demo_text,
        "demo README concept DOI",
    )
    updated_demo = _replace(
        r"doi\s+= \{10\.5281/zenodo\.[0-9]+\},",
        f"doi     = {{{version_doi}}},",
        updated_demo,
        "demo README BibTeX DOI",
    )
    if updated_demo != demo_text:
        changes.append((demo_readme, updated_demo))

    demo_app = ROOT / "demo" / "app.py"
    demo_app_text = demo_app.read_text(encoding="utf-8")
    updated_demo_app = _replace(
        r"\[Zenodo DOI\]\(https://doi\.org/[^\)]+\)",
        f"[Zenodo DOI](https://doi.org/{concept_doi})",
        demo_app_text,
        "demo app concept DOI",
    )
    if updated_demo_app != demo_app_text:
        changes.append((demo_app, updated_demo_app))

    paper_submission = ROOT / "paper" / "SUBMISSION.md"
    paper_text = paper_submission.read_text(encoding="utf-8")
    updated_paper = _replace(
        r"- \[ \] DOI for software artifact correct: `10\.5281/zenodo\.[0-9]+`.*",
        f"- [ ] DOI for software artifact correct: `{version_doi}`",
        paper_text,
        "paper submission DOI checklist",
    )
    updated_paper = _replace(
        r"\| DOI \| 10\.5281/zenodo\.[0-9]+ .* \|",
        f"| DOI | {paper_doi_label} |",
        updated_paper,
        "paper submission DOI table row",
    )
    if updated_paper != paper_text:
        changes.append((paper_submission, updated_paper))

    paper_main = ROOT / "paper" / "main.tex"
    paper_main_text = paper_main.read_text(encoding="utf-8")
    updated_paper_main = _replace(
        r"\\url\{https://doi\.org/10\.5281/zenodo\.[0-9]+\}",
        f"\\\\url{{https://doi.org/{concept_doi}}}",
        paper_main_text,
        "paper main concept DOI",
    )
    if updated_paper_main != paper_main_text:
        changes.append((paper_main, updated_paper_main))

    updated_paths: list[Path] = []
    for path, updated_text in changes:
        updated_paths.append(path)
        if not dry_run:
            path.write_text(updated_text, encoding="utf-8")

    return updated_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version-doi",
        help="Version DOI to sync; defaults to .zenodo_doi",
    )
    parser.add_argument(
        "--concept-doi",
        help="Concept DOI to sync; defaults to resolving the Zenodo record for the version DOI",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ZENODO_BASE", "https://zenodo.org"),
        help="Zenodo base URL; use sandbox.zenodo.org for sandbox metadata sync",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the files that would change without writing them",
    )
    args = parser.parse_args()

    version_doi = _load_version_doi(args.version_doi)
    concept_doi = args.concept_doi or _fetch_concept_doi(args.base_url, version_doi)
    changed = sync_release_metadata(version_doi, concept_doi, dry_run=args.dry_run)

    if changed:
        verb = "Would update" if args.dry_run else "Updated"
        for path in changed:
            print(f"{verb} {path.relative_to(ROOT)}")
    else:
        print("Release metadata already synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
