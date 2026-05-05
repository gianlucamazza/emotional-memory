"""Sync release-facing metadata from release.toml to all dependent files.

Preferred usage (offline, reads release.toml as SSOT):
    uv run python scripts/sync_release_metadata.py --from-toml [--dry-run]

Legacy usage (resolves concept DOI from Zenodo API):
    uv run python scripts/sync_release_metadata.py [--version-doi DOI] [--dry-run]

Managed files:
    README.md, CITATION.cff, .zenodo.json, codemeta.json, demo/README.md,
    demo/requirements.txt, demo/app.py, paper/main.tex, paper/SUBMISSION.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tomllib
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


def _load_from_toml() -> tuple[str, str, str, str]:
    """Return (concept_doi, version_doi, repo_url, arxiv_id) from release.toml."""
    data = tomllib.loads((ROOT / "release.toml").read_text(encoding="utf-8"))["release"]
    return data["concept_doi"], data["version_doi"], data["repo_url"], data.get("arxiv_id", "")


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


def _citation_date(citation_text: str) -> str:
    match = re.search(r'^date-released:\s*"([^"]+)"', citation_text, re.MULTILINE)
    if match is None:
        raise SystemExit("Could not extract date-released from CITATION.cff")
    return match.group(1)


def _replace(pattern: str, repl: str, text: str, description: str) -> str:
    updated, count = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Could not update {description}")
    return updated


def sync_release_metadata(
    version_doi: str,
    concept_doi: str,
    dry_run: bool,
    repo_url: str = "https://github.com/gianlucamazza/emotional-memory",
    arxiv_id: str = "",
) -> list[Path]:
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
        r'^version:\s*"[^"]+"$',
        f'version: "{version}"',
        citation_text,
        "CITATION version",
    )
    updated_citation = _replace(
        r'^doi:\s*"[^"]+"$',
        f'doi: "{version_doi}"',
        updated_citation,
        "CITATION DOI",
    )
    if updated_citation != citation_text:
        changes.append((citation, updated_citation))

    zenodo_json_path = ROOT / ".zenodo.json"
    zenodo_json_text = zenodo_json_path.read_text(encoding="utf-8")
    zenodo_data = json.loads(zenodo_json_text)
    zenodo_data["version"] = version
    zenodo_data["publication_date"] = _citation_date(updated_citation)
    updated_zenodo_json = json.dumps(zenodo_data, indent=2, ensure_ascii=False) + "\n"
    if updated_zenodo_json != zenodo_json_text:
        changes.append((zenodo_json_path, updated_zenodo_json))

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
    updated_demo = _replace(
        r"- \*\*PyPI\*\*: \[`emotional-memory [^`]+`\]\(https://pypi\.org/project/emotional-memory/[^/]+/\)",
        f"- **PyPI**: [`emotional-memory {version}`](https://pypi.org/project/emotional-memory/{version}/)",
        updated_demo,
        "demo README PyPI link",
    )
    updated_demo = _replace(
        r"^  version = \{[^}]+\},$",
        f"  version = {{{version}}},",
        updated_demo,
        "demo README BibTeX version",
    )
    if updated_demo != demo_text:
        changes.append((demo_readme, updated_demo))

    demo_app = ROOT / "demo" / "app.py"
    demo_app_text = demo_app.read_text(encoding="utf-8")
    updated_demo_app = _replace(
        r'(_ZENODO_CONCEPT_DOI\s*=\s*")[^"]*("  # \[ssot:concept_doi\])',
        rf"\g<1>{concept_doi}\g<2>",
        demo_app_text,
        "demo app _ZENODO_CONCEPT_DOI constant",
    )
    updated_demo_app = _replace(
        r'(_REPO_URL\s*=\s*")[^"]*("  # \[ssot:repo_url\])',
        rf"\g<1>{repo_url}\g<2>",
        updated_demo_app,
        "demo app _REPO_URL constant",
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
    updated_paper = _replace(
        r"^- \[ \] PyPI version pinned: `emotional-memory==[0-9][\w.\-+]*`$",
        f"- [ ] PyPI version pinned: `emotional-memory=={version}`",
        updated_paper,
        "paper submission PyPI version pin",
    )
    updated_paper = _replace(
        r"Software: emotional-memory v[0-9][\w.\-+]*",
        f"Software: emotional-memory v{version}",
        updated_paper,
        "paper submission Comments row version",
    )
    if updated_paper != paper_text:
        changes.append((paper_submission, updated_paper))

    demo_requirements = ROOT / "demo" / "requirements.txt"
    demo_req_text = demo_requirements.read_text(encoding="utf-8")
    updated_demo_req = _replace(
        r"^emotional-memory==[0-9][\w.\-+]*$",
        f"emotional-memory=={version}",
        demo_req_text,
        "demo requirements pin",
    )
    if updated_demo_req != demo_req_text:
        changes.append((demo_requirements, updated_demo_req))

    paper_main = ROOT / "paper" / "main.tex"
    paper_main_text = paper_main.read_text(encoding="utf-8")
    updated_paper_main = _replace(
        r"(\\newcommand\{\\zenodoconceptdoi\}\{)[^}]*(}% \[ssot:concept_doi\])",
        rf"\g<1>{concept_doi}\g<2>",
        paper_main_text,
        r"paper main \zenodoconceptdoi newcommand",
    )
    updated_paper_main = _replace(
        r"(\\newcommand\{\\zenodoversiondoi\}\{)[^}]*(}% \[ssot:version_doi\])",
        rf"\g<1>{version_doi}\g<2>",
        updated_paper_main,
        r"paper main \zenodoversiondoi newcommand",
    )
    updated_paper_main = _replace(
        r"(\\newcommand\{\\repourl\}\{)[^}]*(}% \[ssot:repo_url\])",
        rf"\g<1>{repo_url}\g<2>",
        updated_paper_main,
        r"paper main \repourl newcommand",
    )
    updated_paper_main = _replace(
        r"(\\newcommand\{\\arxivid\}\{)[^}]*(}% \[ssot:arxiv_id\])",
        rf"\g<1>{arxiv_id}\g<2>",
        updated_paper_main,
        r"paper main \arxivid newcommand",
    )
    if updated_paper_main != paper_main_text:
        changes.append((paper_main, updated_paper_main))

    # codemeta.json: sync version, doi, datePublished, identifier
    codemeta_path = ROOT / "codemeta.json"
    if codemeta_path.exists():
        codemeta_text = codemeta_path.read_text(encoding="utf-8")
        codemeta_data = json.loads(codemeta_text)
        codemeta_data["version"] = version
        codemeta_data["softwareVersion"] = version
        codemeta_data["identifier"] = f"https://doi.org/{concept_doi}"
        codemeta_data["datePublished"] = _citation_date(
            next((t for p, t in changes if p == ROOT / "CITATION.cff"), None)
            or (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        )
        pypi_url = f"https://pypi.org/project/emotional-memory/{version}/"
        codemeta_data["downloadUrl"] = pypi_url
        codemeta_data["installUrl"] = pypi_url
        updated_codemeta = json.dumps(codemeta_data, indent=2, ensure_ascii=False) + "\n"
        if updated_codemeta != codemeta_text:
            changes.append((codemeta_path, updated_codemeta))

    # arxiv_id → CITATION.cff identifiers + .zenodo.json related_identifiers
    if arxiv_id:
        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

        # CITATION.cff: add identifiers block if absent or update existing arxiv entry
        cff_path = ROOT / "CITATION.cff"
        changes_dict = dict(changes)
        cff_text = changes_dict.get(cff_path) or cff_path.read_text(encoding="utf-8")
        arxiv_entry_yaml = (
            f'  - type: url\n    value: "{arxiv_url}"\n    description: "arXiv preprint"'
        )
        if "identifiers:" not in cff_text:
            updated_cff = cff_text.rstrip() + f"\nidentifiers:\n{arxiv_entry_yaml}\n"
        else:
            updated_cff = re.sub(
                r'(  - type: url\n\s+value: "https://arxiv\.org/abs/)[^"]*(")',
                rf"\g<1>{arxiv_id}\g<2>",
                cff_text,
            )
            if updated_cff == cff_text and arxiv_url not in cff_text:
                updated_cff = re.sub(
                    r"(identifiers:)",
                    rf"\1\n{arxiv_entry_yaml}",
                    cff_text,
                    count=1,
                )
        if updated_cff != cff_text:
            changes_dict[cff_path] = updated_cff
            changes = list(changes_dict.items())

        # .zenodo.json: add/update arXiv related_identifier
        zj_path = ROOT / ".zenodo.json"
        changes_dict2 = dict(changes)
        zj_text = changes_dict2.get(zj_path) or zj_path.read_text(encoding="utf-8")
        zj_data = json.loads(zj_text)
        identifiers = zj_data.get("related_identifiers", [])
        arxiv_entry = {"identifier": arxiv_url, "relation": "isDescribedBy", "scheme": "url"}
        existing = [i for i in identifiers if "arxiv.org" in i.get("identifier", "")]
        if not existing:
            identifiers.append(arxiv_entry)
        else:
            for entry in existing:
                entry["identifier"] = arxiv_url
        zj_data["related_identifiers"] = identifiers
        updated_zj = json.dumps(zj_data, indent=2, ensure_ascii=False) + "\n"
        if updated_zj != zj_text:
            changes_dict2[zj_path] = updated_zj
            changes = list(changes_dict2.items())

    updated_paths: list[Path] = []
    for path, updated_text in changes:
        updated_paths.append(path)
        if not dry_run:
            path.write_text(updated_text, encoding="utf-8")

    return updated_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-toml",
        action="store_true",
        help="Read concept_doi, version_doi, repo_url from release.toml (offline; recommended)",
    )
    parser.add_argument(
        "--version-doi",
        help="Version DOI to sync; defaults to .zenodo_doi (ignored when --from-toml)",
    )
    parser.add_argument(
        "--concept-doi",
        help="Concept DOI; defaults to resolving from Zenodo API (ignored when --from-toml)",
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

    if args.from_toml:
        concept_doi, version_doi, repo_url, arxiv_id = _load_from_toml()
    else:
        version_doi = _load_version_doi(args.version_doi)
        concept_doi = args.concept_doi or _fetch_concept_doi(args.base_url, version_doi)
        repo_url = "https://github.com/gianlucamazza/emotional-memory"
        arxiv_id = ""

    changed = sync_release_metadata(
        version_doi, concept_doi, dry_run=args.dry_run, repo_url=repo_url, arxiv_id=arxiv_id
    )

    if changed:
        verb = "Would update" if args.dry_run else "Updated"
        for path in changed:
            print(f"{verb} {path.relative_to(ROOT)}")
    else:
        print("Release metadata already synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
