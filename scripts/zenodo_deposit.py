"""Deposit emotional-memory release to Zenodo and mint a DOI.

Usage::

    uv run python scripts/zenodo_deposit.py [--draft-only] [--base-url URL]

Environment variables (loaded from .env automatically):
    ZENODO_TOKEN   — API token from https://zenodo.org/account/settings/applications/tokens/new/
    ZENODO_BASE    — override base URL (e.g. https://sandbox.zenodo.org for testing)

Outputs:
    DOI and badge markdown printed to stdout.
    DOI saved to .zenodo_doi (gitignored) for downstream use.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent


def _load_env() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv  # type: ignore[import-untyped]

            load_dotenv(env_file)
        except ImportError:
            pass


def _load_zenodo_json() -> dict:  # type: ignore[type-arg]
    path = ROOT / ".zenodo.json"
    if not path.exists():
        print("[ERROR] .zenodo.json not found — run from repo root", file=sys.stderr)
        sys.exit(1)
    with path.open() as f:
        return json.load(f)


def _get_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("emotional_memory")
    except importlib.metadata.PackageNotFoundError:
        pass
    # fallback: read pyproject.toml
    import re

    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if m:
        return m.group(1)
    return "0.0.0"


def _build_source_archive(version: str) -> Path:
    tag = f"v{version}"
    fd, out_str = tempfile.mkstemp(suffix=f"_emotional_memory-{version}.tar.gz")
    import os

    os.close(fd)
    out = Path(out_str)
    prefix = f"emotional_memory-{version}/"
    result = subprocess.run(
        ["git", "archive", f"--prefix={prefix}", "--format=tar.gz", tag, "-o", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"[WARN] git archive failed: {result.stderr.strip()} — using working tree",
            file=sys.stderr,
        )
        result = subprocess.run(
            ["git", "archive", f"--prefix={prefix}", "--format=tar.gz", "HEAD", "-o", str(out)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[ERROR] git archive: {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
    return out


def deposit(base_url: str, token: str, draft_only: bool) -> None:
    headers = {"Authorization": f"Bearer {token}"}

    meta = _load_zenodo_json()
    version = _get_version()
    meta["version"] = version

    # Add PyPI related identifier if not present
    pypi_url = f"https://pypi.org/project/emotional-memory/{version}/"
    identifiers = meta.get("related_identifiers", [])
    if not any(r.get("identifier") == pypi_url for r in identifiers):
        identifiers.append({"identifier": pypi_url, "relation": "isIdenticalTo", "scheme": "url"})
    meta["related_identifiers"] = identifiers

    # 1. Create deposition
    r = requests.post(
        f"{base_url}/api/deposit/depositions",
        headers=headers,
        json={"metadata": meta},
        timeout=30,
    )
    if not r.ok:
        print(f"[ERROR] create deposition: {r.status_code} {r.text}", file=sys.stderr)
        sys.exit(1)

    dep = r.json()
    dep_id = dep["id"]
    bucket_url = dep["links"]["bucket"]
    print(f"[OK] Deposition created: id={dep_id}")

    # 2. Upload source archive
    archive_path = _build_source_archive(version)
    archive_name = f"emotional_memory-{version}.tar.gz"
    with archive_path.open("rb") as fh:
        r2 = requests.put(
            f"{bucket_url}/{archive_name}",
            headers=headers,
            data=fh,
            timeout=120,
        )
    archive_path.unlink(missing_ok=True)
    if not r2.ok:
        print(f"[ERROR] upload source archive: {r2.status_code} {r2.text}", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] Uploaded {archive_name}")

    # 3. Upload paper PDF if available
    pdf = ROOT / "paper" / "main.pdf"
    if pdf.exists():
        with pdf.open("rb") as fh:
            r3 = requests.put(
                f"{bucket_url}/emotional_memory_{version}_paper.pdf",
                headers=headers,
                data=fh,
                timeout=120,
            )
        if not r3.ok:
            print(f"[WARN] upload PDF: {r3.status_code} {r3.text}", file=sys.stderr)
        else:
            print("[OK] Uploaded paper PDF")
    else:
        print("[WARN] paper/main.pdf not found — skipping PDF upload")

    if draft_only:
        print(f"[INFO] Draft saved (not published). Edit at {base_url}/deposit/{dep_id}")
        return

    # 4. Publish → mint DOI
    r4 = requests.post(
        f"{base_url}/api/deposit/depositions/{dep_id}/actions/publish",
        headers=headers,
        timeout=30,
    )
    if not r4.ok:
        print(f"[ERROR] publish: {r4.status_code} {r4.text}", file=sys.stderr)
        sys.exit(1)

    record = r4.json()
    doi = record.get("doi", record.get("metadata", {}).get("doi", ""))
    record_url = record.get("links", {}).get("record_html", f"{base_url}/record/{dep_id}")

    print()
    print(f"DOI:        {doi}")
    print(f"Record URL: {record_url}")
    print()
    badge_url = f"https://zenodo.org/badge/DOI/{doi}.svg"
    doi_url = f"https://doi.org/{doi}"
    print("README badge:")
    print(f"  [![DOI]({badge_url})]({doi_url})")

    doi_file = ROOT / ".zenodo_doi"
    doi_file.write_text(doi)
    print("\n[OK] DOI saved to .zenodo_doi")


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Deposit emotional-memory to Zenodo.")
    parser.add_argument(
        "--draft-only", action="store_true", help="Upload files but do not publish"
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ZENODO_BASE", "https://zenodo.org"),
        help=(
            "Zenodo base URL (default: https://zenodo.org;"
            " use https://sandbox.zenodo.org for testing)"
        ),
    )
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN", "")
    if not token:
        print(
            "[ERROR] ZENODO_TOKEN not set. Set it in .env or as an environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    deposit(args.base_url, token, args.draft_only)


if __name__ == "__main__":
    main()
