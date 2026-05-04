"""Deposit emotional-memory release to Zenodo and mint a DOI.

Modes::

    # Phase 1 of release: create new draft version + prereserve DOI (no upload yet)
    uv run python scripts/zenodo_deposit.py --reserve-only

    # Phase 2 of release: upload files to reserved draft + publish
    uv run python scripts/zenodo_deposit.py --upload-from-state

    # Legacy: create fresh deposit + upload + publish in one shot
    uv run python scripts/zenodo_deposit.py [--draft-only]

    # Legacy: publish an existing draft by numeric ID
    uv run python scripts/zenodo_deposit.py --publish-id DEPOSIT_ID

Environment variables (loaded from .env automatically):
    ZENODO_TOKEN   — API token
    ZENODO_BASE    — override base URL (e.g. https://sandbox.zenodo.org for testing)

The --reserve-only / --upload-from-state split solves the DOI chicken-and-egg
problem: Phase 1 runs before committing (so the prereserved DOI can be embedded
in paper/main.tex and committed), Phase 2 runs after tagging (uploads the
tag-archived source + PDF and publishes the deposit).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
STATE_FILE = ROOT / ".release_state.json"


# ── Environment ──────────────────────────────────────────────────────────────


def _load_env() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv  # type: ignore[import-untyped]

            load_dotenv(env_file)
        except ImportError:
            pass


# ── State file ───────────────────────────────────────────────────────────────


def _load_state() -> dict[str, object]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))  # type: ignore[return-value]
    return {}


def _save_state(data: dict[str, object]) -> None:
    STATE_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"[STATE] saved to {STATE_FILE.relative_to(ROOT)}")


# ── Metadata helpers ──────────────────────────────────────────────────────────


def _load_zenodo_json() -> dict[str, object]:
    path = ROOT / ".zenodo.json"
    if not path.exists():
        print("[ERROR] .zenodo.json not found — run from repo root", file=sys.stderr)
        sys.exit(1)
    with path.open() as f:
        return json.load(f)  # type: ignore[no-any-return]


def _get_version() -> str:
    import importlib.metadata
    import re

    try:
        return importlib.metadata.version("emotional_memory")
    except importlib.metadata.PackageNotFoundError:
        pass
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else "0.0.0"


def _latest_record_id_from_toml() -> int:
    data = tomllib.loads((ROOT / "release.toml").read_text(encoding="utf-8"))["release"]
    version_doi: str = data["version_doi"]
    return int(version_doi.rsplit(".", 1)[-1])


# ── File upload helper ────────────────────────────────────────────────────────


def _build_source_archive(version: str) -> Path:
    tag = f"v{version}"
    fd, out_str = tempfile.mkstemp(suffix=f"_emotional_memory-{version}.tar.gz")
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
            f"[WARN] git archive {tag} failed — using HEAD: {result.stderr.strip()}",
            file=sys.stderr,
        )
        result = subprocess.run(
            ["git", "archive", f"--prefix={prefix}", "--format=tar.gz", "HEAD", "-o", str(out)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[ERROR] git archive HEAD: {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
    return out


def upload_files(
    headers: dict[str, str], bucket_url: str, version: str, *, verbose: bool = True
) -> None:
    """Upload source archive, paper PDF, and arXiv tarball to *bucket_url*."""

    def _put(name: str, path: Path) -> bool:
        with path.open("rb") as fh:
            r = requests.put(f"{bucket_url}/{name}", headers=headers, data=fh, timeout=180)
        if r.ok:
            if verbose:
                print(f"[OK] Uploaded {name} ({path.stat().st_size // 1024} KB)")
            return True
        print(f"[WARN] upload {name}: {r.status_code} {r.text}", file=sys.stderr)
        return False

    # Source archive (from git tag or HEAD)
    archive = _build_source_archive(version)
    try:
        _put(f"emotional_memory-{version}.tar.gz", archive)
    finally:
        archive.unlink(missing_ok=True)

    # Paper PDF
    pdf = ROOT / "paper" / "main.pdf"
    if pdf.exists():
        _put(f"emotional_memory_{version}_paper.pdf", pdf)
    else:
        print("[WARN] paper/main.pdf not found — skipping PDF upload")

    # arXiv submission tarball
    arxiv_tar = ROOT / "paper" / "arxiv-submission.tar.gz"
    if arxiv_tar.exists():
        _put(f"emotional_memory_{version}_arxiv.tar.gz", arxiv_tar)


# ── Result display ────────────────────────────────────────────────────────────


def _print_record_info(record: dict[str, object], base_url: str, dep_id: int) -> None:
    doi = str(record.get("doi", record.get("metadata", {}).get("doi", "")))
    concept_doi = str(record.get("conceptdoi", ""))
    record_url = str(record.get("links", {}).get("record_html", f"{base_url}/record/{dep_id}"))

    print()
    print(f"Version DOI:  {doi}")
    if concept_doi:
        print(f"Concept DOI:  {concept_doi}")
    print(f"Record URL:   {record_url}")
    print()

    if doi:
        badge_url = f"https://zenodo.org/badge/DOI/{doi}.svg"
        print(f"  [![DOI]({badge_url})](https://doi.org/{doi})")

    if doi:
        (ROOT / ".zenodo_doi").write_text(doi)
        print("\n[OK] DOI saved to .zenodo_doi")


# ── Core operations ───────────────────────────────────────────────────────────


def reserve_new_version(base_url: str, token: str, parent_record_id: int) -> None:
    """Create a new draft version from an existing published record, prereserve DOI.

    This is Phase 1 of the API-driven release flow.  Call BEFORE committing and
    tagging so the prereserved DOI can be embedded in paper/main.tex.

    Saves deposit_id, bucket_url, reserved_doi, concept_doi to STATE_FILE.
    """
    headers = {"Authorization": f"Bearer {token}"}
    version = _get_version()

    # 1. Create new version draft
    print(f"[Zenodo] Creating new version draft from record {parent_record_id} …")
    r = requests.post(
        f"{base_url}/api/deposit/depositions/{parent_record_id}/actions/newversion",
        headers=headers,
        timeout=30,
    )
    if not r.ok:
        print(f"[ERROR] newversion: {r.status_code} {r.text}", file=sys.stderr)
        sys.exit(1)

    parent = r.json()
    latest_draft_url: str = parent["links"]["latest_draft"]

    # 2. Fetch the new draft
    r2 = requests.get(latest_draft_url, headers=headers, timeout=30)
    if not r2.ok:
        print(f"[ERROR] get latest_draft: {r2.status_code} {r2.text}", file=sys.stderr)
        sys.exit(1)
    draft = r2.json()
    draft_id: int = draft["id"]
    bucket_url: str = draft["links"]["bucket"]
    concept_doi: str = draft.get("conceptdoi", "")
    print(f"[OK] New draft id={draft_id}")

    # 3. Delete files inherited from previous version
    rf = requests.get(
        f"{base_url}/api/deposit/depositions/{draft_id}/files", headers=headers, timeout=30
    )
    inherited = rf.json() if rf.ok else []
    if isinstance(inherited, list):
        for f in inherited:
            requests.delete(
                f"{base_url}/api/deposit/depositions/{draft_id}/files/{f['id']}",
                headers=headers,
                timeout=30,
            )
        if inherited:
            print(f"[OK] Cleared {len(inherited)} inherited file(s)")

    # 4. Update metadata: bump version + prereserve DOI
    meta = _load_zenodo_json()
    meta["version"] = version
    meta["prereserve_doi"] = True  # type: ignore[assignment]
    pypi_url = f"https://pypi.org/project/emotional-memory/{version}/"
    identifiers: list[dict[str, str]] = meta.get("related_identifiers", [])  # type: ignore[assignment]
    if not any(i.get("identifier") == pypi_url for i in identifiers):
        identifiers.append({"identifier": pypi_url, "relation": "isIdenticalTo", "scheme": "url"})
    meta["related_identifiers"] = identifiers  # type: ignore[assignment]

    r3 = requests.put(
        f"{base_url}/api/deposit/depositions/{draft_id}",
        headers=headers,
        json={"metadata": meta},
        timeout=30,
    )
    if not r3.ok:
        print(f"[ERROR] update metadata: {r3.status_code} {r3.text}", file=sys.stderr)
        sys.exit(1)

    # 5. Read prereserved DOI from response
    draft_meta = r3.json().get("metadata", {})
    prereserve_info = draft_meta.get("prereserve_doi", {})
    reserved_doi: str = prereserve_info.get("doi", "") if isinstance(prereserve_info, dict) else ""
    if not reserved_doi:
        # Fallback: construct from record id (Zenodo convention)
        reserved_doi = f"10.5281/zenodo.{draft_id}"

    print(f"[OK] Pre-reserved DOI: {reserved_doi}")
    if concept_doi:
        print(f"     Concept DOI:      {concept_doi}")

    # 6. Save state for Phase 2
    state = _load_state()
    state.update(
        {
            "version": version,
            "deposit_id": draft_id,
            "bucket_url": bucket_url,
            "reserved_doi": reserved_doi,
            "concept_doi": concept_doi,
            "base_url": base_url,
            "zenodo_reserved": True,
            "zenodo_published": False,
        }
    )
    _save_state(state)

    print()
    print("Next steps:")
    print(f'  1. Update release.toml: version_doi = "{reserved_doi}"')
    print("  2. make sync-metadata && make paper")
    print("  3. git commit + git tag")
    print("  4. make zenodo-upload-publish")
    print("     (or: python scripts/zenodo_deposit.py --upload-from-state)")


def upload_and_publish_from_state(base_url: str, token: str) -> None:
    """Phase 2: upload files to reserved draft and publish. Reads STATE_FILE."""
    state = _load_state()
    if not state.get("zenodo_reserved"):
        print("[ERROR] No reserved deposit found. Run --reserve-only first.", file=sys.stderr)
        sys.exit(1)
    if state.get("zenodo_published"):
        print("[INFO] Deposit already published. Nothing to do.")
        return

    deposit_id: int = int(state["deposit_id"])  # type: ignore[arg-type]
    bucket_url: str = str(state["bucket_url"])
    version: str = str(state["version"])
    headers = {"Authorization": f"Bearer {token}"}

    print(f"[Zenodo] Uploading files to draft {deposit_id} …")
    upload_files(headers, bucket_url, version)

    print("[Zenodo] Publishing …")
    r = requests.post(
        f"{base_url}/api/deposit/depositions/{deposit_id}/actions/publish",
        headers=headers,
        timeout=30,
    )
    if not r.ok:
        print(f"[ERROR] publish: {r.status_code} {r.text}", file=sys.stderr)
        sys.exit(1)

    record = r.json()
    _print_record_info(record, base_url, deposit_id)

    state["zenodo_published"] = True
    _save_state(state)


def publish_existing(base_url: str, token: str, dep_id: int) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{base_url}/api/deposit/depositions/{dep_id}/actions/publish",
        headers=headers,
        timeout=30,
    )
    if not response.ok:
        print(f"[ERROR] publish: {response.status_code} {response.text}", file=sys.stderr)
        sys.exit(1)
    _print_record_info(response.json(), base_url, dep_id)


def deposit(base_url: str, token: str, draft_only: bool) -> None:
    """Legacy: create a fresh deposit (new concept DOI). Use for first-ever release only."""
    headers = {"Authorization": f"Bearer {token}"}
    meta = _load_zenodo_json()
    version = _get_version()
    meta["version"] = version
    pypi_url = f"https://pypi.org/project/emotional-memory/{version}/"
    identifiers: list[dict[str, str]] = meta.get("related_identifiers", [])  # type: ignore[assignment]
    if not any(r.get("identifier") == pypi_url for r in identifiers):
        identifiers.append({"identifier": pypi_url, "relation": "isIdenticalTo", "scheme": "url"})
    meta["related_identifiers"] = identifiers  # type: ignore[assignment]

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
    dep_id: int = dep["id"]
    bucket_url: str = dep["links"]["bucket"]
    print(f"[OK] Deposition created: id={dep_id}")

    upload_files(headers, bucket_url, version)

    if draft_only:
        print(f"[INFO] Draft saved (not published). Edit at {base_url}/deposit/{dep_id}")
        return

    r4 = requests.post(
        f"{base_url}/api/deposit/depositions/{dep_id}/actions/publish",
        headers=headers,
        timeout=30,
    )
    if not r4.ok:
        print(f"[ERROR] publish: {r4.status_code} {r4.text}", file=sys.stderr)
        sys.exit(1)
    _print_record_info(r4.json(), base_url, dep_id)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(
        description="Deposit emotional-memory to Zenodo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--reserve-only",
        action="store_true",
        help="Phase 1: create new version draft + prereserve DOI (no upload). "
        "Run BEFORE committing/tagging.",
    )
    mode.add_argument(
        "--upload-from-state",
        action="store_true",
        help="Phase 2: upload files + publish using deposit ID from .release_state.json. "
        "Run AFTER tagging.",
    )
    mode.add_argument(
        "--draft-only",
        action="store_true",
        help="Legacy: create fresh deposit + upload files but do not publish.",
    )
    mode.add_argument(
        "--publish-id",
        type=int,
        metavar="DEPOSIT_ID",
        help="Legacy: publish an existing Zenodo draft deposition by numeric ID.",
    )
    parser.add_argument(
        "--parent-record-id",
        type=int,
        help="Record ID of the last published version (for --reserve-only). "
        "Defaults to the numeric suffix of version_doi in release.toml.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ZENODO_BASE", "https://zenodo.org"),
        help="Zenodo base URL (default: https://zenodo.org; "
        "use https://sandbox.zenodo.org for testing).",
    )
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN", "")
    if not token:
        print(
            "[ERROR] ZENODO_TOKEN not set. Set it in .env or as an environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.reserve_only:
        record_id = args.parent_record_id or _latest_record_id_from_toml()
        reserve_new_version(args.base_url, token, record_id)
    elif args.upload_from_state:
        upload_and_publish_from_state(args.base_url, token)
    elif args.publish_id is not None:
        publish_existing(args.base_url, token, args.publish_id)
    else:
        deposit(args.base_url, token, args.draft_only)


if __name__ == "__main__":
    main()
