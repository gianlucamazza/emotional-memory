"""Trigger a Software Heritage save request for the emotional-memory repository.

Usage::

    uv run python scripts/swh_client.py [--repo-url URL]

Software Heritage archives git repositories permanently, providing SWH-IDs
that are stable long-term identifiers independent of GitHub or Zenodo.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWH_API = "https://archive.softwareheritage.org/api/1"


def trigger_save(repo_url: str) -> dict[str, object]:
    url = f"{SWH_API}/origin/save/git/url/{repo_url.rstrip('/')}/"
    req = urllib.request.Request(  # noqa: S310
        url,
        method="POST",
        headers={"Accept": "application/json", "User-Agent": "emotional-memory-release/1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return json.loads(resp.read())  # type: ignore[return-value]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"SWH API {exc.code}: {body}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-url",
        help="Git repository URL (default: repo_url from release.toml)",
    )
    args = parser.parse_args(argv)

    repo_url = args.repo_url
    if not repo_url:
        data = tomllib.loads((ROOT / "release.toml").read_text(encoding="utf-8"))["release"]
        repo_url = data["repo_url"]

    print(f"[SWH] Triggering save for {repo_url} …")
    try:
        result = trigger_save(repo_url)
    except RuntimeError as exc:
        print(f"[WARN] SWH trigger failed (non-fatal): {exc}", file=sys.stderr)
        return 1

    status = result.get("save_task_status", "unknown")
    request_date = result.get("save_request_date", "")
    visit_type = result.get("visit_type", "git")
    print(f"[OK] SWH save request: status={status}, type={visit_type}, date={request_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
