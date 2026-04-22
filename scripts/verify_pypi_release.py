"""Poll PyPI until the expected emotional-memory release becomes visible."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

PYPI_JSON_URL = "https://pypi.org/pypi/emotional-memory/json"
PYPI_RELEASE_URL = "https://pypi.org/project/emotional-memory/{version}/"


def _fetch_release_payload() -> dict[str, Any]:
    with urllib.request.urlopen(PYPI_JSON_URL, timeout=20) as response:  # noqa: S310
        payload = json.load(response)
    return dict(payload)


def _release_is_visible(payload: dict[str, Any], version: str) -> bool:
    releases = payload.get("releases", {})
    if not isinstance(releases, dict):
        return False
    files = releases.get(version)
    return isinstance(files, list) and bool(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Expected version, for example 0.6.3")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Maximum wait time before failing",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=10,
        help="Polling interval between PyPI checks",
    )
    args = parser.parse_args(argv)

    deadline = time.monotonic() + args.timeout_seconds
    last_seen = "<unavailable>"

    while time.monotonic() < deadline:
        try:
            payload = _fetch_release_payload()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_seen = f"error: {exc}"
        else:
            latest = payload.get("info", {}).get("version", "<unknown>")
            last_seen = f"latest {latest}"
            if _release_is_visible(payload, args.version):
                print(f"PyPI release visible: {PYPI_RELEASE_URL.format(version=args.version)}")
                return 0
        time.sleep(args.interval_seconds)

    print(
        (f"PyPI release verification failed: expected {args.version}, last seen {last_seen}"),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
