"""Print the current project version from pyproject.toml."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not version:
        print("ERROR: project.version missing from pyproject.toml", file=sys.stderr)
        return 1
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
