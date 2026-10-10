"""Commit recovered dev/bench onto an orphan bench-data worktree.

Used by the CI benchmarks job before ``github-action-benchmark`` appends a
new point. The dashboard directory is ``recover_bench_history.BENCH_DIR``
(``dev/bench``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from recover_bench_history import BENCH_DIR


def _git(cwd: Path, *args: str) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
        )


def seed_orphan_worktree(
    work: Path,
    recover_dir: Path,
    *,
    branch: str = "bench-data",
) -> None:
    """Populate ``work`` as an orphan ``branch`` commit containing dev/bench."""
    work = work.resolve()
    recover_dir = recover_dir.resolve()
    if not recover_dir.is_dir():
        raise RuntimeError(f"recover dir missing: {recover_dir}")
    dest = work / BENCH_DIR
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in recover_dir.iterdir():
        if src.is_file():
            shutil.copy2(src, dest / src.name)
            copied += 1
    if copied == 0:
        raise RuntimeError(f"recover dir is empty: {recover_dir}")

    _git(work, "init")
    _git(work, "checkout", "--orphan", branch)
    _git(work, "add", BENCH_DIR)
    _git(
        work,
        "-c",
        "user.name=github-actions[bot]",
        "-c",
        "user.email=github-actions[bot]@users.noreply.github.com",
        "commit",
        "-m",
        f"Seed {BENCH_DIR} from recovered gh-pages history",
    )
    print(f"seeded {BENCH_DIR} ({copied} files) on {branch}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-dir", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--branch", default="bench-data")
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    seed_orphan_worktree(args.work, args.recover_dir, branch=args.branch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
