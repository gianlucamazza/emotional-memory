"""Commit recovered dev/bench onto bench-data, merging when the branch exists.

Used by the CI benchmarks job before ``github-action-benchmark`` appends a
new point. The dashboard directory is ``recover_bench_history.BENCH_DIR``
(``dev/bench``). When ``--existing-ref`` is set, recovered entries are
unioned into that history (never replaced, never force-pushed).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from recover_bench_history import (
    BENCH_DIR,
    DATA_JS_PATH,
    count_entries,
    format_data_js,
    has_all_recovered,
    load_data_js,
    union_entries,
)


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


def _commit(work: Path, message: str) -> None:
    _git(
        work,
        "-c",
        "user.name=github-actions[bot]",
        "-c",
        "user.email=github-actions[bot]@users.noreply.github.com",
        "commit",
        "-m",
        message,
    )


def _load_recovered(recover_dir: Path) -> dict[str, object]:
    recover_dir = recover_dir.resolve()
    if not recover_dir.is_dir():
        raise RuntimeError(f"recover dir missing: {recover_dir}")
    data_js = recover_dir / "data.js"
    if not data_js.is_file():
        raise RuntimeError(f"recover dir has no data.js: {recover_dir}")
    return load_data_js(data_js.read_text(encoding="utf-8"))


def seed_orphan_worktree(
    work: Path,
    recover_dir: Path,
    *,
    branch: str = "bench-data",
) -> None:
    """Populate ``work`` as an orphan ``branch`` commit containing dev/bench."""
    work = work.resolve()
    recover_dir = recover_dir.resolve()
    recovered = _load_recovered(recover_dir)
    dest = work / BENCH_DIR
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "data.js").write_text(format_data_js(recovered), encoding="utf-8")
    copied = 1
    index_html = recover_dir / "index.html"
    if index_html.is_file():
        shutil.copy2(index_html, dest / "index.html")
        copied += 1

    _git(work, "init")
    _git(work, "checkout", "--orphan", branch)
    _git(work, "add", BENCH_DIR)
    _commit(work, f"Seed {BENCH_DIR} from recovered gh-pages history")
    print(f"seeded {BENCH_DIR} ({copied} files) on {branch}", file=sys.stderr)


def merge_into_existing(
    work: Path,
    recover_dir: Path,
    *,
    existing_ref: str,
    src_repo: Path,
    branch: str = "bench-data",
) -> bool:
    """Union recovered entries onto ``existing_ref``. Return True if committed.

    No-op when the existing ``data.js`` already contains every recovered
    (name, commit id). Never force-updates the branch.
    """
    work = work.resolve()
    src_repo = src_repo.resolve()
    recovered = _load_recovered(recover_dir)
    work.mkdir(parents=True, exist_ok=True)
    _git(work, "init")
    _git(work, "fetch", "--no-tags", str(src_repo), f"+{existing_ref}:{branch}")
    _git(work, "checkout", branch)

    data_path = work / DATA_JS_PATH
    if data_path.is_file():
        try:
            existing = load_data_js(data_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            print(f"existing data.js unreadable ({exc}); treating as empty", file=sys.stderr)
            existing = {"lastUpdate": 0, "repoUrl": "", "entries": {}}
    else:
        existing = {"lastUpdate": 0, "repoUrl": "", "entries": {}}

    if has_all_recovered(existing, recovered):
        print("bench-data already has every recovered entry", file=sys.stderr)
        return False

    merged = union_entries(existing, recovered)
    dest = work / BENCH_DIR
    dest.mkdir(parents=True, exist_ok=True)
    data_path.write_text(format_data_js(merged), encoding="utf-8")
    index_src = recover_dir / "index.html"
    if index_src.is_file() and not (dest / "index.html").is_file():
        shutil.copy2(index_src, dest / "index.html")
    _git(work, "add", BENCH_DIR)
    _commit(work, f"Merge recovered {BENCH_DIR} history")
    print(
        f"merged {count_entries(merged)} entries onto {branch}",
        file=sys.stderr,
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-dir", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--branch", default="bench-data")
    parser.add_argument(
        "--existing-ref",
        default=None,
        help="if set, fetch this ref and union recovered entries (never replace)",
    )
    parser.add_argument(
        "--src-repo",
        type=Path,
        default=Path.cwd(),
        help="git repository that holds --existing-ref (default: cwd)",
    )
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=True)
    if args.existing_ref:
        merge_into_existing(
            args.work,
            args.recover_dir,
            existing_ref=args.existing_ref,
            src_repo=args.src_repo,
            branch=args.branch,
        )
    else:
        seed_orphan_worktree(args.work, args.recover_dir, branch=args.branch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
