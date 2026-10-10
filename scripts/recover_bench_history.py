"""Reconstruct github-action-benchmark dest/bench history from gh-pages.

``mkdocs gh-deploy --force --clean`` replaces the ``gh-pages`` branch and
drops ``dev/bench/``. Each later ``benchmark-action`` commit then re-adds a
one-point ``data.js``. This walks those commits and merges unique entries by
source commit id so the dashboard is not empty after the split onto
``bench-data``.

Usage:
    python scripts/recover_bench_history.py --source origin/gh-pages --output /tmp/dev-bench
    python scripts/recover_bench_history.py --prefer origin/bench-data --source origin/gh-pages --output site/dev/bench
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

DATA_JS_PATH = "dev/bench/data.js"
INDEX_HTML_PATH = "dev/bench/index.html"
_DATA_JS_RE = re.compile(
    r"window\.BENCHMARK_DATA\s*=\s*(\{.*\})\s*;?\s*\Z",
    re.DOTALL,
)


def _run_git(*args: str, cwd: Path | None = None) -> str:
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
    return result.stdout


def _git_ok(*args: str, cwd: Path | None = None) -> bool:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def parse_data_js(raw: str) -> dict[str, Any]:
    match = _DATA_JS_RE.search(raw.strip())
    if match is None:
        raise ValueError("not a github-action-benchmark data.js payload")
    data = json.loads(match.group(1))
    if not isinstance(data, dict):
        raise ValueError("BENCHMARK_DATA is not an object")
    return data


def format_data_js(data: dict[str, Any]) -> str:
    body = json.dumps(data, indent=2, ensure_ascii=False)
    return f"window.BENCHMARK_DATA = {body};\n"


def _show_file(ref: str, path: str, cwd: Path) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def copy_prefer_ref(prefer: str, output: Path, cwd: Path) -> bool:
    data_js = _show_file(prefer, DATA_JS_PATH, cwd)
    index_html = _show_file(prefer, INDEX_HTML_PATH, cwd)
    if data_js is None:
        return False
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.js").write_text(data_js, encoding="utf-8")
    if index_html is not None:
        (output / "index.html").write_text(index_html, encoding="utf-8")
    print(f"copied {DATA_JS_PATH} from {prefer}", file=sys.stderr)
    return True


def _benchmark_commits(source: str, cwd: Path) -> list[str]:
    log = _run_git(
        "log",
        source,
        "--grep=add Benchmark",
        "--pretty=format:%H",
        cwd=cwd,
    )
    return [line.strip() for line in log.splitlines() if line.strip()]


def merge_history(source: str, cwd: Path) -> tuple[dict[str, Any], str | None]:
    commits = _benchmark_commits(source, cwd)
    if not commits:
        raise RuntimeError(f"no benchmark-action commits on {source}")

    entries_by_name: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    last_update = 0
    repo_url = "https://github.com/gianlucamazza/emotional-memory"
    index_html: str | None = None

    # Newest first so index.html comes from the latest dashboard template.
    for commit in commits:
        raw = _show_file(commit, DATA_JS_PATH, cwd)
        if raw is None:
            continue
        if index_html is None:
            index_html = _show_file(commit, INDEX_HTML_PATH, cwd)
        try:
            data = parse_data_js(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"skip {commit[:12]}: {exc}", file=sys.stderr)
            continue
        last_update = max(last_update, int(data.get("lastUpdate") or 0))
        repo_url = str(data.get("repoUrl") or repo_url)
        entries = data.get("entries")
        if not isinstance(entries, dict):
            continue
        for name, points in entries.items():
            if not isinstance(points, list):
                continue
            bucket = entries_by_name.setdefault(str(name), [])
            for point in points:
                if not isinstance(point, dict):
                    continue
                commit_meta = point.get("commit")
                commit_id = ""
                if isinstance(commit_meta, dict):
                    commit_id = str(commit_meta.get("id") or "")
                key = (str(name), commit_id or str(point.get("date") or ""))
                if key in seen:
                    continue
                seen.add(key)
                bucket.append(point)

    if not any(entries_by_name.values()):
        raise RuntimeError(f"no dest/bench entries recovered from {source}")

    for points in entries_by_name.values():
        points.sort(key=lambda item: int(item.get("date") or 0))

    merged: dict[str, Any] = {
        "lastUpdate": last_update,
        "repoUrl": repo_url,
        "entries": entries_by_name,
    }
    return merged, index_html


def write_dashboard(output: Path, data: dict[str, Any], index_html: str | None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "data.js").write_text(format_data_js(data), encoding="utf-8")
    if index_html is not None:
        (output / "index.html").write_text(index_html, encoding="utf-8")
    n_entries = sum(len(v) for v in data["entries"].values())
    print(
        f"wrote {output / 'data.js'} ({n_entries} merged entries)",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="origin/gh-pages",
        help="git ref whose history holds dest/bench snapshots (default: origin/gh-pages)",
    )
    parser.add_argument(
        "--prefer",
        default=None,
        help="if this ref already has dest/bench/data.js, copy it and skip the merge",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="directory to write data.js and index.html into",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="git repository root (default: cwd)",
    )
    args = parser.parse_args(argv)
    cwd = args.repo.resolve()

    if args.prefer and _git_ok("rev-parse", "--verify", args.prefer, cwd=cwd):
        if copy_prefer_ref(args.prefer, args.output, cwd):
            return 0
        print(
            f"{args.prefer} has no {DATA_JS_PATH}; falling back to {args.source}",
            file=sys.stderr,
        )

    if not _git_ok("rev-parse", "--verify", args.source, cwd=cwd):
        print(f"missing source ref {args.source}; skipping dest/bench", file=sys.stderr)
        return 0

    try:
        data, index_html = merge_history(args.source, cwd)
    except RuntimeError as exc:
        print(f"warning: {exc}; skipping dest/bench", file=sys.stderr)
        return 0
    write_dashboard(args.output, data, index_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
