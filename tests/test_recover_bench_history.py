"""Unit tests for scripts/recover_bench_history.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load() -> Any:
    path = ROOT / "scripts" / "recover_bench_history.py"
    spec = importlib.util.spec_from_file_location("recover_bench_history", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _commit_data(repo: Path, commit_id: str, date: int) -> None:
    dest = repo / "dev" / "bench"
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "lastUpdate": date,
        "repoUrl": "https://github.com/gianlucamazza/emotional-memory",
        "entries": {
            "Benchmark": [
                {
                    "commit": {"id": commit_id, "message": commit_id},
                    "date": date,
                    "tool": "pytest",
                    "benches": [{"name": "x", "value": 1.0, "unit": "iter/sec"}],
                }
            ]
        },
    }
    (dest / "data.js").write_text(
        "window.BENCHMARK_DATA = " + json.dumps(payload) + ";\n",
        encoding="utf-8",
    )
    (dest / "index.html").write_text("<html></html>\n", encoding="utf-8")
    _git(repo, "add", "dev/bench")
    _git(repo, "commit", "-m", f"add Benchmark (pytest) benchmark result for {commit_id}")


def test_parse_roundtrip() -> None:
    mod = _load()
    raw = 'window.BENCHMARK_DATA = {"lastUpdate": 1, "repoUrl": "u", "entries": {}};\n'
    data = mod.parse_data_js(raw)
    assert data["lastUpdate"] == 1
    assert mod.parse_data_js(mod.format_data_js(data))["repoUrl"] == "u"


def test_merge_unique_entries_across_wipes(tmp_path: Path) -> None:
    repo = tmp_path / "gh"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "ci")
    _commit_data(repo, "aaa111", 100)
    _git(repo, "rm", "-r", "dev")
    _git(repo, "commit", "-m", "Deployed wipe")
    _commit_data(repo, "bbb222", 200)

    out = tmp_path / "out"
    mod = _load()
    assert mod.main(["--source", "HEAD", "--output", str(out), "--repo", str(repo)]) == 0
    data = mod.parse_data_js((out / "data.js").read_text(encoding="utf-8"))
    ids = [entry["commit"]["id"] for entry in data["entries"]["Benchmark"]]
    assert ids == ["aaa111", "bbb222"]
    assert (out / "index.html").is_file()
