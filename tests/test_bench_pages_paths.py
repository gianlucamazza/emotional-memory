"""Guard dev/bench path agreement across workflows and the seed step."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import recover_bench_history
import seed_bench_data_branch

ROOT = Path(__file__).resolve().parents[1]
WRONG_PREFIX = "d" + "e" + "s" + "t" + "/"
BENCH_FILES = (
    Path(".github/workflows/ci.yml"),
    Path(".github/workflows/docs.yml"),
    Path(".github/workflows/docs-dispatch.yml"),
    Path("scripts/recover_bench_history.py"),
    Path("scripts/seed_bench_data_branch.py"),
    Path("CHANGELOG.md"),
    Path("CONTRIBUTING.md"),
    Path("docs/benchmarks.md"),
)


def _scalar(text: str, key: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip().strip("'\"")
    raise AssertionError(f"missing {key!r}")


def _docs_output_dir(text: str) -> str:
    marker = "--output "
    for line in text.splitlines():
        if marker in line:
            value = line.split(marker, 1)[1].strip()
            if value.startswith("site/"):
                return value.removeprefix("site/")
    raise AssertionError("docs.yml has no --output site/…")


def test_workflow_bench_paths_agree() -> None:
    bench_dir = recover_bench_history.BENCH_DIR
    assert bench_dir == "dev/bench"
    assert Path(recover_bench_history.DATA_JS_PATH).parent.as_posix() == bench_dir
    assert bench_dir == seed_bench_data_branch.BENCH_DIR

    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    docs = (ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    assert _scalar(ci, "benchmark-data-dir-path") == bench_dir
    assert _scalar(ci, "gh-pages-branch") == "bench-data"
    assert _scalar(ci, "auto-push") == "true"
    assert _docs_output_dir(docs) == bench_dir
    assert "scripts/seed_bench_data_branch.py" in ci
    assert "scripts/recover_bench_history.py" in docs
    assert recover_bench_history.HISTORY_SHA in ci
    assert "--existing-ref origin/bench-data" in ci
    assert f"--output site/{bench_dir}" in docs


def test_docs_deploys_after_successful_ci_on_main() -> None:
    docs = (ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8")
    dispatch = (ROOT / ".github/workflows/docs-dispatch.yml").read_text(encoding="utf-8")
    header, _, jobs = docs.partition("\njobs:")
    on_block = header.split("on:", 1)[1].split("concurrency:", 1)[0]
    build = jobs.split("name: Deploy GitHub Pages", 1)[0]
    deploy = jobs.split("name: Deploy GitHub Pages", 1)[1]
    dispatch_on = dispatch.split("on:", 1)[1].split("concurrency:", 1)[0]

    assert "workflow_run:" in on_block
    assert '"CI"' in on_block
    assert "completed" in on_block
    assert 'branches: ["main"]' in on_block
    # Dispatch is a sibling workflow so this file's cache scope stays
    # workflow_run-only (read-only default-branch cache; no CodeQL hit).
    assert "workflow_dispatch:" not in on_block
    assert "workflow_dispatch:" in dispatch_on
    # CI already runs on every main push (no path filter), including docs-only.
    # A parallel push trigger redeploys stale bench-data before Seed/Store.
    assert "push:" not in on_block
    assert "push:" not in dispatch_on

    assert "github.event.workflow_run.head_sha" in docs
    assert "github.event.workflow_run.conclusion == 'success'" in build
    assert "github.event.workflow_run.conclusion == 'success'" in deploy
    assert "conclusion == 'failure'" not in docs
    assert "conclusion == 'cancelled'" not in docs

    assert "permissions: {}" in header
    assert "pages: write" not in header
    assert "id-token: write" not in header
    assert "contents: read" in build
    assert "pages: write" not in build
    assert "id-token: write" not in build
    assert "pages: write" in deploy
    assert "id-token: write" in deploy

    assert _scalar(header, "group") == "pages"
    assert _scalar(header, "cancel-in-progress") == "false"
    assert _scalar(dispatch.split("jobs:")[0], "group") == "pages"
    assert _scalar(docs, "enable-cache") == "false"
    assert _scalar(dispatch, "enable-cache") == "false"


def test_seed_step_auth_is_idempotent_and_not_bearer() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    seed_idx = ci.index("Seed bench-data from gh-pages history")
    store_idx = ci.index("Store benchmark results")
    extra_idx = ci.index("extra-tests:")
    seed = ci[seed_idx:store_idx]
    store = ci[store_idx:extra_idx]
    job = ci[ci.index("benchmarks:") : extra_idx]

    assert "contents: write" in job
    assert "github.event_name == 'push'" in job
    assert "github.ref == 'refs/heads/main'" in job
    assert "AUTHORIZATION: bearer" not in seed
    assert "AUTHORIZATION: Bearer" not in seed
    assert "x-access-token:${GH_TOKEN}@github.com/${GH_REPO}.git" in seed
    assert "HEAD:refs/heads/bench-data" in seed
    assert "set -x" not in seed
    assert "dev/bench/data.js" in seed
    assert recover_bench_history.HISTORY_SHA in seed
    assert "--existing-ref origin/bench-data" in seed
    assert "exit 0" not in seed
    for line in seed.splitlines():
        if "HEAD:refs/heads/bench-data" in line:
            assert "--force" not in line
            assert " -f " not in line
    assert "github-token:" in store
    assert "if:" not in store
    assert _scalar(store, "gh-pages-branch") == "bench-data"
    assert _scalar(store, "auto-push") == "true"


def test_no_dest_slash_in_bench_paths() -> None:
    for rel in BENCH_FILES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert WRONG_PREFIX not in text, f"{rel} still contains {WRONG_PREFIX!r}"


def test_seed_step_commits_and_pushes_dev_bench(tmp_path: Path) -> None:
    recover = tmp_path / "recover"
    recover.mkdir()
    (recover / "data.js").write_text("window.BENCHMARK_DATA = {};\n", encoding="utf-8")
    (recover / "index.html").write_text("<html></html>\n", encoding="utf-8")

    work = tmp_path / "work"
    assert seed_bench_data_branch.main(["--recover-dir", str(recover), "--work", str(work)]) == 0

    listed = subprocess.check_output(
        ["git", "-C", work, "ls-tree", "-r", "--name-only", "HEAD"],
        text=True,
    ).splitlines()
    assert listed == [
        f"{recover_bench_history.BENCH_DIR}/data.js",
        f"{recover_bench_history.BENCH_DIR}/index.html",
    ]
    assert WRONG_PREFIX + "bench" not in "\n".join(listed)

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", work, "push", str(remote), "HEAD:refs/heads/bench-data"],
        check=True,
        capture_output=True,
        text=True,
    )
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-b", "bench-data", str(remote), str(clone)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (clone / "dev" / "bench" / "data.js").is_file()
    assert (clone / "dev" / "bench" / "index.html").is_file()
    assert not (clone / "dest").exists()
    seeded = (clone / "dev" / "bench" / "data.js").read_text(encoding="utf-8")
    assert not seeded.rstrip().endswith(";")
    assert json.loads(seeded[len(recover_bench_history.SCRIPT_PREFIX) :]) == {}


def _point(commit_id: str, date: int) -> dict[str, object]:
    return {
        "commit": {"id": commit_id, "message": commit_id},
        "date": date,
        "tool": "pytest",
        "benches": [{"name": "x", "value": 1.0, "unit": "iter/sec"}],
    }


def _write_dashboard(dir_path: Path, points: list[dict[str, object]], *, last_update: int) -> None:
    payload = {
        "lastUpdate": last_update,
        "repoUrl": "https://github.com/gianlucamazza/emotional-memory",
        "entries": {"Benchmark": points},
    }
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "data.js").write_text(
        recover_bench_history.format_data_js(payload),
        encoding="utf-8",
    )
    (dir_path / "index.html").write_text("<html></html>\n", encoding="utf-8")


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_merge_unions_existing_with_recovered_history(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init")
    _git(src, "config", "user.email", "ci@example.com")
    _git(src, "config", "user.name", "ci")
    wiped = [_point("new1", 300)]
    _write_dashboard(src / "dev" / "bench", wiped, last_update=300)
    _git(src, "checkout", "-b", "bench-data")
    _git(src, "add", "dev/bench")
    _git(src, "commit", "-m", "wiped to one entry")
    parent = _git(src, "rev-parse", "HEAD").stdout.strip()

    recover = tmp_path / "recover"
    recovered_points = [_point("old1", 100), _point("old2", 200), _point("new1", 300)]
    _write_dashboard(recover, recovered_points, last_update=200)
    # Legacy semicolon payload must still merge (73d1cfb format).
    recover_data = json.loads(
        (recover / "data.js").read_text(encoding="utf-8")[
            len(recover_bench_history.SCRIPT_PREFIX) :
        ]
    )
    (recover / "data.js").write_text(
        recover_bench_history.SCRIPT_PREFIX + json.dumps(recover_data, indent=2) + ";\n",
        encoding="utf-8",
    )

    work = tmp_path / "work"
    assert (
        seed_bench_data_branch.main(
            [
                "--recover-dir",
                str(recover),
                "--work",
                str(work),
                "--existing-ref",
                "bench-data",
                "--src-repo",
                str(src),
            ]
        )
        == 0
    )

    head = _git(work, "rev-parse", "HEAD").stdout.strip()
    parents = _git(work, "rev-parse", "HEAD^").stdout.strip()
    assert head != parent
    assert parents == parent
    log = _git(work, "log", "--oneline").stdout.splitlines()
    assert len(log) == 2

    raw = (work / "dev" / "bench" / "data.js").read_text(encoding="utf-8")
    assert not raw.rstrip().endswith(";")
    data = json.loads(raw[len(recover_bench_history.SCRIPT_PREFIX) :])
    ids = [entry["commit"]["id"] for entry in data["entries"]["Benchmark"]]
    assert ids == ["old1", "old2", "new1"]
    assert data["lastUpdate"] == 300


def test_merge_is_noop_when_existing_has_all_recovered(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init")
    _git(src, "config", "user.email", "ci@example.com")
    _git(src, "config", "user.name", "ci")
    points = [_point("old1", 100), _point("new1", 300)]
    _write_dashboard(src / "dev" / "bench", points, last_update=300)
    _git(src, "checkout", "-b", "bench-data")
    _git(src, "add", "dev/bench")
    _git(src, "commit", "-m", "complete history")
    parent = _git(src, "rev-parse", "HEAD").stdout.strip()

    recover = tmp_path / "recover"
    _write_dashboard(recover, [_point("old1", 100)], last_update=100)

    work = tmp_path / "work"
    assert (
        seed_bench_data_branch.main(
            [
                "--recover-dir",
                str(recover),
                "--work",
                str(work),
                "--existing-ref",
                "bench-data",
                "--src-repo",
                str(src),
            ]
        )
        == 0
    )
    assert _git(work, "rev-parse", "HEAD").stdout.strip() == parent
    raw = (work / "dev" / "bench" / "data.js").read_text(encoding="utf-8")
    data = json.loads(raw[len(recover_bench_history.SCRIPT_PREFIX) :])
    ids = [entry["commit"]["id"] for entry in data["entries"]["Benchmark"]]
    assert ids == ["old1", "new1"]
