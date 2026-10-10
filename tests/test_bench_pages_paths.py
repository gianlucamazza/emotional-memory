"""Guard dev/bench path agreement across workflows and the seed step."""

from __future__ import annotations

import subprocess
from pathlib import Path

import recover_bench_history
import seed_bench_data_branch

ROOT = Path(__file__).resolve().parents[1]
WRONG_PREFIX = "d" + "e" + "s" + "t" + "/"
BENCH_FILES = (
    Path(".github/workflows/ci.yml"),
    Path(".github/workflows/docs.yml"),
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
    assert f"--output site/{bench_dir}" in docs


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
