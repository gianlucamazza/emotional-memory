"""Pre-release preflight check — run BEFORE any publish action.

Usage::

    uv run python scripts/preflight.py [VERSION]

Exits 0 if every gate passes; exits 1 with a readable report if any gate fails.

Gates checked:
    G1  pyproject.toml version matches VERSION (or is consistent across files)
    G2  CITATION.cff version == pyproject version
    G3  CHANGELOG.md has "## [VERSION] - YYYY-MM-DD" entry
    G4  git tag vVERSION does not already exist (local or remote)
    G5  working tree clean (no uncommitted changes)
    G6  on main branch, up-to-date with origin
    G7  no placeholder strings (XXXXXXX, TODO, FIXME-RELEASE) in README/CHANGELOG/CITATION
    G8  LICENSE present
    G9  CITATION.cff parseable as YAML
    G10 uv build succeeds (wheel + sdist)
    G11 twine check on built artefacts PASSED
    G12 smoke install into a fresh venv + import works
    G13 sdist does not contain secret-like patterns (.env, credentials)

Skip gates G10-G13 with --fast for a quick syntactic-only check.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

OK = "\033[32m✔\033[0m"
FAIL = "\033[31m✘\033[0m"
WARN = "\033[33m!\033[0m"


class Gate:
    def __init__(self, name: str, desc: str) -> None:
        self.name = name
        self.desc = desc
        self.passed: bool | None = None
        self.message = ""

    def ok(self, msg: str = "") -> None:
        self.passed = True
        self.message = msg

    def fail(self, msg: str) -> None:
        self.passed = False
        self.message = msg

    def render(self) -> str:
        icon = OK if self.passed else FAIL
        tail = f" — {self.message}" if self.message else ""
        return f"  {icon} {self.name}: {self.desc}{tail}"


def _read_pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else ""


def _read_citation_version() -> str:
    text = (ROOT / "CITATION.cff").read_text()
    m = re.search(r'^version:\s*"?([^"\n]+?)"?\s*$', text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def gate_version_consistency(version: str) -> Gate:
    g = Gate("G1", "pyproject.toml version")
    pv = _read_pyproject_version()
    if not pv:
        g.fail("version not found in pyproject.toml")
    elif version and pv != version:
        g.fail(f"pyproject={pv} but requested {version}")
    else:
        g.ok(pv)
    return g


def gate_citation_version() -> Gate:
    g = Gate("G2", "CITATION.cff version matches pyproject")
    pv = _read_pyproject_version()
    cv = _read_citation_version()
    if pv == cv:
        g.ok(cv)
    else:
        g.fail(f"pyproject={pv} CITATION={cv}")
    return g


def gate_changelog_entry(version: str) -> Gate:
    g = Gate("G3", f"CHANGELOG has entry for {version}")
    text = (ROOT / "CHANGELOG.md").read_text()
    pattern = rf"^## \[{re.escape(version)}\]\s*-\s*\d{{4}}-\d{{2}}-\d{{2}}"
    if re.search(pattern, text, re.MULTILINE):
        g.ok()
    else:
        g.fail(f"no '## [{version}] - YYYY-MM-DD' heading found")
    return g


def gate_tag_absent(version: str) -> Gate:
    g = Gate("G4", f"git tag v{version} does not exist")
    tag = f"v{version}"
    local = subprocess.run(
        ["git", "tag", "-l", tag], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if local.stdout.strip() == tag:
        g.fail(f"local tag {tag} exists")
        return g
    remote = subprocess.run(
        ["git", "ls-remote", "--tags", "origin", tag],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tag in remote.stdout:
        g.fail(f"remote tag {tag} exists on origin")
        return g
    g.ok()
    return g


def gate_clean_tree() -> Gate:
    g = Gate("G5", "working tree clean")
    r = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if r.stdout.strip():
        changed = [line.split(maxsplit=1)[-1] for line in r.stdout.strip().splitlines()][:5]
        g.fail(f"uncommitted changes: {', '.join(changed)}")
    else:
        g.ok()
    return g


def gate_on_main() -> Gate:
    g = Gate("G6", "on main, up-to-date with origin")
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if branch != "main":
        g.fail(f"on branch {branch}, not main")
        return g
    subprocess.run(
        ["git", "fetch", "--quiet", "origin", "main"], cwd=ROOT, capture_output=True, check=False
    )
    local = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    remote = subprocess.run(
        ["git", "rev-parse", "origin/main"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not remote:
        g.fail("cannot resolve origin/main")
    elif local != remote:
        ahead = subprocess.run(
            ["git", "rev-list", "--count", "origin/main..HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        behind = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        g.fail(f"ahead {ahead} / behind {behind} vs origin/main")
    else:
        g.ok()
    return g


def gate_no_placeholders() -> Gate:
    g = Gate("G7", "no placeholder strings in release-visible docs")
    patterns = [r"XXXXXXX", r"FIXME-RELEASE", r"TODO\(release\)"]
    offenders = [
        f"{fname}:{pat}"
        for fname in ("README.md", "CHANGELOG.md", "CITATION.cff")
        for pat in patterns
        if re.search(pat, (ROOT / fname).read_text())
    ]
    if offenders:
        g.fail(", ".join(offenders))
    else:
        g.ok()
    return g


def gate_license_present() -> Gate:
    g = Gate("G8", "LICENSE file present")
    if (ROOT / "LICENSE").exists():
        g.ok()
    else:
        g.fail("LICENSE missing")
    return g


def gate_citation_parseable() -> Gate:
    g = Gate("G9", "CITATION.cff parseable as YAML")
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        g.ok("skipped (pyyaml not installed)")
        return g
    try:
        data = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    except yaml.YAMLError as exc:
        g.fail(f"YAML error: {exc}")
        return g
    required = {"cff-version", "title", "authors", "version", "license"}
    missing = required - set(data)
    if missing:
        g.fail(f"missing keys: {missing}")
    else:
        g.ok()
    return g


def gate_build(dist_dir: Path) -> Gate:
    g = Gate("G10", "uv build succeeds")
    shutil.rmtree(dist_dir, ignore_errors=True)
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    r = subprocess.run(
        ["uv", "build", "--out-dir", str(dist_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-5:]
        g.fail("\n    " + "\n    ".join(tail))
    else:
        g.ok(str(dist_dir.relative_to(ROOT)))
    return g


def gate_twine_check(dist_dir: Path) -> Gate:
    g = Gate("G11", "twine check PASSED on dist/*")
    artefacts = list(dist_dir.glob("*.whl")) + list(dist_dir.glob("*.tar.gz"))
    if not artefacts:
        g.fail("no artefacts in dist/")
        return g
    r = subprocess.run(
        ["twine", "check", *map(str, artefacts)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        g.fail((r.stderr or r.stdout).strip().splitlines()[-3:])
    else:
        g.ok()
    return g


def gate_smoke_install(dist_dir: Path, version: str) -> Gate:
    g = Gate("G12", "fresh-venv install + import")
    whls = list(dist_dir.glob("*.whl"))
    if not whls:
        g.fail("no wheel to test")
        return g
    tmp = Path(tempfile.mkdtemp(prefix="em-preflight-"))
    try:
        r = subprocess.run(
            [sys.executable, "-m", "venv", str(tmp / "v")],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            g.fail("venv creation failed")
            return g
        pip = tmp / "v" / "bin" / "pip"
        py = tmp / "v" / "bin" / "python"
        r = subprocess.run(
            [str(pip), "install", "--quiet", str(whls[0])],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).splitlines()[-3:]
            g.fail("install: " + " ".join(tail))
            return g
        r = subprocess.run(
            [str(py), "-c", "from emotional_memory import EmotionalMemory; print('ok')"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0 or "ok" not in r.stdout:
            g.fail(f"import failed: {r.stderr.strip()}")
        else:
            g.ok()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return g


def gate_sdist_no_secrets(dist_dir: Path) -> Gate:
    g = Gate("G13", "sdist contains no secret-like files")
    sdists = list(dist_dir.glob("*.tar.gz"))
    if not sdists:
        g.fail("no sdist")
        return g
    import tarfile

    offenders: list[str] = []
    forbidden_names = {".env", "credentials.json", ".zenodo_doi"}
    forbidden_patterns = [re.compile(r"\.pem$"), re.compile(r"id_rsa")]
    with tarfile.open(sdists[0]) as tar:
        for member in tar.getmembers():
            leaf = Path(member.name).name
            if leaf in forbidden_names:
                offenders.append(member.name)
                continue
            if any(p.search(leaf) for p in forbidden_patterns):
                offenders.append(member.name)
    if offenders:
        g.fail(", ".join(offenders[:3]))
    else:
        g.ok()
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description="Release preflight checks.")
    parser.add_argument(
        "version",
        nargs="?",
        default="",
        help="Expected version (e.g. 0.5.2). If omitted, read from pyproject.toml",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow gates (build, twine, smoke install, sdist scan)",
    )
    args = parser.parse_args()

    version = args.version or _read_pyproject_version()
    if not version:
        print(f"{FAIL} Could not determine version.", file=sys.stderr)
        sys.exit(1)

    print(f"Preflight for emotional_memory v{version}")
    print(f"Root: {ROOT}")
    print()

    gates: list[Gate] = [
        gate_version_consistency(version),
        gate_citation_version(),
        gate_changelog_entry(version),
        gate_tag_absent(version),
        gate_clean_tree(),
        gate_on_main(),
        gate_no_placeholders(),
        gate_license_present(),
        gate_citation_parseable(),
    ]

    if not args.fast:
        dist_dir = ROOT / "dist"
        gates.append(gate_build(dist_dir))
        if gates[-1].passed:
            gates.append(gate_twine_check(dist_dir))
            gates.append(gate_smoke_install(dist_dir, version))
            gates.append(gate_sdist_no_secrets(dist_dir))

    failed = 0
    for g in gates:
        print(g.render())
        if g.passed is False:
            failed += 1

    print()
    if failed:
        print(f"{FAIL} {failed}/{len(gates)} gate(s) failed — release blocked.")
        sys.exit(1)
    else:
        print(f"{OK} All {len(gates)} gates passed — cleared for release.")


if __name__ == "__main__":
    main()
