"""Single-shot release orchestrator for emotional-memory.

Usage::

    uv run python scripts/release.py VERSION [--resume] [--skip-space] [--base-url URL]
    uv run python scripts/release.py VERSION --sandbox   # dry run against Zenodo sandbox

Phases (each is idempotent; --resume skips already-completed phases):
    0  preflight          — full gate check (clean tree, tests, build, twine)
    1  zenodo_reserve     — create new Zenodo draft version + prereserve DOI
    2  doi_sync           — write prereserved DOI to release.toml, sync-metadata, rebuild PDF
    3  commit_tag         — git commit "chore(release): vX.Y.Z" + annotated tag
    4  zenodo_publish     — upload source archive + PDF + arXiv tarball, publish deposit
    5  pypi               — uv publish dist/*
    6  github_release     — gh release create vX.Y.Z + upload assets
    7  hf_space           — deploy demo to Hugging Face Space
    8  swh                — trigger Software Heritage save
    9  report             — print summary table of all published URLs

State is persisted to .release_state.json (gitignored) so a failed run can be
resumed from the last successful phase without re-running irreversible operations.

IMPORTANT: disable the GitHub → Zenodo webhook before running (Settings →
Integrations on zenodo.org) to prevent a duplicate auto-deposit when the tag
is pushed.  The API-driven deposit (Phase 4) is the canonical one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / ".release_state.json"

OK = "\033[32m✔\033[0m"
FAIL = "\033[31m✘\033[0m"
INFO = "\033[34m→\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ── State helpers ─────────────────────────────────────────────────────────────


def _load_state() -> dict[str, object]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))  # type: ignore[return-value]
    return {}


def _save_state(state: dict[str, object]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _phase_done(state: dict[str, object], phase: str) -> bool:
    done: list[str] = state.get("phases_done", [])  # type: ignore[assignment]
    return phase in done


def _mark_done(state: dict[str, object], phase: str) -> None:
    done: list[str] = state.get("phases_done", [])  # type: ignore[assignment]
    if phase not in done:
        done.append(phase)
    state["phases_done"] = done
    _save_state(state)


# ── Subprocess helpers ────────────────────────────────────────────────────────


def _run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> int:
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, cwd=ROOT, env=merged_env)
    if check and result.returncode != 0:
        print(f"{FAIL} Command failed: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)
    return result.returncode


def _run_capture(cmd: list[str]) -> tuple[int, str, str]:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


# ── Changelog extraction ──────────────────────────────────────────────────────


def _extract_changelog_section(version: str) -> str:
    """Return the body of the ## [version] section from CHANGELOG.md."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    pattern = rf"(^## \[{re.escape(version)}\][^\n]*\n)(.*?)(?=^## |\Z)"
    m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not m:
        return f"Release v{version}"
    body = m.group(2).strip()
    return body if body else f"Release v{version}"


# ── Phase implementations ─────────────────────────────────────────────────────


def phase0_preflight(version: str, fast: bool = False) -> None:
    print(f"\n{BOLD}Phase 0 — Preflight{RESET}")
    args = ["uv", "run", "python", "scripts/preflight.py", version]
    if fast:
        args.append("--fast")
    _run(args)
    print(f"{OK} Preflight passed")


def phase1_zenodo_reserve(state: dict[str, object], base_url: str) -> None:
    print(f"\n{BOLD}Phase 1 — Zenodo DOI prereservation{RESET}")
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/zenodo_deposit.py",
            "--reserve-only",
            "--base-url",
            base_url,
        ]
    )
    # Read back state written by zenodo_deposit.py
    fresh = _load_state()
    state.update(fresh)
    reserved_doi = str(state.get("reserved_doi", ""))
    if not reserved_doi:
        print(f"{FAIL} No reserved_doi found in state file", file=sys.stderr)
        sys.exit(1)
    print(f"{OK} Reserved DOI: {reserved_doi}")


def phase2_doi_sync(state: dict[str, object], version: str) -> None:
    print(f"\n{BOLD}Phase 2 — DOI sync + paper rebuild{RESET}")
    reserved_doi = str(state.get("reserved_doi", ""))
    concept_doi = str(state.get("concept_doi", ""))

    # Update release.toml SSOT
    toml_path = ROOT / "release.toml"
    toml_text = toml_path.read_text(encoding="utf-8")
    toml_text = re.sub(
        r'^version_doi\s*=\s*"[^"]*"',
        f'version_doi = "{reserved_doi}"',
        toml_text,
        flags=re.MULTILINE,
    )
    toml_path.write_text(toml_text, encoding="utf-8")
    print(f"{OK} release.toml version_doi = {reserved_doi}")

    # Propagate to all files
    _run(["uv", "run", "python", "scripts/sync_release_metadata.py", "--from-toml"])
    print(f"{OK} Metadata synced")

    # Validate
    _run(["uv", "run", "python", "scripts/check_release_metadata.py"])
    print(f"{OK} Metadata validated")

    # Rebuild paper PDF with correct DOI embedded
    paper_dir = ROOT / "paper"
    rc = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=paper_dir,
        capture_output=True,
    ).returncode
    if rc != 0:
        print(f"{FAIL} Paper PDF build failed — check paper/main.log", file=sys.stderr)
        sys.exit(rc)
    print(f"{OK} Paper PDF rebuilt with DOI {reserved_doi}")

    state["concept_doi_confirmed"] = concept_doi
    _save_state(state)


def phase3_commit_tag(state: dict[str, object], version: str) -> None:
    print(f"\n{BOLD}Phase 3 — Commit + tag{RESET}")
    tag = f"v{version}"

    # Stage all SSOT-managed files
    managed = [
        "release.toml",
        "CITATION.cff",
        ".zenodo.json",
        "codemeta.json",
        "paper/main.tex",
        "paper/SUBMISSION.md",
        "demo/README.md",
        "demo/requirements.txt",
        "demo/app.py",
    ]
    existing = [f for f in managed if (ROOT / f).exists()]
    _run(["git", "add", *existing])

    rc, _, _ = _run_capture(["git", "diff", "--cached", "--quiet"])
    if rc == 0:
        print(f"{INFO} Nothing new to commit — tree already clean")
    else:
        msg = (
            f"chore(release): v{version}\n\n"
            f"Prereserved Zenodo DOI: {state.get('reserved_doi', '')}\n\n"
            "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
        )
        _run(["git", "commit", "-m", msg])
        print(f"{OK} Committed release metadata")

    # Check tag doesn't already exist
    _, out, _ = _run_capture(["git", "tag", "-l", tag])
    if out == tag:
        print(f"{INFO} Tag {tag} already exists — skipping")
    else:
        _run(["git", "tag", "-a", tag, "-m", f"Release {tag}"])
        print(f"{OK} Created tag {tag}")

    state["tag"] = tag
    _save_state(state)


def phase4_zenodo_publish(state: dict[str, object], base_url: str) -> None:
    print(f"\n{BOLD}Phase 4 — Zenodo upload + publish{RESET}")
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/zenodo_deposit.py",
            "--upload-from-state",
            "--base-url",
            base_url,
        ]
    )
    # Refresh state from file (zenodo_deposit.py updates it)
    state.update(_load_state())
    print(f"{OK} Zenodo published — DOI: {state.get('reserved_doi', '')}")


def phase5_pypi(state: dict[str, object], version: str) -> None:
    print(f"\n{BOLD}Phase 5 — PyPI publish{RESET}")
    # Build fresh dist
    _run(["uv", "build"])

    token = os.environ.get("PYPI_TOKEN", "")
    if token:
        _run(["uv", "publish", "--token", token])
    else:
        _run(["uv", "publish"])

    # Poll until visible
    _run(["uv", "run", "python", "scripts/verify_pypi_release.py", version], check=False)
    print(f"{OK} PyPI: emotional-memory=={version}")
    state["pypi_url"] = f"https://pypi.org/project/emotional-memory/{version}/"
    _save_state(state)


def phase6_github_release(state: dict[str, object], version: str) -> None:
    print(f"\n{BOLD}Phase 6 — GitHub Release{RESET}")
    tag = f"v{version}"
    notes = _extract_changelog_section(version)

    # Assets: wheel, sdist, paper PDF
    dist_files = list((ROOT / "dist").glob(f"*{version}*"))
    pdf = ROOT / "paper" / "main.pdf"
    assets = [str(f) for f in dist_files]
    if pdf.exists():
        assets.append(str(pdf))

    # Check if release already exists
    rc, _, _ = _run_capture(
        ["gh", "release", "view", tag, "--repo", "gianlucamazza/emotional-memory"]
    )
    if rc == 0:
        print(f"{INFO} GitHub Release {tag} already exists — uploading any missing assets")
        for asset in assets:
            _run(
                [
                    "gh",
                    "release",
                    "upload",
                    tag,
                    asset,
                    "--clobber",
                    "--repo",
                    "gianlucamazza/emotional-memory",
                ],
                check=False,
            )
    else:
        _run(
            [
                "gh",
                "release",
                "create",
                tag,
                "--title",
                f"v{version}",
                "--notes",
                notes,
                "--repo",
                "gianlucamazza/emotional-memory",
                *assets,
            ]
        )

    _, url, _ = _run_capture(
        [
            "gh",
            "release",
            "view",
            tag,
            "--repo",
            "gianlucamazza/emotional-memory",
            "--json",
            "url",
            "-q",
            ".url",
        ]
    )
    state["github_release_url"] = url
    _save_state(state)
    print(f"{OK} GitHub Release: {url}")


def phase7_hf_space(state: dict[str, object]) -> None:
    print(f"\n{BOLD}Phase 7 — Hugging Face Space deploy{RESET}")
    rc, _, _ = _run_capture(["git", "config", "--get", "remote.space.url"])
    if rc != 0:
        print(f"{INFO} Remote 'space' not configured — skipping HF deploy")
        state["hf_skipped"] = True
        _save_state(state)
        return
    _run(["make", "release-space"])
    print(f"{OK} HF Space deployed")
    state["hf_deployed"] = True
    _save_state(state)


def phase8_swh(state: dict[str, object]) -> None:
    print(f"\n{BOLD}Phase 8 — Software Heritage{RESET}")
    rc = _run(["uv", "run", "python", "scripts/swh_client.py"], check=False)
    state["swh_triggered"] = rc == 0
    _save_state(state)
    if rc == 0:
        print(f"{OK} Software Heritage save triggered")
    else:
        print(f"{INFO} SWH trigger failed (non-fatal — retry manually later)")


def phase9_report(state: dict[str, object], version: str) -> None:
    print(f"\n{BOLD}Phase 9 — Release summary{RESET}\n")
    doi = state.get("reserved_doi", "—")
    concept_doi = state.get("concept_doi", "—")
    pypi_url = state.get("pypi_url", f"https://pypi.org/project/emotional-memory/{version}/")
    gh_url = state.get("github_release_url", "—")

    rows = [
        ("PyPI", pypi_url),
        ("GitHub Release", gh_url),
        ("Zenodo version DOI", f"https://doi.org/{doi}"),
        ("Zenodo concept DOI", f"https://doi.org/{concept_doi}"),
        ("Software Heritage", "https://archive.softwareheritage.org (save queued)"),
    ]
    width = max(len(label) for label, _ in rows)
    for label, url in rows:
        print(f"  {label:<{width}}  {url}")

    print(f"\n{OK} v{version} fully released.")
    print("\nPost-release todo:")
    print(f"  git push origin main v{version}")
    print("  # After arXiv announcement: set arxiv_id in release.toml + make sync-metadata")


# ── Main ──────────────────────────────────────────────────────────────────────


ALL_PHASES = [
    "preflight",
    "zenodo_reserve",
    "doi_sync",
    "commit_tag",
    "zenodo_publish",
    "pypi",
    "github_release",
    "hf_space",
    "swh",
    "report",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("version", help="Version to release, e.g. 0.9.0")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip phases already marked done in .release_state.json",
    )
    parser.add_argument(
        "--skip-space",
        action="store_true",
        help="Skip Hugging Face Space deploy (phase 7)",
    )
    parser.add_argument(
        "--fast-preflight",
        action="store_true",
        help="Run preflight with --fast (skip slow build/install gates)",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use Zenodo sandbox (https://sandbox.zenodo.org) — for testing",
    )
    parser.add_argument(
        "--base-url",
        help="Zenodo base URL override",
    )
    args = parser.parse_args(argv)

    version = args.version
    base_url = args.base_url or (
        "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"
    )

    # Load env
    env_file = ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv  # type: ignore[import-untyped]

            load_dotenv(env_file)
        except ImportError:
            pass

    # Validate tokens early
    missing = [t for t in ("ZENODO_TOKEN", "PYPI_TOKEN") if not os.environ.get(t)]
    if missing:
        print(f"{FAIL} Missing required env vars: {', '.join(missing)}", file=sys.stderr)
        print("  Set them in .env or export before running.", file=sys.stderr)
        return 1

    state = _load_state()

    # Detect version mismatch in existing state
    if state.get("version") and state["version"] != version and not args.resume:
        print(
            f"{FAIL} .release_state.json is for v{state['version']}, not v{version}.\n"
            "  Delete .release_state.json to start fresh, or use --resume.",
            file=sys.stderr,
        )
        return 1

    state["version"] = version
    _save_state(state)

    resume = args.resume

    def _skip(phase: str) -> bool:
        if resume and _phase_done(state, phase):
            print(f"{INFO} Skipping {phase} (already done)")
            return True
        return False

    # ── Phase 0: Preflight ────────────────────────────────────────────────────
    if not _skip("preflight"):
        phase0_preflight(version, fast=args.fast_preflight)
        _mark_done(state, "preflight")

    # ── Phase 1: Zenodo reserve ───────────────────────────────────────────────
    if not _skip("zenodo_reserve"):
        phase1_zenodo_reserve(state, base_url)
        _mark_done(state, "zenodo_reserve")

    # ── Phase 2: DOI sync + paper rebuild ────────────────────────────────────
    if not _skip("doi_sync"):
        phase2_doi_sync(state, version)
        _mark_done(state, "doi_sync")

    # ── Phase 3: Commit + tag ─────────────────────────────────────────────────
    if not _skip("commit_tag"):
        phase3_commit_tag(state, version)
        _mark_done(state, "commit_tag")

    # ── Phase 4: Zenodo upload + publish ──────────────────────────────────────
    if not _skip("zenodo_publish"):
        phase4_zenodo_publish(state, base_url)
        _mark_done(state, "zenodo_publish")

    # ── Phase 5: PyPI ─────────────────────────────────────────────────────────
    if not _skip("pypi"):
        phase5_pypi(state, version)
        _mark_done(state, "pypi")

    # ── Phase 6: GitHub Release ───────────────────────────────────────────────
    if not _skip("github_release"):
        phase6_github_release(state, version)
        _mark_done(state, "github_release")

    # ── Phase 7: HF Space ────────────────────────────────────────────────────
    if not (args.skip_space or _skip("hf_space")):
        phase7_hf_space(state)
        _mark_done(state, "hf_space")
    elif args.skip_space:
        print(f"{INFO} HF Space deploy skipped (--skip-space)")

    # ── Phase 8: Software Heritage ────────────────────────────────────────────
    if not _skip("swh"):
        phase8_swh(state)
        _mark_done(state, "swh")

    # ── Phase 9: Report ───────────────────────────────────────────────────────
    phase9_report(state, version)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
