# Single Source of Truth (SSOT) Policy

This repository follows a **Single Source of Truth** discipline to prevent metadata drift across release-facing files (PyPI, citation files, Zenodo, codemeta, documentation, the LaTeX paper). This document describes which files are *sources* and which are *derived*, and how to make changes safely.

If you're a contributor opening a PR, the short version is: **never edit a derived file directly**. Edit the source, then run the appropriate sync or check script. CI will catch drift either way.

---

## 1. The SSOT axes

The project tracks seven categories of metadata that must stay consistent:

| Axis | Source of truth | Derived files | Sync mechanism |
|---|---|---|---|
| **Project version** | git tag (`vX.Y.Z`) | `pyproject.toml` `[project].version`, `CITATION.cff`, `.zenodo.json`, `codemeta.json` | `make bump VERSION=X.Y.Z` (`scripts/bump_version.py`) |
| **Release-facing identifiers** (DOIs, repo URL, arXiv ID) | `release.toml` | `README.md` badge, `CITATION.cff`, `.zenodo.json`, `codemeta.json`, `paper/main.tex`, `paper/SUBMISSION.md`, `demo/README.md`, `demo/app.py` | `make sync-metadata` (`scripts/sync_release_metadata.py`) |
| **Python version floor** | `pyproject.toml` `[project].requires-python` | `[tool.ruff].target-version`, `[tool.mypy].python_version`, `[tool.basedpyright].pythonVersion`, `[project].classifiers`, `.github/workflows/ci.yml` matrix | `scripts/check_python_version_consistency.py` (validates only — manual fix) |
| **Author / license / keywords** | `pyproject.toml` `[project]` | `CITATION.cff`, `codemeta.json`, `.zenodo.json` | `scripts/check_metadata_ssot.py` (validates only — manual fix) |
| **Positioning hero & comparison** (Why, table, 30-sec example) | `README.md` between `<!-- ssot:positioning-start -->` and `<!-- ssot:positioning-end -->` | `docs/index.md` | `mkdocs-include-markdown-plugin` (build-time inclusion) |
| **Getting started** (install + quickstart) | `README.md` between `<!-- ssot:getting-started-start -->` and `<!-- ssot:getting-started-end -->` | `docs/getting-started.md` | `mkdocs-include-markdown-plugin` (build-time inclusion) |
| **LLM environment variables** (`EMOTIONAL_MEMORY_LLM_*`) | `docs/contributing/llm-environment.md` | `README.md` mirror, `CLAUDE.md`, `CONTRIBUTING.md` | manual mirror (validates by review; see §5.4) |

---

## 2. Rules

1. **Never edit a derived file directly.** Find its source in the table above and edit there.
2. **If you find a discrepancy**, open an issue or fix the source — do not fix the derived file in isolation, since it will be overwritten on the next sync.
3. **New SSOT axes**: if you introduce a new piece of metadata that ends up in more than one file, add it to this document and either extend the appropriate sync script or add a check script.
4. **CI enforces drift**: the `meta-integrity` job in `.github/workflows/ci.yml` runs all the validation scripts above. A red CI on drift is intentional, not noise.

---

## 3. Practical workflows

### Bumping the version

```bash
make bump VERSION=0.11.0 DATE=2026-06-15
git add -A
git commit -m "chore(release): bump to v0.11.0"
git tag v0.11.0
git push --tags
```

The `bump_version.py` script propagates the version into `pyproject.toml`, `CITATION.cff`, `CHANGELOG.md`, `codemeta.json`, and `.zenodo.json`.

### After a new Zenodo deposit (per-release DOI changes)

```bash
# 1. Edit release.toml: update version_doi = "10.5281/zenodo.NNNNNNNN"
# 2. Propagate
make sync-metadata
git add -A
git commit -m "chore(release): sync metadata after Zenodo deposit"
```

### Changing the Python minimum version

```bash
# 1. Edit pyproject.toml: requires-python = ">=3.12"
# 2. See what else needs updating
uv run python scripts/check_python_version_consistency.py --fix-suggest

# 3. Apply suggestions manually (this is a deliberate, audited change):
#    - tool.ruff.target-version = "py312"
#    - tool.mypy.python_version = "3.12"
#    - tool.basedpyright.pythonVersion = "3.12"
#    - remove "Python :: 3.11" classifier
#    - drop "3.11" from .github/workflows/ci.yml matrix
#    - update CONTRIBUTING.md install instructions

# 4. Verify
uv run python scripts/check_python_version_consistency.py
```

### Changing author / license / keywords

```bash
# 1. Edit pyproject.toml [project]
# 2. Manually update CITATION.cff, codemeta.json, .zenodo.json
# 3. Verify
uv run python scripts/check_metadata_ssot.py
```

(A future iteration may extend `sync_release_metadata.py` to do this automatically. For now, validation only — these fields change rarely enough that the cost-benefit favours checking over auto-writing.)

### Updating the positioning hero (Why / comparison table / 30-sec example)

```bash
# 1. Edit README.md between the SSOT markers:
#    <!-- ssot:positioning-start -->
#    ... your changes ...
#    <!-- ssot:positioning-end -->
# 2. Verify locally
uv run mkdocs build --strict
# The docs site will pick up the change automatically on next deploy.
```

---

## 4. Files marked as "managed"

The following files contain machine-managed sections. Each carries a header comment naming the source script:

- `CITATION.cff` — version / DOI / date sections managed by `sync_release_metadata.py`
- `.zenodo.json` — version / publication_date managed by `sync_release_metadata.py`
- `codemeta.json` — managed by `sync_release_metadata.py`
- `paper/main.tex` — DOI / URL / arxiv_id macros tagged with `[ssot:*]` comments, managed by `sync_release_metadata.py`
- `demo/app.py` — constants tagged with `[ssot:*]` comments, managed by `sync_release_metadata.py`
- `docs/index.md` — positioning hero block included from `README.md` at build time
- `docs/getting-started.md` — install + quickstart block included from `README.md` at build time
- `README.md` / `CLAUDE.md` / `CONTRIBUTING.md` — `EMOTIONAL_MEMORY_LLM_*` tables are mirrors of `docs/contributing/llm-environment.md` (kept in sync manually; deliberate duplication per §5.4)

---

## 5. Adding a new SSOT axis

When you find yourself copying the same piece of metadata into a second file, stop and ask: should this be SSOT-managed?

Decision tree:

1. **Does the value change frequently** (per release, per quarter)? → yes: invest in a sync script.
2. **Does it change rarely but its drift would be embarrassing** (e.g. author affiliation, license)? → at minimum, add a validator to `check_metadata_ssot.py` style.
3. **Is it duplicated only because of MkDocs / PyPI / Zenodo schema constraints**? → use a build-time include where possible (MkDocs include-markdown, or a Jinja template).
4. **None of the above** → leave it duplicated; not every duplication is harmful. Document the deliberate choice in this file.

Always:
- Add the new axis to the table in §1 of this document.
- Add a CI step under `meta-integrity` so drift is caught early.
- Add an entry to §4 explaining what file is managed and by which script.
