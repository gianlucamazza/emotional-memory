---
name: release-gate
description: Run the pre-publish validation gate for a new release. Use before `make publish` or Zenodo deposit. Takes a version argument, runs make release-check, validates CHANGELOG [Unreleased] is non-empty, verifies version strings align, and previews the Zenodo metadata sync.
disable-model-invocation: true
arguments: [version]
allowed-tools: Bash(make release-check*) Bash(make meta-check*) Bash(make sync-release-metadata*) Bash(grep *) Read
---

# Release Gate

Argument: `$version` (e.g. `0.7.0`).

**Never run `make publish` or `make zenodo-draft` before this skill exits PASS.**

Validates the full pre-publish state before touching PyPI or Zenodo.

## Steps

### 1. Full make release-check

```bash
make release-check VERSION=$version
```

This runs: lint + typecheck + test + test-llm + bench-appraisal + preflight.

On failure: stop. Report which gate failed and what the error message was.
Do not proceed until all gates pass.

### 2. Version string consistency

The canonical version source is `pyproject.toml` (`[project] version`).
`src/emotional_memory/__init__.py` resolves `__version__` dynamically via
`importlib.metadata` — do not grep it for the literal version string.

Run the canonical version scripts:

```bash
uv run python scripts/resolve_version.py
uv run python scripts/check_release_metadata.py
uv run python scripts/check_metadata_ssot.py
```

`resolve_version.py` must print exactly `$version`.
`check_release_metadata.py` validates pyproject.toml + CHANGELOG heading.
`check_metadata_ssot.py` validates CITATION.cff, codemeta.json, and
.zenodo.json against pyproject.toml as SSOT.

Any mismatch = FAIL. Fix with `make bump VERSION=$version` then re-run.

### 3. CHANGELOG [Unreleased] non-empty

Read `CHANGELOG.md`. The `## [Unreleased]` section must contain at least one
bullet under `### Added`, `### Changed`, or `### Fixed`.

Empty `[Unreleased]` = FAIL. The changelog entry must be written before release.

### 4. Release metadata consistency

```bash
make meta-check-local
```

Validates DOI, authors, and citation metadata are internally consistent.

### 5. Zenodo metadata preview (dry run)

```bash
make sync-release-metadata
```

Show what metadata would be deposited. Does not create or update any Zenodo
deposition — review the output before proceeding.

## Next steps (printed on PASS)

```
All gates passed. Ready to publish v$version.

  1. Build and publish to PyPI:
       make dist
       make publish              # or: make publish-pypi-manual

  2. Verify PyPI release (poll until visible):
       make verify-pypi-release VERSION=$version

  3. Create Zenodo draft and review:
       make zenodo-draft         # review in Zenodo UI before publishing

  4. Publish Zenodo deposition:
       make zenodo-publish DEPOSIT_ID=<id from zenodo-draft output>

  5. Push HF Space:
       make release-space

  6. Tag and push:
       git tag v$version
       git push origin v$version
```
