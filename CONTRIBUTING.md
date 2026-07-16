# Contributing to emotional-memory

Thank you for your interest. This guide covers everything from dev setup to getting a PR merged, including how to contribute new psychological benchmarks and research.

## Contents

- [Prerequisites](#prerequisites)
- [Development setup](#development-setup)
- [Local Secrets](#local-secrets)
- [Test suites](#test-suites)
- [Type checking](#type-checking)
- [Code style](#code-style)
- [Commit messages](#commit-messages)
- [SSOT policy](#ssot-policy)
- [Working with Claude Code](#working-with-claude-code)
- [Pull request process](#pull-request-process)
- [Maintainer Release](#maintainer-release)
- [Adding a fidelity benchmark](#adding-a-fidelity-benchmark)
- [Adding a store or embedder](#adding-a-store-or-embedder)
- [Contributing research](#contributing-research)

---

## SSOT policy

Several files in this repository carry metadata that is also duplicated in package indices, citation files, and the LaTeX paper (DOI, version, author, license, Python version floor, …). To prevent drift, the project maintains a **Single Source of Truth** discipline: there is one canonical source for each piece of metadata, and the rest are derived.

**Before editing any of the following files, read [`docs/contributing/ssot-policy.md`](docs/contributing/ssot-policy.md):**

- `CITATION.cff` — most fields are derived
- `.zenodo.json` — most fields are derived
- `codemeta.json` — most fields are derived
- `paper/main.tex` — DOI / URL / arxiv_id macros are derived
- `demo/app.py` — DOI / URL constants are derived
- `docs/index.md` — positioning hero is derived from `README.md`

CI runs `check_release_metadata.py`, `check_python_version_consistency.py`, and `check_metadata_ssot.py` on every PR. A failure of any of these is intentional, not a flaky test.

---

## Prerequisites

- Python 3.11–3.14
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Git

## Development setup

```bash
git clone https://github.com/gianlucamazza/emotional-memory
cd emotional-memory

# Canonical local setup
make install

# Optional local demo stack
make install-demo

# Maintainer / full release toolchain
make install-release

# Additional targeted extras
make install-sqlite
make install-docs

# Scored LLM benchmarks (Addenda X–Z, LoCoMo, A3, …)
cp .env.example .env        # fill EMOTIONAL_MEMORY_LLM_API_KEY
make install-scored-bench   # bench + llm-test + dotenv + sentence-transformers
make bench-deps-strict      # verify key + all bench deps before a scored run
```

Verify everything works:

```bash
make check   # lint + typecheck + tests — must pass before any commit
```

### Environment variables for LLM tests

The full `EMOTIONAL_MEMORY_LLM_*` surface is documented canonically in
[docs/contributing/llm-environment.md](docs/contributing/llm-environment.md) — set
`EMOTIONAL_MEMORY_LLM_API_KEY` at minimum. Copy `.env.example` to `.env` or export the
variables manually. Scored benchmark runners auto-load `.env` when `python-dotenv` is
installed (`make install-scored-bench` includes it).

Release secrets (not LLM configuration) used by maintainer targets:

| Variable              | Required | Default              | Purpose                                                                                     |
| --------------------- | -------- | -------------------- | ------------------------------------------------------------------------------------------- |
| `PYPI_TOKEN`          | No       | —                    | Manual PyPI fallback token for `make publish-pypi-manual`                                   |
| `ZENODO_TOKEN`        | No       | —                    | Zenodo API token for `make zenodo-draft` / `make zenodo-publish`                            |
| `ZENODO_BASE`         | No       | `https://zenodo.org` | Zenodo base URL; use sandbox for dry runs                                                   |
| `ORCID_CLIENT_ID`     | No       | —                    | ORCID Public API client id (public) — client-credentials, scope `/read-public`              |
| `ORCID_CLIENT_SECRET` | No       | —                    | ORCID Public API client secret (sensitive); register at <https://orcid.org/developer-tools> |

Real-LLM tests and benchmarks need the HTTP client — run `make install-llm-test` (installs
`httpx`). `make` targets export `.env` automatically; to have `.env` auto-loaded when invoking
a benchmark module directly (e.g. `python -m benchmarks.appraisal_diagnostics.runner`), also
run `make install-dotenv` (installs `python-dotenv`). Verify the resolved config any time with
`make llm-config`.

## Local Secrets

Use `.env` only for local CLI secrets that need to be read by tools in this repo.

- Good candidates for `.env`: `EMOTIONAL_MEMORY_LLM_*`, `ZENODO_TOKEN`, `ORCID_CLIENT_ID` / `ORCID_CLIENT_SECRET`, temporary `PYPI_TOKEN`
- `demo/app.py` does not call `load_dotenv()`; use `make demo-run` or export values explicitly
- Prefer shell-exported values for one-off publish commands so tokens do not linger on disk
- Never store credentials in git remotes; use a credential helper, OS keychain, or `hf auth login`
- The Hugging Face `space` remote should use a tokenless URL such as
  `https://huggingface.co/spaces/<user>/<space>` and rely on your credential manager

## Test suites

| Command                | Scope                                          | Speed                  |
| ---------------------- | ---------------------------------------------- | ---------------------- |
| `make test`            | Unit + integration (835+ tests)                | ~1s                    |
| `make cov`             | Same with branch coverage (≥ 80% enforced)     | ~2s                    |
| `make bench-fidelity`  | 127 parametrized psychological invariant tests | ~5s                    |
| `make bench-perf`      | Latency/throughput benchmarks                  | ~30s                   |
| `make test-llm`        | Real-LLM integration (requires API key)        | ~30s                   |
| `make bench-appraisal` | Scherer CPM prompt quality (requires API key)  | ~60s                   |
| `make demo-check`      | Demo wiring + runtime regression tests         | ~seconds to model-load |

Recommended local demo validation flow:

```bash
make llm-config-strict
make demo-check
make test-llm
```

Run a single test:

```bash
uv run python -m pytest tests/test_engine.py::test_encode_stores_memory -v
```

**SQLite and visualization tests** require their extras and run as separate CI jobs:

```bash
uv pip install -e ".[dev,sqlite]" && uv run python -m pytest tests/test_sqlite_store.py -v
uv pip install -e ".[dev,viz]"    && uv run python -m pytest tests/test_visualization.py -v
```

Coverage must stay above 80%. Check with:

```bash
make cov
```

## Type checking

All code must pass mypy in strict mode — no `Any`, full annotations everywhere:

```bash
make typecheck
```

Patterns used in this codebase:

```python
# Protocols for duck-typed interfaces (never inherit from them)
from typing import Protocol, runtime_checkable

class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...

# Pydantic frozen models for value objects
from pydantic import BaseModel

class CoreAffect(BaseModel, frozen=True):
    valence: float
    arousal: float

# __slots__ on all concrete non-Pydantic classes (memory efficiency + safety)
class MoodField:
    __slots__ = ("_pad", "_config")
```

## Code style

Formatting and linting use [ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`):

```bash
make format   # auto-format in place
make lint     # check only (CI mode)
```

Enabled rule groups: `E`, `F`, `I` (isort), `W`, `UP` (pyupgrade), `B` (bugbear), `SIM`, `RUF`, `C4`, `T20`, `PERF`, `S` (security), `PTH`.

**Comments**: write no comments by default. Add one only when the _why_ is non-obvious — a hidden constraint, a theory invariant, a workaround for a specific behavior. Never describe _what_ the code does.

**Theory references**: every formula, coefficient, or design decision that comes from a paper must cite the source inline: `# Bower 1981`, `# ACT-R power-law (Anderson 1983)`.

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <short imperative summary>

[optional body — explain why, not what]
```

Types: `fix`, `feat`, `docs`, `chore`, `refactor`, `test`, `perf`.

Breaking changes: `feat!:` and a `BREAKING CHANGE:` footer.

Examples:

```
fix: SQLiteStore cross-thread safety via threading.RLock
feat: SentenceTransformerEmbedder — embedders sub-package + [sentence-transformers] extra
docs: add 08_limitations.md and fix Anderson 1983 citation
test: add fidelity benchmark for PAD dominance (Mehrabian & Russell 1974)
```

Always update `CHANGELOG.md` under `## [Unreleased]`.

## Working with Claude Code

Contributions assisted by Claude Code (or other LLM assistants) should follow
the project's collaboration guide:
[`docs/contributing/claude-code-guide.md`](docs/contributing/claude-code-guide.md).
It captures the guiding principles (theory fidelity over raw performance), a base
system prompt, reusable prompts for common tasks (review, feature, refactor,
debug), and the pre-PR checklist. See also [`CLAUDE.md`](CLAUDE.md) for the
canonical command and architecture reference.

---

## Pull request process

1. Fork → branch: `git checkout -b feat/my-feature`
2. Make changes. `make check` must pass locally.
3. Update `CHANGELOG.md`.
4. Open a PR against `main` and fill in the template.
5. CI runs automatically. A maintainer will review within a few days.

**What makes a PR easy to merge:**

- Single logical change per PR
- Tests for every new behaviour
- Theory reference for changes to retrieval, decay, or resonance logic
- `make bench-fidelity` still passes

## Maintainer Release

Install the release toolchain first:

```bash
make install-release
```

Recommended release gate:

```bash
make release-check VERSION=X.Y.Z
```

That target runs:

- `make check`
- `make test-llm`
- `make bench-appraisal`
- `uv run python scripts/preflight.py X.Y.Z`

Publishing order:

```bash
git push origin main
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

Normal PyPI path:

- GitHub Actions workflow `Release to PyPI` triggers from the pushed tag
- The workflow now runs fast preflight, validates artefacts with `twine check`,
  uploads the built `dist/` files as workflow artefacts, and polls PyPI until the
  tagged version is visible

Manual PyPI fallback:

```bash
make publish-pypi-manual
make verify-pypi-release VERSION=X.Y.Z
```

Zenodo:

```bash
make zenodo-draft
# inspect the draft ID in output, then:
make zenodo-publish DEPOSIT_ID=123456
```

The Zenodo script prints both the version DOI and concept DOI. Use:

- concept DOI for stable badges and generic project links
- version DOI for release-specific citation blocks

After Zenodo publish, sync metadata from `.zenodo_doi`:

```bash
make sync-release-metadata
```

This updates the public DOI surfaces with the policy used in this repo:

- `README.md` badge -> concept DOI
- `demo/app.py` and `paper/main.tex` -> concept DOI
- `CITATION.cff` -> version DOI
- release-specific citation snippets -> version DOI

If you want to verify the local Zenodo sync against the gitignored `.zenodo_doi`
file as a maintainer-only check:

```bash
make meta-check-local
```

`make sync-release-metadata` also respects `ZENODO_BASE`, so sandbox deposits can
be synchronized without patching the script.

Hugging Face Space deployment:

```bash
make release-space
```

This pushes a `git subtree split --prefix=demo` snapshot to the configured `space` remote, which
keeps the Space repo isolated from the rest of the project tree.

## Adding a fidelity benchmark

Fidelity benchmarks in `benchmarks/fidelity/` test that the library implements psychological theories correctly. Each file covers one phenomenon and must:

1. Be named `test_<phenomenon>.py`
2. Mark every test function with `@pytest.mark.fidelity`
3. Cite the source paper in the module docstring
4. Use `HashEmbedder` (from `tests/conftest.py`) unless the test genuinely needs semantic similarity

Minimal template:

```python
"""Test: <Phenomenon> (<Author Year>).

<One-line description of the psychological invariant being validated>.
"""
import pytest
from emotional_memory import EmotionalMemory, InMemoryStore, CoreAffect

# HashEmbedder is imported from tests/conftest.py via pytest's conftest mechanism


@pytest.mark.fidelity
@pytest.mark.parametrize("valence,expected_rank", [
    (0.8, 0),   # high valence → target memory ranked first
    (-0.8, -1), # low valence → target memory ranked last
])
def test_phenomenon(em_factory, valence, expected_rank):
    em = EmotionalMemory(store=InMemoryStore(), embedder=...)
    # ... encode, retrieve, assert
    assert result[expected_rank].id == target_id, (
        f"<Phenomenon>: expected target at rank {expected_rank}, "
        f"got {[m.id for m in result]}"
    )
```

After adding the test, add it to the README fidelity table and run `make bench-fidelity`.

## Adding a store or embedder

**New MemoryStore**: implement `save`, `get`, `update`, `delete`, `list_all`, `search_by_embedding`, `__len__` (see `interfaces.py`). No inheritance required — `MemoryStore` is a `Protocol`. Place in `src/emotional_memory/stores/my_store.py`, add a guarded import in `stores/__init__.py` (pattern: `contextlib.suppress(ImportError)`), add an optional extra in `pyproject.toml`.

**New Embedder**: subclass `SequentialEmbedder` from `interfaces.py` and implement `embed(text) -> list[float]`. Override `embed_batch` for native batching. Place in `src/emotional_memory/embedders/my_embedder.py`.

Checklist for both:

- [ ] Thread-safety: writes must be serialised if the object is shared across threads
- [ ] `close()` method if the resource needs cleanup (engine calls it via duck-type)
- [ ] `__repr__` with meaningful content
- [ ] `__slots__` for memory efficiency
- [ ] Tests covering happy path, error paths, and thread-safety

## Contributing research

`docs/research/` contains the theoretical foundations of AFT. Contributions adding:

- A new psychological phenomenon with implementation and fidelity test
- An extension to the appraisal schema
- A correction or update to a theory reference

...are especially welcome. Please cite primary sources in both docstrings and `docs/research/06_bibliography.md` using the format already in use: `Author, Initial. (Year). *Title*. Publisher.`

New theoretical content should link to `docs/research/08_limitations.md` if it introduces assumptions that are contestable or culturally specific.

### Adding a pre-registered study (addendum lifecycle)

Every confirmatory study in this repo follows the same five-step lifecycle,
validated end to end by Addenda X and X2 (use their documents as templates). The
full prereg → closure → verdict index is
[`benchmarks/README.md`](benchmarks/README.md).

1. **Pre-registration first, in its own PR.** The prereg
   (`benchmarks/preregistration_addendum_<id>_*.md`) is committed and merged
   **before any scored or smoke run**. Follow the established skeleton
   (Status/Dataset header with sha256 + row counts, Motivation with a dated
   dataset-selection audit and any ex-ante priors, Protocol, Arms — exploratory
   arms marked "not in family, pre-declared droppable" —, Hypotheses +
   diagnostics, Statistical analysis plan, ex-ante Decision rule with explicit
   Branch A/PASS and Branch B/FAIL propagation, Scope, Execution).
2. **Harness in a separate PR, still pre-run.** Third-party datasets are
   vendored **byte-identical** under `benchmarks/datasets/<name>/` and pinned by
   sha256 in the loader (fail on mismatch), with a "Source & License" section in
   `benchmarks/datasets/README.md`. Add per-hook `exclude` entries in
   `.pre-commit-config.yaml` for the vendored file — the byte-mutating hooks
   (`end-of-file-fixer`, `trailing-whitespace`, `mixed-line-ending`) and
   `check-added-large-files` would otherwise silently break byte-identity. The
   runner must support `--dry-run` (no LLM key, small slice) and write
   `results.dry.*` (gitignored) so smoke runs can never clobber committed scored
   artifacts. Replicate upstream metric formulas verbatim (quirks included, unit
   tested against hand-computed examples) when comparability with published
   baselines matters.
3. **Pre-run amendments are legitimate only if labeled.** An "Amendment A<n>"
   section appended before any execution may correct descriptive or
   implementation details, and must end with the sentence "No hypothesis,
   metric, decision rule, N, or statistical plan changed" — and mean it.
4. **Scored run + closure + core propagation in one PR.** Commit
   `results.{json,md,protocol.json}`, update the prereg Status header, write the
   closure (Verdict with power/MDE, Diagnostics, clearly-labeled post-hoc, Bound
   update, decisions on exploratory arms, Follow-ups, Propagation list). Core
   propagation: `docs/research/08_limitations.md`,
   `docs/research/claim_validation_matrix.json` (the `allowed_public_wording`
   must be mirrored **verbatim** in `docs/research/09_current_evidence.md` — a
   test enforces this), CHANGELOG, ROADMAP, and the paper if affected.
5. **Residual-surface propagation in a final PR.** README, the
   `benchmarks/README.md` verdict table and research index ladder, the problem
   register, and any comparison/quality-bar documents.

Negative results are committed with the same care as positive ones — the
decision rule is fixed ex-ante and the result stands as measured.
