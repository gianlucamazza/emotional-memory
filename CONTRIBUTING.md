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
- [Pull request process](#pull-request-process)
- [Maintainer Release](#maintainer-release)
- [Adding a fidelity benchmark](#adding-a-fidelity-benchmark)
- [Adding a store or embedder](#adding-a-store-or-embedder)
- [Contributing research](#contributing-research)

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
```

Verify everything works:

```bash
make check   # lint + typecheck + tests — must pass before any commit
```

### Environment variables for LLM tests

Copy `.env.example` (if present) or set these manually:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `EMOTIONAL_MEMORY_LLM_API_KEY` | Yes | — | API key for real-LLM tests |
| `EMOTIONAL_MEMORY_LLM_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `EMOTIONAL_MEMORY_LLM_MODEL` | No | `gpt-5-mini` | Model name |
| `EMOTIONAL_MEMORY_LLM_REASONING_EFFORT` | No | `""` | Reasoning budget for o-series / gpt-5 models (`minimal` / `low` / `medium` / `high`); omitted when empty |
| `EMOTIONAL_MEMORY_LLM_OUTPUT_MODE` | No | `plain` | Response mode: `plain` or `json_object` |
| `EMOTIONAL_MEMORY_LLM_TIMEOUT_SECONDS` | No | `30` | HTTP timeout in seconds |
| `PYPI_TOKEN` | No | — | Manual PyPI fallback token for `make publish-pypi-manual` |
| `ZENODO_TOKEN` | No | — | Zenodo API token for `make zenodo-draft` / `make zenodo-publish` |
| `ZENODO_BASE` | No | `https://zenodo.org` | Zenodo base URL; use sandbox for dry runs |

## Local Secrets

Use `.env` only for local CLI secrets that need to be read by tools in this repo.

- Good candidates for `.env`: `EMOTIONAL_MEMORY_LLM_*`, `ZENODO_TOKEN`, temporary `PYPI_TOKEN`
- `demo/app.py` does not call `load_dotenv()`; use `make demo-run` or export values explicitly
- Prefer shell-exported values for one-off publish commands so tokens do not linger on disk
- Never store credentials in git remotes; use a credential helper, OS keychain, or `hf auth login`
- The Hugging Face `space` remote should use a tokenless URL such as
  `https://huggingface.co/spaces/<user>/<space>` and rely on your credential manager

## Test suites

| Command | Scope | Speed |
|---|---|---|
| `make test` | Unit + integration (835+ tests) | ~1s |
| `make cov` | Same with branch coverage (≥ 80% enforced) | ~2s |
| `make bench-fidelity` | 126 parametrized psychological invariant tests | ~5s |
| `make bench-perf` | Latency/throughput benchmarks | ~30s |
| `make test-llm` | Real-LLM integration (requires API key) | ~30s |
| `make bench-appraisal` | Scherer CPM prompt quality (requires API key) | ~60s |
| `make demo-check` | Demo wiring + runtime regression tests | ~seconds to model-load |

Recommended local demo validation flow:

```bash
make llm-config-strict
make demo-check
make test-llm
```

Run a single test:

```bash
uv run pytest tests/test_engine.py::test_encode_stores_memory -v
```

**SQLite and visualization tests** require their extras and run as separate CI jobs:

```bash
uv pip install -e ".[dev,sqlite]" && uv run pytest tests/test_sqlite_store.py -v
uv pip install -e ".[dev,viz]"    && uv run pytest tests/test_visualization.py -v
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

**Comments**: write no comments by default. Add one only when the *why* is non-obvious — a hidden constraint, a theory invariant, a workaround for a specific behavior. Never describe *what* the code does.

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
