# Contributing to emotional-memory

Thank you for your interest. This guide covers everything from dev setup to getting a PR merged, including how to contribute new psychological benchmarks and research.

## Contents

- [Prerequisites](#prerequisites)
- [Development setup](#development-setup)
- [Test suites](#test-suites)
- [Type checking](#type-checking)
- [Code style](#code-style)
- [Commit messages](#commit-messages)
- [Pull request process](#pull-request-process)
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

# Core dev dependencies (tests, lint, typecheck)
uv pip install -e ".[dev]"

# Recommended: also install sqlite and real embeddings
uv pip install -e ".[dev,sqlite,sentence-transformers]"

# All extras
uv pip install -e ".[dev,sqlite,sentence-transformers,viz,llm-test,bench]"
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
| `EMOTIONAL_MEMORY_LLM_MODEL` | No | `gpt-4o-mini` | Model name |

## Test suites

| Command | Scope | Speed |
|---|---|---|
| `make test` | Unit + integration (513+ tests) | ~1s |
| `make cov` | Same with branch coverage (≥ 80% enforced) | ~2s |
| `make bench-fidelity` | 126 parametrized psychological invariant tests | ~5s |
| `make bench-perf` | Latency/throughput benchmarks | ~30s |
| `make test-llm` | Real-LLM integration (requires API key) | ~30s |
| `make bench-appraisal` | Scherer CPM prompt quality (requires API key) | ~60s |

Run a single test:

```bash
uv run pytest tests/test_engine.py::test_encode_stores_memory -v
```

**SQLite and visualization tests** require their extras and run as separate CI jobs:

```bash
uv pip install -e ".[dev,sqlite]" && pytest tests/test_sqlite_store.py -v
uv pip install -e ".[dev,viz]"    && pytest tests/test_visualization.py -v
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

Formatting and linting use [ruff](https://docs.astral.sh/ruff/) (config in `ruff.toml`):

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
