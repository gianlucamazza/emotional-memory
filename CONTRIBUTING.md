# Contributing to emotional_memory

## Setup

```bash
git clone https://github.com/gianlucamazza/emotional-memory
cd emotional-memory
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,bench,sqlite]"
pre-commit install
```

## Development loop

```bash
make check      # lint + typecheck + test (full suite)
make cov        # tests with branch coverage report
make bench      # fidelity + performance benchmarks
```

Individual commands:

```bash
ruff check .                          # lint
ruff format .                         # format
mypy src/emotional_memory/            # type check
pytest                                # unit + integration tests
pytest benchmarks/fidelity/ -v        # psychological fidelity tests
pytest benchmarks/perf/ --benchmark-only  # performance benchmarks
```

## Code style

- **Formatter/linter**: ruff (config in `ruff.toml`)
- **Types**: mypy strict — all public functions need type annotations
- **Docstrings**: required on all public classes, methods, and functions
- **Comments**: English only; only where logic is non-obvious

## Tests

| Location | Purpose |
|---|---|
| `tests/` | Unit and integration tests — run on every PR |
| `benchmarks/fidelity/` | Psychological invariants (Bower, Yerkes-Dodson, etc.) |
| `benchmarks/perf/` | Performance benchmarks — run on push to main |

Add tests for all new code. Coverage must stay above 80% (`fail_under = 80` in `pyproject.toml`).

## Pull requests

1. One logical change per PR
2. Add or update tests
3. Add an entry to `CHANGELOG.md` under `## [Unreleased]`
4. CI must pass: lint, typecheck, tests on Python 3.11-3.14

## Architecture overview

The library implements **Affective Field Theory (AFT)** — five layers of emotional representation:

| Layer | Class | Reference |
|---|---|---|
| 1 | `CoreAffect` | Russell 1980 circumplex |
| 2 | `AffectiveMomentum` | Spinoza — affect as transition |
| 3 | `StimmungField` | Heidegger §29 / PAD model |
| 4 | `AppraisalVector` | Scherer CPM / Lazarus |
| 5 | `ResonanceLink` | Aristotle / Bower spreading activation |

Entry point for encode/retrieve is `EmotionalMemory` in `src/emotional_memory/engine.py`.

### Additional Modules

| Module | Purpose |
|---|---|
| `appraisal_llm.py` | `LLMAppraisalEngine` + `KeywordAppraisalEngine` |
| `async_engine.py` | `AsyncEmotionalMemory` — async facade |
| `async_adapters.py` | `SyncToAsync*` bridge adapters + `as_async()` |
| `interfaces_async.py` | `AsyncEmbedder`, `AsyncMemoryStore`, `AsyncAppraisalEngine` protocols |
| `stores/sqlite.py` | `SQLiteStore` — persistent store with sqlite-vec |
