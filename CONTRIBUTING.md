# Contributing to emotional_memory

## Setup

```bash
git clone https://github.com/gianlucamazza/emotional-memory
cd emotional-memory
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,bench,sqlite]"
# For real-LLM tests (optional):
pip install -e ".[dev,llm-test]"
# For visualization (optional):
pip install -e ".[dev,viz]"
# For documentation (optional):
pip install -e ".[docs]"
pre-commit install
```

## Development loop

```bash
make check      # lint + typecheck + test (full suite)
make cov        # tests with branch coverage report
make bench      # fidelity + performance benchmarks
make docs       # build static documentation site
make docs-serve # serve docs locally with live reload
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

Real-LLM tests (require `EMOTIONAL_MEMORY_LLM_API_KEY`):

```bash
make test-llm        # end-to-end integration tests with a real LLM
make bench-appraisal # Scherer CPM prompt quality benchmarks (15 phrases)
# Use any OpenAI-compatible endpoint:
EMOTIONAL_MEMORY_LLM_BASE_URL=http://localhost:11434/v1 \
EMOTIONAL_MEMORY_LLM_MODEL=llama3.2 \
EMOTIONAL_MEMORY_LLM_API_KEY=dummy \
make bench-appraisal
```

## Code style

- **Formatter/linter**: ruff (config in `ruff.toml`)
- **Types**: mypy strict — all public functions need type annotations
- **Docstrings**: required on all public classes, methods, and functions
- **Comments**: English only; only where logic is non-obvious

## Tests

| Location | Purpose | CI |
|---|---|---|
| `tests/` | Unit and integration tests | Every PR |
| `tests/test_llm_integration.py` | Real-LLM end-to-end tests (`pytest.mark.llm`) | Manual (API key required) |
| `benchmarks/fidelity/` | Psychological invariants (Bower, Yerkes-Dodson, etc.) | Push to main |
| `benchmarks/perf/` | Performance benchmarks | Push to main |
| `benchmarks/appraisal_quality/` | LLM appraisal prompt quality (`pytest.mark.appraisal_quality`) | Manual (API key required) |

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
| `interfaces.py` | `Embedder`, `MemoryStore` protocols + `SequentialEmbedder` base class |
| `interfaces_async.py` | `AsyncEmbedder`, `AsyncMemoryStore`, `AsyncAppraisalEngine` protocols |
| `stores/sqlite.py` | `SQLiteStore` — persistent store with sqlite-vec |
| `visualization.py` | 8 matplotlib plotting functions (optional `viz` extra) |
