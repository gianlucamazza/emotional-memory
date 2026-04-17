# Load .env if present (for LLM test vars)
-include .env
export

.PHONY: install install-sqlite install-sentence-transformers install-bench install-llm-test install-viz install-docs install-all lint format test cov typecheck check bench-perf bench-fidelity bench bench-appraisal bench-comparative reproduce-paper test-llm docs-images docs docs-serve dist publish clean help

install:
	uv pip install -e ".[dev]"

install-viz:
	uv pip install -e ".[dev,viz]"

install-docs:
	uv pip install -e ".[docs]"

install-bench:
	uv pip install -e ".[dev,bench]"

install-llm-test:
	uv pip install -e ".[dev,llm-test]"

install-dotenv:
	uv pip install -e ".[dev,dotenv]"

install-sqlite:
	uv pip install -e ".[dev,sqlite]"

install-sentence-transformers:
	uv pip install -e ".[dev,sentence-transformers]"

install-all:
	uv pip install -e ".[dev,viz,docs,bench,llm-test,dotenv,sqlite,sentence-transformers]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

test:
	pytest

cov:
	pytest --cov --cov-report=term-missing

typecheck:
	mypy src/emotional_memory/

check: lint typecheck test

bench-fidelity:
	pytest benchmarks/fidelity/ -v -m fidelity

bench-perf:
	pytest benchmarks/perf/ --benchmark-only --benchmark-sort=mean

bench: bench-fidelity bench-perf

bench-comparative:
	@echo "Comparative benchmark vs Mem0/Letta/Zep — coming in v0.6"
	@echo "Requires: pip install -e '.[dev,bench]' + external system installs"
	@echo "See benchmarks/comparative/ (not yet implemented)"

reproduce-paper:
	@echo "=== Reproducing paper tables ==="
	$(MAKE) bench-fidelity
	$(MAKE) bench-perf
	@echo ""
	@echo "Fidelity table: see benchmarks/fidelity/ test output above"
	@echo "Performance table: see benchmark-results.json"
	@echo "Comparative table: run 'make bench-comparative' (v0.6)"

bench-appraisal:
	pytest benchmarks/appraisal_quality/ -v -m appraisal_quality

test-llm:
	pytest tests/test_llm_integration.py -v -m llm

docs-images:
	uv run python scripts/generate_docs_images.py

docs:
	uv run mkdocs build --strict

docs-serve:
	uv run mkdocs serve

dist:
	uv build

publish: dist
	uv publish

clean:
	rm -rf dist/ build/ site/ htmlcov/ .coverage coverage.xml benchmark-results.json
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "emotional-memory — available make targets"
	@echo ""
	@echo "Setup:"
	@echo "  install                    Install package with dev dependencies"
	@echo "  install-sqlite             + sqlite-vec (SQLiteStore)"
	@echo "  install-sentence-transformers  + sentence-transformers (real embeddings)"
	@echo "  install-viz                + matplotlib (visualization)"
	@echo "  install-bench              + pytest-benchmark (performance benchmarks)"
	@echo "  install-llm-test           + httpx (real-LLM tests)"
	@echo "  install-docs               + mkdocs (documentation)"
	@echo "  install-all                All extras"
	@echo ""
	@echo "Quality:"
	@echo "  check                      lint + typecheck + test (run before commit)"
	@echo "  lint                       ruff check"
	@echo "  format                     ruff format (in-place)"
	@echo "  typecheck                  mypy strict"
	@echo "  test                       pytest (unit + integration)"
	@echo "  cov                        pytest with branch coverage"
	@echo ""
	@echo "Benchmarks:"
	@echo "  bench-fidelity             126 psychological invariant tests"
	@echo "  bench-perf                 latency / throughput benchmarks"
	@echo "  bench                      fidelity + performance"
	@echo "  bench-appraisal            LLM appraisal quality (requires API key)"
	@echo "  bench-comparative          Cross-system comparison (v0.6)"
	@echo "  reproduce-paper            Regenerate all paper tables"
	@echo ""
	@echo "LLM tests:"
	@echo "  test-llm                   Integration tests (requires EMOTIONAL_MEMORY_LLM_API_KEY)"
	@echo ""
	@echo "Docs:"
	@echo "  docs                       Build static site"
	@echo "  docs-serve                 Live-reload local server"
	@echo "  docs-images                Regenerate docs/images/ PNGs"
	@echo ""
	@echo "Release:"
	@echo "  dist                       Build wheel + sdist"
	@echo "  publish                    Build and publish to PyPI"
	@echo "  clean                      Remove build artefacts"
