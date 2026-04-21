# Load .env if present (for LLM test vars)
-include .env
export

.PHONY: install install-sqlite install-sentence-transformers install-langchain install-mem0 install-langmem install-bench install-llm-test install-viz install-docs install-all lint format test cov typecheck check bench-perf bench-fidelity bench bench-appraisal bench-comparative reproduce-paper paper test-llm llm-config llm-config-strict docs-images docs docs-serve dist publish clean help

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

install-langchain:
	uv pip install -e ".[dev,langchain]"

install-mem0:
	uv pip install -e ".[dev,mem0]"

install-langmem:
	uv pip install -e ".[dev,langmem]"

install-all:
	uv pip install -e ".[dev,viz,docs,bench,llm-test,dotenv,sqlite,sentence-transformers,langchain]"

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run pytest

cov:
	uv run pytest --cov --cov-report=term-missing

typecheck:
	uv run mypy src/emotional_memory/

check: lint typecheck test

bench-fidelity:
	uv run pytest benchmarks/fidelity/ -v -m fidelity

bench-perf:
	uv run pytest benchmarks/perf/ --benchmark-only --benchmark-sort=mean

bench: bench-fidelity bench-perf

bench-comparative:
	uv run python -m benchmarks.comparative.runner

reproduce-paper:
	uv run python scripts/reproduce_paper.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

paper-arxiv:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex
	mkdir -p paper/arxiv-build
	cp paper/main.tex paper/arxiv-build/
	cp paper/main.bbl paper/arxiv-build/
	cp paper/refs.bib paper/arxiv-build/
	cp -r paper/figures paper/arxiv-build/
	cp -r paper/tables paper/arxiv-build/
	tar -czf paper/arxiv-submission.tar.gz -C paper/arxiv-build .
	rm -rf paper/arxiv-build
	@echo "arXiv bundle ready: paper/arxiv-submission.tar.gz"
	@tar -tzf paper/arxiv-submission.tar.gz

bench-appraisal: llm-config-strict
	uv run pytest benchmarks/appraisal_quality/ -v -m appraisal_quality

test-llm: llm-config-strict
	uv run pytest tests/test_llm_integration.py -v -m llm

llm-config:
	uv run python scripts/check_llm_config.py

llm-config-strict:
	uv run python scripts/check_llm_config.py --strict --require-key

docs-images:
	uv run python scripts/generate_docs_images.py

docs:
	uv run mkdocs build --strict

docs-serve:
	uv run mkdocs serve

dist:
	uv build

preflight:
	uv run python scripts/preflight.py

preflight-fast:
	uv run python scripts/preflight.py --fast

publish: preflight dist
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
	@echo "  install-langchain              + langchain-core (LangChain adapter)"
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
	@echo "  llm-config                 Print resolved LLM config without secrets"
	@echo "  llm-config-strict          Fail fast on missing/unsupported LLM config"
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
