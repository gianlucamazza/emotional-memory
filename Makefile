# Load .env if present (for LLM test vars)
-include .env
export

.PHONY: install install-bench install-llm-test install-viz install-docs lint format test cov typecheck check bench-perf bench-fidelity bench bench-appraisal test-llm docs-images docs docs-serve dist publish

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
