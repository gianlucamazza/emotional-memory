.PHONY: install install-bench lint format test cov typecheck check bench-perf bench-fidelity bench

install:
	pip install -e ".[dev]"

install-bench:
	pip install -e ".[dev,bench]"

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
