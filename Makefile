.PHONY: install lint format test typecheck check

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

test:
	pytest

typecheck:
	mypy src/emotional_memory/

check: lint typecheck test
