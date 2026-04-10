"""Smoke tests for the examples/ scripts."""

import runpy
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def test_basic_usage_runs() -> None:
    """basic_usage.py must execute without errors end-to-end."""
    runpy.run_path(str(EXAMPLES_DIR / "basic_usage.py"), run_name="__main__")
