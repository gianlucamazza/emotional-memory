"""Verify optional dependencies for scored LLM benchmarks.

Scored third-party benchmarks (Addenda X-Z, LoCoMo, A3, ...) need:

- ``httpx`` (``[llm-test]`` extra) — LLM HTTP client
- ``python-dotenv`` (``[dotenv]`` extra) — auto-load ``.env`` in runners
- ``sentence-transformers`` (``[sentence-transformers]`` extra) — SBERT/BGE embedders

Install everything with ``make install-scored-bench``, then verify with
``make bench-deps-strict``.
"""

from __future__ import annotations

import argparse
import importlib
import sys

_BENCH_MODULES: tuple[tuple[str, str], ...] = (
    ("httpx", "llm-test extra — make install-llm-test"),
    ("dotenv", "dotenv extra — make install-dotenv"),
    (
        "sentence_transformers",
        "sentence-transformers extra — make install-sentence-transformers",
    ),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when any required module is missing",
    )
    parser.add_argument(
        "--skip-embedder",
        action="store_true",
        help="do not require sentence-transformers (hash-embedder benchmarks only)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    missing: list[tuple[str, str]] = []
    for module, hint in _BENCH_MODULES:
        if args.skip_embedder and module == "sentence_transformers":
            continue
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append((module, hint))

    if not missing:
        print("bench_deps_ok: True")
        for module, _ in _BENCH_MODULES:
            if args.skip_embedder and module == "sentence_transformers":
                continue
            print(f"{module}: installed")
        return 0

    print("bench_deps_ok: False", file=sys.stderr)
    for module, hint in missing:
        print(f"missing: {module} ({hint})", file=sys.stderr)
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
