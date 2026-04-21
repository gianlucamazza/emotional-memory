"""Print a safe summary of the resolved LLM HTTP configuration."""

from __future__ import annotations

import argparse
import sys

from emotional_memory.llm_http import (
    DEFAULT_OPENAI_COMPAT_MODEL,
    OpenAICompatibleLLMConfig,
    project_config_issues,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when the resolved config is incompatible with project defaults",
    )
    parser.add_argument(
        "--require-key",
        action="store_true",
        help="exit non-zero when EMOTIONAL_MEMORY_LLM_API_KEY is missing",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    try:
        config = OpenAICompatibleLLMConfig.from_env()
    except ValueError as exc:
        print(f"config_error: {exc}", file=sys.stderr)
        return 1

    if config is None:
        print("api_key_set: False")
        print("base_url: <unset>")
        print("model: <unset>")
        print("output_mode: <unset>")
        print("timeout_seconds: <unset>")
        print(f"project_default_model: {DEFAULT_OPENAI_COMPAT_MODEL}")
        print("uses_project_default_model: False")
        print("project_compatible: False")
        if args.require_key:
            print("config_error: EMOTIONAL_MEMORY_LLM_API_KEY is required", file=sys.stderr)
            return 1
        return 0

    issues = project_config_issues(config)
    summary = {
        **config.public_summary(),
        "project_default_model": DEFAULT_OPENAI_COMPAT_MODEL,
        "uses_project_default_model": config.model == DEFAULT_OPENAI_COMPAT_MODEL,
        "project_compatible": not issues,
    }
    for key, value in summary.items():
        print(f"{key}: {value}")
    for issue in issues:
        print(f"issue: {issue}", file=sys.stderr)

    if args.strict and issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
