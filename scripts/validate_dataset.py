"""Validate an affect-labeled dataset against the AffectExample schema.

Usage::

    uv run python scripts/validate_dataset.py benchmarks/datasets/affect_reference_v1.jsonl

Exits 0 on success; exits 1 and prints errors if any example fails validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, ValidationError


class AffectExample(BaseModel):
    id: str
    text: Annotated[str, Field(min_length=1)]
    valence: Annotated[float, Field(ge=-1.0, le=1.0)]
    arousal: Annotated[float, Field(ge=0.0, le=1.0)]
    dominance: Annotated[float, Field(ge=-1.0, le=1.0)]
    expected_label: str
    source: str


def validate(path: Path) -> int:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    errors: list[str] = []
    for i, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
            AffectExample.model_validate(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {i}: JSON parse error — {exc}")
        except ValidationError as exc:
            for e in exc.errors():
                field = ".".join(str(x) for x in e["loc"])
                errors.append(f"line {i}: {field} — {e['msg']}")

    if errors:
        print(f"Validation FAILED for {path} ({len(errors)} error(s)):")
        for err in errors:
            print(f"  {err}")
        return 1

    print(f"OK — {len(lines)} examples valid in {path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an affect-labeled JSONL dataset.")
    parser.add_argument("dataset", help="Path to .jsonl file")
    args = parser.parse_args()
    sys.exit(validate(Path(args.dataset)))


if __name__ == "__main__":
    main()
