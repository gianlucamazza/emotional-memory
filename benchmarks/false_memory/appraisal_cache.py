"""Durable, model-scoped cache for costly direct-VAD benchmark appraisals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from emotional_memory.appraisal import GenericAppraisalVector
from emotional_memory.appraisal_llm import LLMAppraisalEngine
from emotional_memory.appraisal_schema import DIRECT_VAD_SCHEMA


class CheckpointedAppraiser:
    """Append each successful appraisal before continuing the benchmark."""

    def __init__(self, engine: LLMAppraisalEngine, *, model: str, path: Path) -> None:
        self.engine = engine
        self.model = model
        self.path = path
        self.cache: dict[str, dict[str, float]] = {}
        if path.exists():
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    entry = json.loads(line)
                    self.cache[str(entry["key"])] = {
                        str(k): float(v) for k, v in entry["dimensions"].items()
                    }
                except (ValueError, KeyError, AttributeError, TypeError) as exc:
                    raise RuntimeError(f"invalid appraisal checkpoint line {line_number}") from exc

    def appraise(
        self, event_text: str, context: dict[str, Any] | None = None
    ) -> GenericAppraisalVector:
        payload = json.dumps(
            [DIRECT_VAD_SCHEMA.name, self.model, event_text, context],
            sort_keys=True,
            ensure_ascii=False,
        )
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if key not in self.cache:
            vector = self.engine.appraise(event_text, context=context)
            if not isinstance(vector, GenericAppraisalVector):
                raise RuntimeError("direct-VAD appraisal returned an unexpected vector type")
            dimensions = dict(vector.dimensions)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"key": key, "dimensions": dimensions}) + "\n")
            self.cache[key] = dimensions
        return GenericAppraisalVector(self.cache[key], DIRECT_VAD_SCHEMA)

    def close(self) -> None:
        self.engine.close()
