"""Dump per-event SEC vectors + oracle affect for the mapping recalibration (Addendum O).

For every event in realistic_recall_v3.json, call ``LLMAppraisalEngine.appraise()`` once and
record the 5 raw Scherer SEC dimensions alongside the dataset's oracle valence/arousal and the
owning scenario_id (so the frozen by-scenario split can be reconstructed offline). The output
JSONL is the single LLM-costed artifact of the study; ``fit.py`` consumes it with no LLM calls.

Pre-registration: benchmarks/preregistration_addendum_o_mapping_recalibration.md
Output:           benchmarks/appraisal_calibration/sec_dump.gpt5mini.jsonl

Usage::

    # smoke test (fixed vector, no LLM, no key):
    uv run python -m benchmarks.appraisal_calibration.dump_sec --dry-run --limit 3

    # full dump (requires EMOTIONAL_MEMORY_LLM_API_KEY):
    uv run python -m benchmarks.appraisal_calibration.dump_sec
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
DEFAULT_DATASET = _ROOT / "benchmarks" / "datasets" / "realistic_recall_v3.json"
DEFAULT_OUT = _HERE / "sec_dump.gpt5mini.jsonl"

_SEC_DIMS = ("novelty", "goal_relevance", "coping_potential", "norm_congruence", "self_relevance")


def _iter_events(dataset: dict[str, Any]):
    for sc in dataset["scenarios"]:
        sid = sc["scenario_id"]
        for se in sc["sessions"]:
            for ev in se["events"]:
                yield sid, ev


def _build_engine(dry_run: bool) -> Any:
    if dry_run:
        from emotional_memory.appraisal import AppraisalVector

        fixed = AppraisalVector(
            novelty=0.5,
            goal_relevance=0.2,
            coping_potential=0.6,
            norm_congruence=0.1,
            self_relevance=0.4,
        )

        class _Fixed:
            def appraise(self, _text: str, context: dict[str, Any] | None = None) -> Any:
                return fixed

        return _Fixed()

    from emotional_memory.appraisal_llm import LLMAppraisalEngine
    from emotional_memory.llm_http import make_httpx_llm_from_env

    llm = make_httpx_llm_from_env()
    if llm is None:
        raise RuntimeError(
            "EMOTIONAL_MEMORY_LLM_API_KEY is not set. Set it in .env or use --dry-run."
        )
    return LLMAppraisalEngine(llm=llm)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="Cap events (debug).")
    parser.add_argument("--dry-run", action="store_true", help="Fixed vector; no LLM.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    engine = _build_engine(args.dry_run)

    rows: list[dict[str, Any]] = []
    for i, (sid, ev) in enumerate(_iter_events(dataset)):
        if args.limit is not None and i >= args.limit:
            break
        if args.verbose:
            print(f"  [{i}] {ev['memory_id']} …")
        av = engine.appraise(ev["content"])
        row = {
            "memory_id": ev["memory_id"],
            "scenario_id": sid,
            "oracle_valence": float(ev["valence"]),
            "oracle_arousal": float(ev["arousal"]),
            **{dim: float(getattr(av, dim)) for dim in _SEC_DIMS},
        }
        rows.append(row)

    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} rows → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
