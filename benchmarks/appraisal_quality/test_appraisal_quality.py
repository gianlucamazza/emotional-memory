"""LLM appraisal prompt quality benchmarks.

Validates that the Scherer CPM prompt in LLMAppraisalEngine produces
directionally correct appraisal vectors for a gold-standard phrase dataset.

Gated behind:
    - pytest.mark.appraisal_quality
    - EMOTIONAL_MEMORY_LLM_API_KEY env var (skipped if missing)

Each phrase is run N times (default 3, via EMOTIONAL_MEMORY_LLM_REPEATS env var)
and assertions are evaluated against the median to dampen non-determinism.

Run with:
    EMOTIONAL_MEMORY_LLM_API_KEY=... pytest benchmarks/appraisal_quality/ -v -m appraisal_quality
"""

from __future__ import annotations

import os
import statistics

import pytest
from llm_helpers import make_llm_or_skip

from benchmarks.appraisal_quality.dataset import APPRAISAL_DATASET, AppraisalCase
from emotional_memory.appraisal_llm import LLMAppraisalConfig, LLMAppraisalEngine

pytestmark = pytest.mark.appraisal_quality

_REPEATS = int(os.environ.get("EMOTIONAL_MEMORY_LLM_REPEATS", "3"))


@pytest.fixture(scope="module")
def appraisal_engine() -> LLMAppraisalEngine:
    llm = make_llm_or_skip()
    # Disable cache so each repeat issues a fresh LLM call
    config = LLMAppraisalConfig(cache_size=0)
    return LLMAppraisalEngine(llm=llm, config=config)


@pytest.mark.parametrize(
    "case",
    APPRAISAL_DATASET,
    ids=[c.label for c in APPRAISAL_DATASET],
)
def test_directional_appraisal(appraisal_engine: LLMAppraisalEngine, case: AppraisalCase) -> None:
    """Validate LLM appraisal matches directional expectations (median over N repeats)."""
    vectors = [appraisal_engine.appraise(case.phrase) for _ in range(_REPEATS)]

    for assertion in case.assertions:
        values = [getattr(v, assertion.dimension) for v in vectors]
        median_val = statistics.median(values)

        if assertion.operator == ">":
            assert median_val > assertion.threshold, (
                f"[{case.label}] Expected median {assertion.dimension} > {assertion.threshold}, "
                f"got {median_val:.3f}  (all values: {[f'{v:.3f}' for v in values]})\n"
                f"  phrase: {case.phrase!r}"
            )
        else:
            assert median_val < assertion.threshold, (
                f"[{case.label}] Expected median {assertion.dimension} < {assertion.threshold}, "
                f"got {median_val:.3f}  (all values: {[f'{v:.3f}' for v in values]})\n"
                f"  phrase: {case.phrase!r}"
            )
