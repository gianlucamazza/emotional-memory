"""AFT adapter with retrieve-time query appraisal for DailyDialog (Addendum T2A).

Identical to ``AFTDailyDialogAdapter`` on the encode side (oracle session PAD
injected per session). The only difference is the query channel: instead of
relying on the leftover runtime affect, it appraises the query text with
``DIRECT_VAD_SCHEMA`` (Addendum V) and passes the result as ``query_affect`` to
the public retrieval API — the production-reachable mechanism from Addendum T,
with no oracle on the query side and no runtime-state mutation.
"""

from __future__ import annotations

from benchmarks.dailydialog.adapters.aft import AFTDailyDialogAdapter
from emotional_memory import DIRECT_VAD_SCHEMA
from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal_llm import LLMAppraisalConfig, LLMAppraisalEngine
from emotional_memory.llm_http import OpenAICompatibleLLMConfig, make_httpx_llm


class AFTQueryAppraisedDailyDialogAdapter(AFTDailyDialogAdapter):
    """AFT + retrieve-time query appraisal (direct-VAD)."""

    name = "aft_query_appraised"

    def __init__(self, *, embedder_name: str = "multilingual-e5-small") -> None:
        super().__init__(embedder_name=embedder_name)
        cfg = OpenAICompatibleLLMConfig.from_env()
        if cfg is None:
            raise RuntimeError("EMOTIONAL_MEMORY_LLM_API_KEY not set — cannot appraise queries.")
        # Cache enabled: templated queries repeat across personas, so each distinct
        # query text is appraised once.
        self._query_appraiser = LLMAppraisalEngine(
            llm=make_httpx_llm(cfg),
            config=LLMAppraisalConfig(
                cache_size=4096, fallback_on_error=True, appraisal_schema=DIRECT_VAD_SCHEMA
            ),
        )
        # query_text -> appraised CoreAffect, for the diagnostic correlation.
        self.appraised_affect: dict[str, CoreAffect] = {}

    def retrieve(self, query_text: str, *, top_k: int) -> list[str]:
        engine = self._require_engine()
        ca = self._query_appraiser.appraise(query_text).to_core_affect()
        self.appraised_affect[query_text] = ca
        memories = engine.retrieve(query_text, top_k=top_k, query_affect=ca)
        return [self._memory_session_map.get(m.id, "") for m in memories]
