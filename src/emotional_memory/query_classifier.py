"""Query-type classifier for per-category retrieval weight routing.

Two implementations of the ``QueryClassifier`` protocol:

* ``HeuristicQueryClassifier`` — regex/keyword rules, negligible latency.
* ``LLMQueryClassifier`` — LLM-backed, mirrors ``LLMAppraisalEngine`` pattern
  (SHA-256 LRU cache, thread-safe, ``fallback_on_error`` semantics).

The routing table pre-derived from Addendum J closure is exported as
``LOCOMO_ROUTING`` for direct use as ``QueryClassifierConfig.routed_weights``.

See: benchmarks/preregistration_addendum_l_query_routing.md
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable

from emotional_memory.appraisal_llm import LLMCallable

__all__ = [
    "LOCOMO_ROUTING",
    "QUERY_TYPES",
    "HeuristicQueryClassifier",
    "LLMQueryClassifier",
    "QueryClassifier",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUERY_TYPES: frozenset[str] = frozenset(
    {"single_hop", "multi_hop", "temporal", "open_domain", "default"}
)

# Routing table from Addendum J closure (preregistration_addendum_j_closure.md:79-91).
# Weights: [semantic, mood_congruence, affect_proximity, momentum, recency, resonance_boost]
LOCOMO_ROUTING: dict[str, list[float]] = {
    # W7: suppress affect, maximise semantic for literal-fact questions
    "single_hop": [0.70, 0.10, 0.05, 0.05, 0.05, 0.05],
    # W2: closest to naive_rag on multi_hop; moderate mood, no resonance
    "multi_hop": [0.50, 0.30, 0.10, 0.05, 0.05, 0.00],
    # W5: moderate semantic + mood for narrative questions
    "open_domain": [0.40, 0.20, 0.15, 0.10, 0.10, 0.05],
    # W2: best for temporal (no config helped; W2 minimises damage)
    "temporal": [0.50, 0.30, 0.10, 0.05, 0.05, 0.00],
    # W0: standard baseline (no routing override)
    "default": [0.35, 0.25, 0.15, 0.10, 0.10, 0.05],
}

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class QueryClassifier(Protocol):
    """Classifies a query string into a named query-type label."""

    def classify(self, query: str) -> str:
        """Return a query-type label (e.g. 'single_hop', 'temporal').

        Unknown labels are safe — the engine falls back to ``default_type``.
        """
        ...


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------

# Patterns are applied in priority order: temporal > multi_hop > single_hop > open_domain.
_TEMPORAL_RE = re.compile(
    r"\b(when|after|before|how long|until|since|during|at what (time|point)"
    r"|how (many|much) (years?|months?|days?|weeks?|hours?|times?)"
    r"|how (soon|late|early|often|frequently)"
    r"|did .+ (before|after|first|last))\b",
    re.IGNORECASE,
)
_MULTI_HOP_RE = re.compile(
    r"\b(and then|both|also|in addition|as well as|additionally|furthermore"
    r"|at the same time|meanwhile|alongside|together with)\b",
    re.IGNORECASE,
)
# Single-hop: short question starting with an interrogative, no comma cascade
_SINGLE_HOP_RE = re.compile(
    r"^(who|what|where|which|is |are |does |did |was |were )",
    re.IGNORECASE,
)
_SINGLE_HOP_MAX_CHARS = 100


class HeuristicQueryClassifier:
    """Regex/keyword routing — ~0 overhead, deterministic.

    Priority: temporal > multi_hop > single_hop > open_domain (default).
    """

    __slots__ = ()

    def classify(self, query: str) -> str:
        q = query.strip()
        if _TEMPORAL_RE.search(q):
            return "temporal"
        if _MULTI_HOP_RE.search(q):
            return "multi_hop"
        if len(q) <= _SINGLE_HOP_MAX_CHARS and _SINGLE_HOP_RE.match(q):
            return "single_hop"
        return "open_domain"

    def __repr__(self) -> str:
        return "HeuristicQueryClassifier()"


# ---------------------------------------------------------------------------
# LLM-backed classifier
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """\
Classify the following query into exactly one category:
- single_hop: a simple factual question with a single, direct answer
- multi_hop: requires combining information from multiple facts or events
- temporal: asks about time, sequence, duration, or when events occurred
- open_domain: a broad, narrative, or multi-faceted question

Return ONLY a JSON object {"query_type": "<category>"}. No explanation.\
"""

_LLM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query_type": {
            "type": "string",
            "enum": ["single_hop", "multi_hop", "temporal", "open_domain"],
        }
    },
    "required": ["query_type"],
    "additionalProperties": False,
}


class LLMQueryClassifier:
    """LLM-backed query classifier mirroring ``LLMAppraisalEngine`` pattern.

    Thread-safe SHA-256 LRU cache. Falls back to ``default_type`` on error
    when ``fallback_on_error=True`` (default).
    """

    __slots__ = ("_cache", "_cache_lock", "_cache_size", "_default_type", "_fallback", "_llm")

    def __init__(
        self,
        llm: LLMCallable,
        *,
        cache_size: int = 256,
        default_type: str = "open_domain",
        fallback_on_error: bool = True,
    ) -> None:
        self._llm = llm
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_size = cache_size
        self._default_type = default_type
        self._fallback = fallback_on_error

    def classify(self, query: str) -> str:
        cache_key = hashlib.sha256(query.encode()).hexdigest()
        with self._cache_lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]

        try:
            prompt = f'{_LLM_SYSTEM_PROMPT}\n\nQuery: "{query}"'
            raw = self._llm(prompt, _LLM_SCHEMA)
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            match = re.search(r"\{[^{}]*\}", cleaned)
            if not match:
                raise ValueError(f"No JSON in LLM response: {raw!r}")
            result: dict[str, Any] = json.loads(match.group())
            query_type = str(result.get("query_type", self._default_type))
        except Exception:
            if not self._fallback:
                raise
            query_type = self._default_type

        with self._cache_lock:
            self._cache[cache_key] = query_type
            self._cache.move_to_end(cache_key)
            if self._cache_size > 0 and len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return query_type

    def __repr__(self) -> str:
        with self._cache_lock:
            n = len(self._cache)
        return (
            f"LLMQueryClassifier("
            f"cache_size={self._cache_size}, "
            f"cached={n}, "
            f"default_type={self._default_type!r})"
        )
