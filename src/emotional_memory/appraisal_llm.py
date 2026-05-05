"""LLM-backed and heuristic AppraisalEngine implementations.

Provides two concrete AppraisalEngine implementations:

LLMAppraisalEngine
    Provider-agnostic: the caller supplies any callable that accepts a prompt
    string and a JSON schema dict and returns a raw JSON string.  This avoids
    a hard dependency on any specific SDK (openai, anthropic, etc.).

    Features:
    - LRU cache (configurable size) to avoid re-appraising identical inputs
    - Fallback to a neutral AppraisalVector on parse/call errors
    - Context forwarded to the prompt when provided by the engine

KeywordAppraisalEngine
    Rule-based fallback that requires no external calls.  Keyword patterns map
    to dimension scores via accumulation.  Useful as a default when no LLM is
    available or as the ``fallback`` inside LLMAppraisalEngine.

Usage example::

    import anthropic

    client = anthropic.Anthropic()

    def my_llm(prompt: str, json_schema: dict) -> str:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    engine = LLMAppraisalEngine(llm=my_llm)
    vector = engine.appraise("I just finished a hard project successfully!")
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from emotional_memory.appraisal import AppraisalVector, GenericAppraisalVector
from emotional_memory.appraisal_schema import SCHERER_CPM_SCHEMA, AppraisalSchema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON schema exposed to the LLM
# ---------------------------------------------------------------------------

_APPRAISAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "novelty": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": "How unexpected: -1=fully expected, 0=neutral, 1=totally new",
        },
        "goal_relevance": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": "Relation to goals: -1=obstructs, 0=irrelevant, 1=furthers",
        },
        "coping_potential": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Perceived ability to handle: 0=helpless, 1=full control",
        },
        "norm_congruence": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": "Alignment with norms/values: -1=violates, 0=neutral, 1=conforms",
        },
        "self_relevance": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Personal significance: 0=irrelevant, 1=deeply personal",
        },
    },
    "required": [
        "novelty",
        "goal_relevance",
        "coping_potential",
        "norm_congruence",
        "self_relevance",
    ],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """\
You are an emotion appraisal system implementing Scherer's Component Process Model.
Given an event description, evaluate it on 5 dimensions and return ONLY a JSON object.

Dimensions:
- novelty          [-1, 1]  How unexpected. -1=fully expected, 1=totally new.
- goal_relevance   [-1, 1]  Relation to goals. -1=obstructs, 1=furthers.
- coping_potential [0,  1]  Perceived ability to handle. 0=helpless, 1=full control.
- norm_congruence  [-1, 1]  Alignment with norms/values. -1=violates, 1=conforms.
- self_relevance   [0,  1]  Personal significance. 0=irrelevant, 1=deeply personal.

Return ONLY valid JSON with these exact keys. No explanation, no markdown.\
"""


# ---------------------------------------------------------------------------
# LLMCallable protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMCallable(Protocol):
    """Provider-agnostic LLM interface.

    The user wraps their SDK of choice (openai, anthropic, local GGUF, …) in
    a callable matching this signature.  The ``json_schema`` parameter is
    passed for informational purposes — structured-output enforcement is the
    caller's responsibility if the backend supports it.
    """

    def __call__(self, prompt: str, json_schema: dict[str, Any]) -> str:
        """Send ``prompt`` to the LLM, return a raw JSON string."""
        ...


# ---------------------------------------------------------------------------
# LLMAppraisalConfig
# ---------------------------------------------------------------------------


class LLMAppraisalConfig(BaseModel):
    """Configuration for LLMAppraisalEngine."""

    model_config = {"arbitrary_types_allowed": True}

    system_prompt: str = _SYSTEM_PROMPT
    """System prompt describing the appraisal task.

    When ``appraisal_schema`` is set and this field is left at its default, the engine
    uses ``appraisal_schema.system_prompt`` instead.  Explicitly setting this field
    always takes precedence.
    """

    cache_size: int = 128
    """Number of (text, context) pairs to cache.  0 disables the cache."""

    fallback_on_error: bool = True
    """Return a neutral AppraisalVector instead of raising on LLM/parse errors."""

    appraisal_schema: AppraisalSchema | None = None
    """Appraisal schema to use.  ``None`` defaults to ``SCHERER_CPM_SCHEMA``."""


# ---------------------------------------------------------------------------
# LLMAppraisalEngine
# ---------------------------------------------------------------------------


class LLMAppraisalEngine:
    """AppraisalEngine backed by any LLM via a provider-agnostic callable.

    Thread-safety: the LRU cache is protected by a ``threading.Lock``.
    Concurrent ``appraise()`` calls with the same key will each call the LLM
    independently (no stampede protection), but the cache dict is always
    mutated under the lock.
    """

    __slots__ = (
        "_cache",
        "_cache_lock",
        "_config",
        "_effective_prompt",
        "_fallback",
        "_json_schema",
        "_llm",
        "_schema",
    )

    def __init__(
        self,
        llm: LLMCallable,
        config: LLMAppraisalConfig | None = None,
        fallback: AppraisalVector | None = None,
    ) -> None:
        self._llm = llm
        self._config = config or LLMAppraisalConfig()
        self._schema: AppraisalSchema = self._config.appraisal_schema or SCHERER_CPM_SCHEMA
        self._fallback = fallback or AppraisalVector.neutral()
        self._cache: OrderedDict[str, AppraisalVector | GenericAppraisalVector] = OrderedDict()
        self._cache_lock = threading.Lock()

        # Effective prompt: user-set system_prompt in config overrides schema default.
        # Detect via model_fields_set so the schema prompt wins when config is default.
        if "system_prompt" in self._config.model_fields_set:
            self._effective_prompt: str = self._config.system_prompt
        else:
            self._effective_prompt = self._schema.system_prompt

        self._json_schema: dict[str, Any] = self._schema.to_json_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def appraise(
        self, event_text: str, context: dict[str, Any] | None = None
    ) -> AppraisalVector | GenericAppraisalVector:
        """Appraise *event_text* via the LLM.

        Returns an ``AppraisalVector`` for the default Scherer CPM schema or a
        ``GenericAppraisalVector`` when a custom ``AppraisalSchema`` is configured.

        Results are cached by (schema, event_text, context) key.  On error, if
        ``fallback_on_error`` is True, returns the neutral fallback vector
        instead of propagating the exception.
        """
        cache_key = self._make_cache_key(event_text, context, self._schema.name)

        if self._config.cache_size > 0:
            with self._cache_lock:
                if cache_key in self._cache:
                    self._cache.move_to_end(cache_key)
                    logger.debug("appraise cache hit: key=%s", cache_key[:8])
                    return self._cache[cache_key]

        prompt = self._build_prompt(event_text, context)
        vector: AppraisalVector | GenericAppraisalVector
        try:
            raw = self._llm(prompt, self._json_schema)
        except Exception as exc:
            if self._config.fallback_on_error:
                logger.warning("appraise: LLM call failed, using fallback: %s", exc)
                vector = self._fallback
            else:
                raise
        else:
            try:
                data = self._extract_json(raw)
                if self._schema is SCHERER_CPM_SCHEMA:
                    vector = AppraisalVector(**data)
                else:
                    vector = GenericAppraisalVector(dimensions=data, schema=self._schema)
            except Exception:
                if self._config.fallback_on_error:
                    logger.debug(
                        "appraise: parse/validation error, using fallback (text_len=%d)",
                        len(event_text),
                        exc_info=True,
                    )
                    vector = self._fallback
                else:
                    raise

        if self._config.cache_size > 0:
            with self._cache_lock:
                if len(self._cache) >= self._config.cache_size:
                    self._cache.popitem(last=False)
                self._cache[cache_key] = vector

        return vector

    def clear_cache(self) -> None:
        """Evict all cached appraisals."""
        with self._cache_lock:
            self._cache.clear()

    def __repr__(self) -> str:
        with self._cache_lock:
            cache_size = len(self._cache)
        return (
            f"{type(self).__name__}("
            f"cache_size={self._config.cache_size}, "
            f"cached={cache_size}, "
            f"fallback_on_error={self._config.fallback_on_error})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, event_text: str, context: dict[str, Any] | None) -> str:
        parts = [self._effective_prompt, "", f'Event: "{event_text}"']
        if context:
            parts.append(f"Context: {json.dumps(context, ensure_ascii=False)}")
        return "\n".join(parts)

    @staticmethod
    def _make_cache_key(event_text: str, context: dict[str, Any] | None, schema_name: str) -> str:
        raw = schema_name + event_text + (json.dumps(context, sort_keys=True) if context else "")
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        """Extract the first JSON object from *raw*, tolerating markdown fences."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        # Find first {...} block (non-greedy to avoid spanning multiple objects)
        match = re.search(r"\{[^{}]*\}", cleaned)
        if not match:
            raise ValueError(f"No JSON object found in LLM response: {raw!r}")
        result = json.loads(match.group())
        if not isinstance(result, dict):
            raise TypeError(f"Expected JSON object, got {type(result)}")
        return result


# ---------------------------------------------------------------------------
# KeywordAppraisalEngine
# ---------------------------------------------------------------------------


class KeywordRule:
    """A single keyword pattern contributing dimension score *deltas*.

    All scores are deltas from the neutral baseline (0.0 for signed dims,
    0.0 for coping_potential/self_relevance — the engine adds the baseline
    0.5 for coping_potential after accumulation).
    """

    __slots__ = ("_re", "scores")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(pattern={self._re.pattern!r})"

    def __init__(
        self,
        pattern: str,
        novelty: float = 0.0,
        goal_relevance: float = 0.0,
        coping_potential: float = 0.0,
        norm_congruence: float = 0.0,
        self_relevance: float = 0.0,
    ) -> None:
        self._re = re.compile(pattern, re.IGNORECASE)
        self.scores = {
            "novelty": novelty,
            "goal_relevance": goal_relevance,
            "coping_potential": coping_potential,
            "norm_congruence": norm_congruence,
            "self_relevance": self_relevance,
        }

    def matches(self, text: str) -> bool:
        return bool(self._re.search(text))


# Coping-potential values are deltas from the 0.5 resting baseline.
# Positive delta → more in control; negative → less in control.
_DEFAULT_RULES: list[KeywordRule] = [
    KeywordRule(
        r"\b(success|succeed(?:ed|s)?|accomplished|completed|achieved|won|victory)\b",
        goal_relevance=0.7,
        coping_potential=+0.3,  # 0.5 + 0.3 = 0.8 — high control
        norm_congruence=0.5,
        self_relevance=0.6,
    ),
    KeywordRule(
        r"\b(fail|failed|failure|mistake|error|wrong|broken)\b",
        goal_relevance=-0.6,
        coping_potential=-0.3,  # 0.5 - 0.3 = 0.2 — low control
        norm_congruence=-0.3,
        self_relevance=0.5,
    ),
    KeywordRule(
        r"\b(surprise|unexpected|sudden|shock|startl|amazed|astonish)\b",
        novelty=0.8,
        self_relevance=0.3,
    ),
    KeywordRule(
        r"\b(boring|routine|usual|same as|as expected|predictable)\b",
        novelty=-0.7,
        self_relevance=0.1,
    ),
    KeywordRule(
        r"\b(danger|threat|risk|harm|attack|crisis|emergency)\b",
        goal_relevance=-0.5,
        coping_potential=-0.4,  # 0.5 - 0.4 = 0.1 — very low control
        norm_congruence=-0.2,
        self_relevance=0.7,
    ),
    KeywordRule(
        r"\b(help|support|kind|generous|caring|compassionate)\b",
        goal_relevance=0.4,
        norm_congruence=0.6,
        self_relevance=0.3,
    ),
    KeywordRule(
        r"\b(unfair|unjust|violat|cheat|lie|betray|abuse)\b",
        goal_relevance=-0.4,
        coping_potential=-0.3,  # 0.5 - 0.3 = 0.2
        norm_congruence=-0.8,
        self_relevance=0.6,
    ),
    KeywordRule(
        r"\b(I|me|my|myself|mine|personally)\b",
        self_relevance=0.4,
    ),
]


# Italian keyword rules — same delta semantics as _DEFAULT_RULES.
_ITALIAN_RULES: list[KeywordRule] = [
    KeywordRule(
        r"\b(success(?:so|o)|riusc(?:ito|ita|iti)|complet(?:ato|ata)|raggiunt(?:o|a)|vint(?:o|a)|vittoria)\b",
        goal_relevance=0.7,
        coping_potential=+0.3,
        norm_congruence=0.5,
        self_relevance=0.6,
    ),
    KeywordRule(
        r"\b(fallim(?:ento|ento)|fallito|errore|sbaglio|rotto|guasto)\b",
        goal_relevance=-0.6,
        coping_potential=-0.3,
        norm_congruence=-0.3,
        self_relevance=0.5,
    ),
    KeywordRule(
        r"\b(sorpres[ao]|inaspettat[ao]|improvvis[ao]|shock|stupito|meravigliat[ao])\b",
        novelty=0.8,
        self_relevance=0.3,
    ),
    KeywordRule(
        r"\b(noioso|routine|solito|come sempre|prevedibile)\b",
        novelty=-0.7,
        self_relevance=0.1,
    ),
    KeywordRule(
        r"\b(pericolo|minaccia|rischio|danno|attacco|crisi|emergenza)\b",
        goal_relevance=-0.5,
        coping_potential=-0.4,
        norm_congruence=-0.2,
        self_relevance=0.7,
    ),
    KeywordRule(
        r"\b(aiuto|supporto|gentile|generoso|premuroso|compassionevole)\b",
        goal_relevance=0.4,
        norm_congruence=0.6,
        self_relevance=0.3,
    ),
    KeywordRule(
        r"\b(ingiusto|ingiustizia|imbroglio|menzogna|tradimento|abuso)\b",
        goal_relevance=-0.4,
        coping_potential=-0.3,
        norm_congruence=-0.8,
        self_relevance=0.6,
    ),
    KeywordRule(
        r"\b(io|me|mio|mia|miei|mie|stesso|stessa|personalmente)\b",
        self_relevance=0.4,
    ),
]


class KeywordAppraisalEngine:
    """Rule-based AppraisalEngine using keyword pattern matching.

    Each rule contributes *deltas* from neutral baselines.  After accumulation
    the results are averaged (when multiple rules fire) and the coping_potential
    baseline of 0.5 is re-applied.  AppraisalVector validators clamp to range.
    """

    __slots__ = ("_rules",)

    def __init__(self, rules: list[KeywordRule] | None = None) -> None:
        self._rules = rules if rules is not None else _DEFAULT_RULES

    @classmethod
    def make_multilingual(cls) -> KeywordAppraisalEngine:
        """Return an engine with English + Italian rules combined."""
        return cls(rules=_DEFAULT_RULES + _ITALIAN_RULES)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(rules={len(self._rules)})"

    def appraise(self, event_text: str, context: dict[str, Any] | None = None) -> AppraisalVector:
        # Accumulate deltas and per-dimension hit counts from zero.
        # Each dimension is averaged independently so that a dimension set
        # by only one rule is not diluted by other rules that did not touch it.
        accum: dict[str, float] = {
            "novelty": 0.0,
            "goal_relevance": 0.0,
            "coping_potential": 0.0,
            "norm_congruence": 0.0,
            "self_relevance": 0.0,
        }
        dim_hits: dict[str, int] = dict.fromkeys(accum, 0)
        for rule in self._rules:
            if rule.matches(event_text):
                for dim, val in rule.scores.items():
                    accum[dim] += val
                    dim_hits[dim] += 1

        # Average each dimension by the number of rules that contributed to it.
        for dim in accum:
            if dim_hits[dim] > 1:
                accum[dim] /= dim_hits[dim]

        # Re-apply the 0.5 resting baseline for coping_potential
        accum["coping_potential"] += 0.5

        return AppraisalVector(**accum)
