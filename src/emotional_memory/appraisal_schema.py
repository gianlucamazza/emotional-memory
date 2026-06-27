"""Pluggable appraisal schemas for parametric emotion-theory support.

Ships one built-in schema: ``SCHERER_CPM_SCHEMA`` (Scherer 1984 CPM, 5 SECs).
Custom schemas (OCC, GRID, domain-specific) can be plugged into
``LLMAppraisalEngine`` without forking the library.

Usage::

    from emotional_memory import AppraisalSchema, AppraisalDimension, LLMAppraisalEngine

    my_schema = AppraisalSchema(
        name="occ_mini",
        dimensions=(
            AppraisalDimension(
                "desirability", (-1.0, 1.0), 0.0, "Desired outcome: -1=bad, 1=good"
            ),
            AppraisalDimension(
                "praiseworthiness", (-1.0, 1.0), 0.0, "Action quality: -1=blame, 1=praise"
            ),
        ),
        system_prompt="Rate the event on desirability and praiseworthiness. Return JSON.",
        project_to_core_affect=lambda d: CoreAffect(
            valence=0.6 * d["desirability"] + 0.4 * d["praiseworthiness"],
            arousal=0.5 * abs(d["desirability"]),
            dominance=0.5,
        ),
    )

    engine = LLMAppraisalEngine(llm=my_llm, config=LLMAppraisalConfig(appraisal_schema=my_schema))
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from emotional_memory.affect import CoreAffect


class AppraisalDimension(BaseModel):
    """One axis of an AppraisalSchema.

    Attributes:
        name: Machine-readable identifier (used as JSON key and dict key).
        range: ``(lo, hi)`` — the valid numeric interval for this dimension.
        neutral: Value representing the neutral / resting point.
        description: Human-readable description forwarded to the LLM prompt.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    range: tuple[float, float]
    neutral: float
    description: str

    @field_validator("range")
    @classmethod
    def _range_ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError(f"range lo must be < hi, got {v}")
        return v


class AppraisalSchema:
    """A complete appraisal theory schema.

    Bundles the dimension definitions, the LLM system prompt, and the
    projection function that maps a dimension dict to ``CoreAffect``.

    Attributes:
        name: Short identifier string (e.g. ``"scherer_cpm"``).
        dimensions: Ordered tuple of ``AppraisalDimension`` objects.
        system_prompt: System prompt to send to the LLM.
        project_to_core_affect: Callable mapping ``{dim_name: float}`` → ``CoreAffect``.
    """

    __slots__ = ("dimensions", "name", "project_to_core_affect", "system_prompt")

    def __init__(
        self,
        name: str,
        dimensions: tuple[AppraisalDimension, ...],
        system_prompt: str,
        project_to_core_affect: Callable[[Mapping[str, float]], CoreAffect],
    ) -> None:
        self.name = name
        self.dimensions = dimensions
        self.system_prompt = system_prompt
        self.project_to_core_affect = project_to_core_affect

    def to_json_schema(self) -> dict[str, Any]:
        """Derive a JSON schema dict from ``self.dimensions`` for LLM structured output."""
        props: dict[str, Any] = {}
        for dim in self.dimensions:
            lo, hi = dim.range
            props[dim.name] = {
                "type": "number",
                "minimum": lo,
                "maximum": hi,
                "description": dim.description,
            }
        return {
            "type": "object",
            "properties": props,
            "required": [d.name for d in self.dimensions],
            "additionalProperties": False,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"dimensions={[d.name for d in self.dimensions]!r})"
        )


# ---------------------------------------------------------------------------
# Scherer CPM schema — the library default
# ---------------------------------------------------------------------------

_SCHERER_SYSTEM_PROMPT = """\
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


def _scherer_project(dims: Mapping[str, float]) -> CoreAffect:
    """Scherer CPM → valence/arousal projection.

    Weights recalibrated against oracle affect on a held-out scenario split
    (Addendum O, gpt-5-mini, N=750; closure:
    benchmarks/preregistration_addendum_o_mapping_recalibration_closure.md). The
    Scherer feature basis is preserved (``coping_signed``, ``|novelty|``,
    ``1 - coping_potential``); only the coefficients changed, fitted least-squares
    on the train scenarios. The valence intercept is held at 0 so a neutral
    appraisal still maps to valence 0; the arousal intercept (0.1399) is the
    free-fit offset. ``CoreAffect`` validators clamp the outputs to range.
    """
    coping_signed = 2.0 * dims["coping_potential"] - 1.0  # [0,1] → [-1,+1]
    valence = (
        0.4805 * dims["goal_relevance"]
        + 0.1862 * dims["norm_congruence"]
        + 0.1643 * coping_signed
        + 0.0179 * dims["novelty"]
    )
    arousal = (
        0.3694 * abs(dims["novelty"])
        + 0.1357 * (1.0 - dims["coping_potential"])
        + 0.2208 * dims["self_relevance"]
        + 0.1399
    )
    return CoreAffect(valence=valence, arousal=arousal, dominance=dims["coping_potential"])


SCHERER_CPM_SCHEMA: AppraisalSchema = AppraisalSchema(
    name="scherer_cpm",
    dimensions=(
        AppraisalDimension(
            name="novelty",
            range=(-1.0, 1.0),
            neutral=0.0,
            description="How unexpected: -1=fully expected, 0=neutral, 1=totally new",
        ),
        AppraisalDimension(
            name="goal_relevance",
            range=(-1.0, 1.0),
            neutral=0.0,
            description="Relation to goals: -1=obstructs, 0=irrelevant, 1=furthers",
        ),
        AppraisalDimension(
            name="coping_potential",
            range=(0.0, 1.0),
            neutral=0.5,
            description="Perceived ability to handle: 0=helpless, 1=full control",
        ),
        AppraisalDimension(
            name="norm_congruence",
            range=(-1.0, 1.0),
            neutral=0.0,
            description="Alignment with norms/values: -1=violates, 0=neutral, 1=conforms",
        ),
        AppraisalDimension(
            name="self_relevance",
            range=(0.0, 1.0),
            neutral=0.0,
            description="Personal significance: 0=irrelevant, 1=deeply personal",
        ),
    ),
    system_prompt=_SCHERER_SYSTEM_PROMPT,
    project_to_core_affect=_scherer_project,
)


_DIRECT_VAD_SYSTEM_PROMPT = """\
You are an affect rating system. Given an event description, rate the emotion it
expresses on three dimensions and return ONLY a JSON object (no explanation, no markdown):
- valence   [-1, 1]  unpleasant/negative -> pleasant/positive
- arousal   [0, 1]   calm/subdued -> excited/activated
- dominance [0, 1]   controlled/submissive -> in-control/dominant
Return ONLY valid JSON with these exact keys.\
"""


DIRECT_VAD_SCHEMA: AppraisalSchema = AppraisalSchema(
    name="direct_vad",
    dimensions=(
        AppraisalDimension(
            name="valence",
            range=(-1.0, 1.0),
            neutral=0.0,
            description="Pleasantness: -1=unpleasant/negative, 0=neutral, 1=pleasant/positive",
        ),
        AppraisalDimension(
            name="arousal",
            range=(0.0, 1.0),
            neutral=0.5,
            description="Activation: 0=calm/subdued, 1=excited/activated",
        ),
        AppraisalDimension(
            name="dominance",
            range=(0.0, 1.0),
            neutral=0.5,
            description="Control: 0=controlled/submissive, 1=in-control/dominant",
        ),
    ),
    system_prompt=_DIRECT_VAD_SYSTEM_PROMPT,
    project_to_core_affect=lambda d: CoreAffect(
        valence=d["valence"], arousal=d["arousal"], dominance=d["dominance"]
    ),
)
"""Direct valence/arousal/dominance appraisal — the LLM rates V/A/D directly
(identity projection) instead of the 5 Scherer SECs.

Opt-in alternative to the default ``SCHERER_CPM_SCHEMA``. Against human-annotated
affect (EmoBank, N=300) it is better correlated on every axis — valence r=0.79
(near-zero bias), arousal r=0.58, dominance r=0.43 — vs the SEC->projection's
0.70 / 0.23 / 0.31 (Addendum V,
``benchmarks/preregistration_addendum_v_direct_vad_closure.md``). Trade-offs: it
yields a ``GenericAppraisalVector`` (no Scherer SEC fields), so the dual-path
``elaborate()`` SEC-appraisal storage and any SEC-reading feature require the
default schema; and its arousal absolute scale is less calibrated (higher MAE)
than its correlation. Use when you only need ``CoreAffect`` and want stronger
human-gold agreement. Select via
``LLMAppraisalConfig(appraisal_schema=DIRECT_VAD_SCHEMA)``.
"""
