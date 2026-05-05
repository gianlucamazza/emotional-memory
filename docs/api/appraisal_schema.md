# Appraisal Schema

The `appraisal_schema` module provides a pluggable appraisal-theory mechanism.
Instead of hard-coding Scherer's 5 SECs everywhere, you define an `AppraisalSchema`
that bundles dimension definitions, an LLM prompt, and a projection function.

## Classes

::: emotional_memory.appraisal_schema.AppraisalDimension

::: emotional_memory.appraisal_schema.AppraisalSchema

::: emotional_memory.appraisal.GenericAppraisalVector

## Built-in schema

::: emotional_memory.appraisal_schema.SCHERER_CPM_SCHEMA
    options:
      show_docstring_attributes: false

## Defining a custom schema

```python
from emotional_memory import (
    AppraisalDimension,
    AppraisalSchema,
    LLMAppraisalConfig,
    LLMAppraisalEngine,
)
from emotional_memory.affect import CoreAffect

occ_schema = AppraisalSchema(
    name="occ_subset",
    dimensions=(
        AppraisalDimension(
            name="desirability",
            range=(-1.0, 1.0),
            neutral=0.0,
            description="Desired outcome: -1=bad, 1=good",
        ),
        AppraisalDimension(
            name="praiseworthiness",
            range=(-1.0, 1.0),
            neutral=0.0,
            description="Agent's action quality: -1=blame, 1=praise",
        ),
    ),
    system_prompt=(
        "Evaluate the event on OCC dimensions: desirability and praiseworthiness. "
        "Return ONLY a JSON object with those two keys."
    ),
    project_to_core_affect=lambda d: CoreAffect(
        valence=0.6 * d["desirability"] + 0.4 * d["praiseworthiness"],
        arousal=0.5 * abs(d["desirability"]),
        dominance=0.5,
    ),
)

engine = LLMAppraisalEngine(
    llm=my_llm,
    config=LLMAppraisalConfig(appraisal_schema=occ_schema),
)
result = engine.appraise("She won the competition against all odds.")
# result is GenericAppraisalVector(schema_name="occ_subset", ...)
core = result.to_core_affect()
```

The `project_to_core_affect` callable receives a `Mapping[str, float]` where each
key is a dimension name and each value is the LLM's rating, clamped to `dim.range`.
It must return a `CoreAffect` instance.

## Back-compatibility

`LLMAppraisalEngine()` without an explicit `appraisal_schema` continues to use
`SCHERER_CPM_SCHEMA` and returns `AppraisalVector` objects as before.
All existing code that constructs `AppraisalVector` directly or reads its named
fields is unaffected.
