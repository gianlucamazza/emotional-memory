# Custom appraisal schema

`emotional_memory` ships with the Scherer CPM schema (5 Stimulus Evaluation Checks)
as its default appraisal theory. You can replace it with any schema that maps event
text to emotion dimensions — OCC, the GRID model, or a domain-specific theory.

This tutorial shows how to build and plug in a custom schema.

## What is an AppraisalSchema?

An `AppraisalSchema` bundles:

1. **Dimension definitions** — what the LLM should score and on what scale.
2. **System prompt** — the instruction forwarded to the LLM.
3. **Projection function** — a callable that maps the scored dimensions to `CoreAffect(valence, arousal, dominance)`.

## Example: OCC mini-schema

The OCC model (Ortony, Clore & Collins 1988) evaluates events on two primary axes:
_desirability_ (how good/bad an outcome is) and _praiseworthiness_ (whether an agent's
action deserves credit or blame).

```python
from emotional_memory.affect import CoreAffect
from emotional_memory.appraisal_schema import AppraisalDimension, AppraisalSchema
from emotional_memory.appraisal_llm import LLMAppraisalConfig, LLMAppraisalEngine

occ_schema = AppraisalSchema(
    name="occ_mini",
    dimensions=(
        AppraisalDimension(
            name="desirability",
            range=(-1.0, 1.0),
            neutral=0.0,
            description="How desirable is this event outcome? -1 = very bad, 1 = very good.",
        ),
        AppraisalDimension(
            name="praiseworthiness",
            range=(-1.0, 1.0),
            neutral=0.0,
            description="Is the agent's action praiseworthy? -1 = blameworthy, 1 = praiseworthy.",
        ),
    ),
    system_prompt=(
        "You are an emotion appraisal system based on the OCC model. "
        "Score the event on desirability and praiseworthiness. "
        "Return a JSON object with numeric values in [-1, 1]."
    ),
    project_to_core_affect=lambda d: CoreAffect(
        valence=0.6 * d["desirability"] + 0.4 * d["praiseworthiness"],
        arousal=0.5 * abs(d["desirability"]),
        dominance=0.5 + 0.3 * d["praiseworthiness"],
    ),
)
```

## Plug it into the engine

```python
from emotional_memory import EmotionalMemory, InMemoryStore, EmotionalMemoryConfig
from emotional_memory.appraisal_llm import LLMAppraisalConfig, LLMAppraisalEngine
from emotional_memory.llm_http import make_httpx_llm_from_env

appraisal_engine = LLMAppraisalEngine(
    llm=make_httpx_llm_from_env(),
    config=LLMAppraisalConfig(appraisal_schema=occ_schema),
)

em = EmotionalMemory(
    store=InMemoryStore(),
    embedder=my_embedder,
    appraisal_engine=appraisal_engine,
)

em.encode("The project succeeded and the team was recognised for their effort.")
```

The engine calls `occ_schema.project_to_core_affect()` to convert the LLM scores
into a `CoreAffect` that drives retrieval weighting.

## Dimension validation

`LLMAppraisalEngine` validates LLM output against each dimension's `range`:
values outside `[lo, hi]` are clamped and a warning is logged.

```python
# Verify the schema's JSON schema representation
print(occ_schema.to_json_schema())
# {'desirability': {'type': 'number', 'minimum': -1.0, 'maximum': 1.0, ...}, ...}
```

## Keyword fallback

For development and testing without an LLM, `KeywordAppraisalEngine` provides
a rule-based fallback. It does not support custom schemas — it always outputs
`AppraisalVector` (Scherer CPM). Combine with `LLMAppraisalEngine`'s `fallback=`
parameter for graceful degradation:

```python
from emotional_memory.appraisal_llm import KeywordAppraisalEngine

appraisal_engine = LLMAppraisalEngine(
    llm=make_httpx_llm_from_env(),
    config=LLMAppraisalConfig(appraisal_schema=occ_schema),
    fallback=KeywordAppraisalEngine(),  # activates if LLM call fails
)
```

## Built-in alternative: `DIRECT_VAD_SCHEMA`

A ready-made opt-in schema ships with the library: instead of the 5 Scherer SECs,
the LLM rates **valence / arousal / dominance directly** (identity projection).

```python
from emotional_memory import DIRECT_VAD_SCHEMA, LLMAppraisalConfig
from emotional_memory.appraisal_llm import LLMAppraisalEngine

appraisal_engine = LLMAppraisalEngine(
    llm=make_httpx_llm_from_env(),
    config=LLMAppraisalConfig(appraisal_schema=DIRECT_VAD_SCHEMA),
)
```

Against human-annotated affect (EmoBank, N=300) direct-VAD is **better correlated on
every axis** than the default SEC→projection — valence r=0.79 (near-zero bias), arousal
r=0.58, dominance r=0.43, vs 0.70 / 0.23 / 0.31 (Addendum V). Choose it when you only
need `CoreAffect` and want stronger human-gold agreement.

**Caveats (why SEC remains the default).** Any custom schema — `DIRECT_VAD_SCHEMA`
included — produces a `GenericAppraisalVector`, not the Scherer `AppraisalVector`.
Consequently: the dual-path `elaborate()` slow path consumes its blended affect but does
**not** persist the appraisal on the tag (`tag.appraisal` stays `None`), and any feature
that reads the 5 SEC fields requires the default `SCHERER_CPM_SCHEMA`. Direct-VAD's
arousal _absolute scale_ is also less calibrated (higher MAE) than its correlation.

## API reference

[Appraisal Schema API](../api/appraisal_schema.md) · [Appraisal API](../api/appraisal.md)
