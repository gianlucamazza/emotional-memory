# Human Evaluation Pilot Protocol

## Question being answered

> Do human raters perceive better affective coherence and memory usefulness in
> selected scenarios when AFT-backed recall is available?

## Pilot scope

- 10–20 scenarios in the first live run (minimum 50 for publication-grade α)
- two or more system conditions if available
- blind or lightly blinded presentation where possible
- **minimum 3 raters** required for inter-rater agreement (Krippendorff's alpha)

## Rating dimensions

- **Affective coherence**: does the recalled context fit the emotional arc?
- **Usefulness**: does the recalled memory help with the scenario?
- **Continuity**: does the recalled memory feel consistent across sessions?
- **Plausibility**: does the behavior feel like a coherent memory process?

Use a 1–5 Likert scale for each dimension.

## Inter-rater agreement target

`make human-eval-summary` computes Krippendorff's alpha (ordinal) per dimension
from completed multi-rater rating files.

- Acceptable agreement: alpha ≥ 0.67 (Krippendorff 2004)
- Strong agreement: alpha ≥ 0.80
- Minimum raters: 3 (alpha is undefined with fewer than 2 raters)

## Minimum reporting

- scenario identifier
- system condition
- rater identifier or anonymized code
- scores on all four dimensions
- free-text note for disagreement or ambiguity

## Caveat

This pilot is a preparatory study asset. It does not substitute for a larger
ecological or behavioral validation program.
