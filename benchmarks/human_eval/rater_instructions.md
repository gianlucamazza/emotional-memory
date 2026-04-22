# Human-Eval Pilot v1 Instructions

This pilot compares two retrieval conditions on the same scenario:

- `aft`
- `naive_cosine`

Do not interpret the condition names as quality labels. Rate only the material
shown in the packet.

## What you will rate

Each packet contains:

- one scenario prompt
- two conditions
- the top retrieved memory for each query under that condition

Rate each condition independently on four dimensions from `1` to `5`:

- `affective_coherence`: does the retrieval fit the emotional arc of the scenario?
- `usefulness`: would this retrieval help an agent respond well in context?
- `continuity`: does it preserve continuity with what happened across sessions?
- `plausibility`: does the retrieval feel believable for this scenario?

## Rating guidance

- `1` = clearly poor / misleading
- `3` = mixed or only partly convincing
- `5` = clearly strong and contextually appropriate

Use the note field when:

- a retrieval is partly right but incomplete
- one query card is much stronger or weaker than the others
- both conditions seem equally weak or equally strong for different reasons

## Procedure

1. Open the packet for one scenario.
2. Read the scenario prompt.
3. Review both conditions.
4. Rate each condition separately.
5. Repeat for all packets.

The canonical local flow is:

```bash
make bench-realistic
make human-eval-packets
# copy ratings_template.jsonl to ratings.jsonl and fill it
make human-eval-summary
```
