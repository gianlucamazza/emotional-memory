# Rater FAQ — AFT Human Evaluation Pilot

Answers to common questions from raters before and during the evaluation.

---

## What am I evaluating?

You are comparing two memory-retrieval conditions (`aft` and `naive_cosine`)
in a hypothetical AI assistant scenario. Both conditions use the same
conversation history; they differ in how past memories are selected when the
assistant needs to respond.

You are **not** asked to evaluate the assistant's generated response — only
which memories it retrieved.

---

## Do I need AI or NLP knowledge?

No. The evaluation is designed so that any thoughtful reader can judge whether
a retrieved memory fits the context. If you find a packet confusing, note it
in the `notes` field and rate as best you can.

---

## What does each rating dimension mean?

| Dimension | Short description |
|---|---|
| `affective_coherence` | Does the retrieved memory match the emotional tone of the query? |
| `usefulness` | Would this memory help an agent give a relevant response? |
| `continuity` | Does it preserve the thread of what happened across sessions? |
| `plausibility` | Does it feel like a believable thing to recall here? |

**Rating scale:** 1 = clearly poor, 3 = mixed or partial, 5 = clearly appropriate.

---

## What if both conditions seem equally good or bad?

Give them the same rating. There is no requirement that one must be better than
the other. Honest ratings are more valuable than rankings.

---

## What if the retrieved memory seems irrelevant to the query?

Rate it a 1 or 2 on the relevant dimensions (especially `usefulness` and
`affective_coherence`). Use the `notes` field to explain briefly.

---

## What if a condition retrieves the right memory but the phrasing is odd?

Rate the retrieval on the memory's content, not its phrasing. If the memory
content is relevant, rate it accordingly; if only the phrasing is off, note
it but don't penalize the `usefulness` or `coherence` dimensions.

---

## Can I change a rating after submitting it?

If you are using the JSONL file: yes — just overwrite the row. If you are using
Google Sheets: yes — just edit the cell. Please notify the coordinator if you
change a rating after the collection deadline.

---

## How long does it take?

Typically 20–40 minutes for 10 packets. The packets are short; the main effort
is reading the scenario context carefully before rating.

---

## Should I rate the conditions in order (aft first, then naive_cosine)?

No. Rate each condition independently — try not to compare them directly while
rating. Start with one condition, fill in all four dimensions, then move to the
other.

---

## What is `rater_id`?

A string that identifies you. Use a short pseudonym or initials you'll remember
(e.g., `"rater_a"` or `"jd"`). Do not use your full name unless you consent to
being credited.

---

## I found a scenario that seems broken or confusing. What should I do?

Rate it as best you can and add a note in the `notes` field describing the
problem. Do not skip it — partial ratings are more useful than missing data.
Report the issue to the coordinator after you finish.

---

## Is my participation anonymous?

Your `rater_id` is pseudonymous by default. Completed `ratings.jsonl` files
will be committed to a public repository as part of the reproducibility record.
If you prefer not to have your ratings published, let the coordinator know
before submitting.
