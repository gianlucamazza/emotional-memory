# Addendum AA — wrong-person memory retrieval

This is an exploratory stress test on the pinned ES-MemEval/EvoEmo corpus. It
transplants a real session from another seeker by changing only the speaker
prefix, then measures whether the resulting non-gold memory enters the top four.
The selection rule uses cosine similarity, never affect labels or AFT scores.

Run the no-LLM smoke test with `make bench-aa-false-memory-dry`. Run the full
200-query study with `make bench-aa-false-memory` after `make install-scored-bench`.
The full run needs `EMOTIONAL_MEMORY_LLM_API_KEY` and uses the project-default
`gpt-5-mini` unless the standard `EMOTIONAL_MEMORY_LLM_*` settings override it.

Successful LLM appraisals are saved to the ignored
`appraisals.checkpoint.jsonl` file. Rerun the **same command**, model, corpus,
and protocol to resume after interruption. The cache keys include model and
schema; do not delete the checkpoint until results have been reviewed. Final
`results.json` and `results.md` are written only after all 200 queries finish.

The result is a retrieval measurement. It does not test generated answers or
the natural frequency of false memories. See the committed
`../preregistration_addendum_aa_false_memory.md` for the full protocol.
