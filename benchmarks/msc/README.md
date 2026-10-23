# MSC external human-grounded study

**Implemented preparation and evaluation tools; scientific study not executed.**

- [Protocol](protocol.md): single primary comparison, human gold, power and stopping rules.
- [Accepted decision](../../docs/research/decisions/0001-external-human-retrieval.md).
- [Verified source inventory](source_inventory.json): native archive and member hashes,
  5,001 grouped histories; no system outcomes or human labels.
- [Configuration template](config.example.json): intentionally refuses live execution
  until actual model revisions, provider configuration and package versions are pinned.

```bash
uv run python -m benchmarks.msc.cli --help
uv run python -m pytest tests/test_msc_study.py
```

Preparation was exercised on the full checksum-verified MSC v0.1 archive. The pilot
packet export produces 1,608 candidate cards per rater, with no filled ratings or
system labels. Synthetic tests exercise the real AFT engine and contract doubles
for external services; they do not validate live LLM or Mem0 performance.

The Mem0 secondary reference preserves the original annotated exchanges
(`infer=False`); it is not a test of native fact extraction/consolidation. The
primary comparator is the development-tuned semantic + recency + valence reranker.

Before scientific execution: verify dataset terms, recruit independent panels,
record measured annotation time/cost, complete development and pilot, compute N,
then freeze and commit registration. No human results or frozen confirmatory
registration have been created. Retain raw human submissions and private packet
keys separately; do not send the keys to raters.

Local preparation artifacts for the 2026-10-04 implementation check are in
`/tmp/emotional-memory-msc-preparation` and
`/tmp/emotional-memory-msc-pilot-packets-v2`. They are temporary, unscored and
reproducible using the protocol commands. No raw corpus is committed here.
