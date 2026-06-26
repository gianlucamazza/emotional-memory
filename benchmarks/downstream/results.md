# Downstream A3 — encode→retrieve→generate→judge (Addendum R)

Dataset: `realistic_recall_v2` v2.0 · embedder: `sbert-bge` · bootstrap n=2000 · seed=42 · judge=on.

Both systems share the generator, LLM and judge; only retrieval differs (full AFT vs embedding cosine). Gold = content of `expected_memory_ids`.

## Hypothesis tests

| Hyp | Metric | N | AFT | cosine | Δ [95% CI] | p | p_holm | McNemar | Result |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| **Hr1** | judge_correct | 200 | 0.5950 | 0.4400 | +0.1550 [+0.0950, +0.2200] | 0.0000 | 0.0000 | 0.0000 | **✅ PASS** |
| **Hr2** | f1 | 200 | 0.4927 | 0.3407 | +0.1520 [+0.1004, +0.2052] | 0.0000 | 0.0000 | 0.0000 | **✅ PASS** |

## Ranking reference (retrieval top-1, not a hypothesis)

AFT top1 0.5300 vs cosine 0.3250 · Δ +0.2050 [+0.1500, +0.2650] · shows whether the ranking edge converts to answer quality above.
