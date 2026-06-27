# Addendum T — retrieve-time query appraisal vs oracle state-injection

Dataset: `realistic_recall_v2` v2.0 · embedder: `sbert-bge` · N=200 · bootstrap n=2000 · seed=42.

Three arms; only the query-affect source differs. `aft_query_appraised` injects the query's affect appraised by direct-VAD instead of the oracle `query.state`.

## Top-1 (full set)

- **aft_oracle vs cosine** (upper bound): 0.5200 vs 0.3250 · Δ +0.1950 [+0.1450, +0.2500] · p=0.0000
- **aft_query_appraised vs cosine** (Ht1): 0.4400 vs 0.3250 · Δ +0.1150 [+0.0550, +0.1800] · p=0.0005
- aft_query_appraised vs aft_oracle (gap): 0.4400 vs 0.5200 · Δ -0.0800 [-0.1250, -0.0400] · p=0.0000

**Recovery fraction** (appraised minus cosine)/(oracle minus cosine): **0.5897**

**Ht1:** ✅ PASS — query appraisal beats cosine (production-reachable)

## Affect-favorable subset (Addendum U criterion)

- N = 125
- aft_oracle vs cosine: 0.3040 vs 0.0000 · Δ +0.3040 [+0.2240, +0.3840] · p=0.0000
- aft_query_appraised vs cosine: 0.2480 vs 0.0000 · Δ +0.2480 [+0.1680, +0.3200] · p=0.0000

## Diagnostic — corr(appraised query affect, oracle state)

- valence r = 0.8024 · arousal r = 0.5581
