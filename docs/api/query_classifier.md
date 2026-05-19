# Query Classifier

The query classifier routes each retrieval query to a per-type weight profile,
enabling the engine to apply different retrieval strategies depending on whether
the query targets an emotion state, an affect-conditioned topic, or a temporal
trajectory.

See also: [tutorial — query routing](../tutorials/query_routing.md) for end-to-end usage,
and `LOCOMO_ROUTING` for the pre-registered weight table from Addendum L.

## Protocol

::: emotional_memory.query_classifier.QueryClassifier

## Built-in Classifiers

::: emotional_memory.query_classifier.HeuristicQueryClassifier

::: emotional_memory.query_classifier.LLMQueryClassifier

## Routing Tables

::: emotional_memory.query_classifier.LOCOMO_ROUTING

::: emotional_memory.query_classifier.QUERY_TYPES
