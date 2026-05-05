# Observability (OpenTelemetry)

`emotional_memory` emits OpenTelemetry spans on every engine operation when the `[otel]`
extra is installed. Without it, all tracing code is a zero-overhead no-op.

## Installation

```bash
uv pip install "emotional-memory[otel]"
# plus an exporter of your choice:
uv pip install opentelemetry-exporter-otlp        # OTLP (Jaeger, Grafana Tempo, …)
uv pip install opentelemetry-exporter-zipkin      # Zipkin
```

## Quick setup — console exporter

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

from opentelemetry import trace
trace.set_tracer_provider(provider)

# From this point all emotional_memory engine calls emit spans automatically
from emotional_memory import EmotionalMemory, InMemoryStore
from emotional_memory.embedders import SentenceTransformerEmbedder

em = EmotionalMemory(store=InMemoryStore(), embedder=SentenceTransformerEmbedder())
em.encode("Today's meeting went well.")        # emits emotional_memory.encode root span
results = em.retrieve("meeting", top_k=3)     # emits emotional_memory.retrieve root span
```

## OTLP / Jaeger

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317")))

from opentelemetry import trace
trace.set_tracer_provider(provider)
```

## Spans emitted

| Span name | Operation | Key attributes |
|---|---|---|
| `emotional_memory.encode` | `em.encode()` | `content_length` |
| `emotional_memory.encode_batch` | `em.encode_batch()` | `batch_size` |
| `emotional_memory.retrieve` | `em.retrieve()` | `top_k` |
| `emotional_memory.elaborate` | `em.elaborate()` | `memory_id` |
| `emotional_memory.observe` | `em.observe()` | `content_length` |
| `emotional_memory.prune` | `em.prune()` | `threshold` |
| `emotional_memory.embed` | embedding call (child) | — |
| `emotional_memory.store.search_by_embedding` | ANN search (child) | — |

Child spans (`embed`, `store.search_by_embedding`) appear nested inside the root span
for the enclosing operation.

## Error recording

If the engine body raises, the exception is recorded on the active span and span status
is set to `ERROR` before re-raising. No information is swallowed.

## Testing with InMemorySpanExporter

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import emotional_memory.telemetry as telemetry_mod

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))

# Inject the tracer directly (bypass the one-shot set_tracer_provider() constraint)
telemetry_mod._tracer = provider.get_tracer(telemetry_mod._TRACER_NAME)

# ... run your code ...

spans = exporter.get_finished_spans()
assert any(s.name == "emotional_memory.encode" for s in spans)

# cleanup between tests
telemetry_mod._reset_tracer_cache()
```

## See also

- [`traced_span` API reference](../api/telemetry.md)
- [OpenTelemetry Python SDK](https://opentelemetry-python.readthedocs.io/)
