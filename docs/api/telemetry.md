# Telemetry

Optional OpenTelemetry tracing for the `emotional_memory` pipeline.

Install the `[otel]` extra to activate spans:

```bash
uv pip install "emotional-memory[otel]"
```

Without the extra, all helpers are **zero-overhead no-ops** — no import errors, no performance cost.

## Usage

```python
from emotional_memory.telemetry import traced_span

with traced_span("my.operation", attributes={"key": "value"}) as span:
    ...  # span is None when [otel] extra is absent
```

Use `traced_span` in your own code to add child spans inside the engine's root spans.
If the body raises, the exception is recorded on the span and status is set to `ERROR`
before re-raising.

## API

::: emotional_memory.telemetry.traced_span
