# Logging

`configure_logging` is a top-level convenience for the package logger
(`emotional_memory.*`). Engine modules log pipeline events at `DEBUG`;
`warnings.warn` is reserved for user-visible degradation.

See the [observability tutorial](../tutorials/observability.md) for structured
logs plus OpenTelemetry spans.

```python
from emotional_memory import configure_logging

configure_logging(level="DEBUG")
configure_logging(level="INFO", json_format=True)
```

With no argument, the level comes from `EMOTIONAL_MEMORY_LOG_LEVEL` (default
`WARNING`).

::: emotional_memory.logging_config.configure_logging
