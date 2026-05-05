"""OpenTelemetry tracing helpers (optional).

Lazily imports `opentelemetry`. When the ``[otel]`` extra is not installed,
all helpers become no-ops with zero overhead beyond a single attribute check.
When installed, the caller configures their own ``TracerProvider`` and the
spans emitted by this module flow through it.

Usage:

    from emotional_memory.telemetry import traced_span

    with traced_span("emotional_memory.encode", attributes={"len": 42}):
        ...

If the body raises, the exception is recorded on the span and span status is
set to ERROR before re-raising.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator, Mapping
from typing import Any

_TRACER_NAME = "emotional_memory"
_tracer: Any | None = None
_otel_unavailable = False


def _get_tracer() -> Any | None:
    """Return the OTel tracer, or ``None`` if the ``[otel]`` extra is missing."""
    global _tracer, _otel_unavailable
    if _tracer is not None:
        return _tracer
    if _otel_unavailable:
        return None
    try:
        trace_module = importlib.import_module("opentelemetry.trace")
    except ImportError:
        _otel_unavailable = True
        return None
    _tracer = trace_module.get_tracer(_TRACER_NAME)
    return _tracer


def _reset_tracer_cache() -> None:
    """Reset the cached tracer — for tests that swap TracerProvider mid-run."""
    global _tracer, _otel_unavailable
    _tracer = None
    _otel_unavailable = False


@contextlib.contextmanager
def traced_span(
    name: str,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Emit an OTel span if the ``[otel]`` extra is installed; otherwise no-op."""
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return

    attrs = dict(attributes) if attributes else None
    with tracer.start_as_current_span(name, attributes=attrs) as span:
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            trace_module = importlib.import_module("opentelemetry.trace")
            span.set_status(trace_module.Status(trace_module.StatusCode.ERROR, str(exc)))
            raise
