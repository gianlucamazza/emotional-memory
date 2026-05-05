"""Tests for OpenTelemetry tracing helpers (requires [otel] extra).

Run with:
    pytest tests/test_telemetry.py -v
"""

from __future__ import annotations

import pytest
from conftest import FixedEmbedder

pytest.importorskip("opentelemetry")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import emotional_memory.telemetry as telemetry_mod
from emotional_memory import EmotionalMemory
from emotional_memory.stores.in_memory import InMemoryStore
from emotional_memory.telemetry import traced_span

# ---------------------------------------------------------------------------
# Fixture: fresh TracerProvider + InMemorySpanExporter per test
# ---------------------------------------------------------------------------


@pytest.fixture()
def span_exporter() -> InMemorySpanExporter:
    """Inject a fresh TracerProvider+exporter directly into telemetry_mod._tracer.

    OTel's global set_tracer_provider() can only be called once per process, so
    we bypass it by writing the test tracer into the module cache directly.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Direct injection — no global provider swap needed.
    telemetry_mod._tracer = provider.get_tracer(telemetry_mod._TRACER_NAME)
    telemetry_mod._otel_unavailable = False

    yield exporter

    telemetry_mod._reset_tracer_cache()


def _span_names(exporter: InMemorySpanExporter) -> list[str]:
    return [s.name for s in exporter.get_finished_spans()]


# ---------------------------------------------------------------------------
# telemetry module unit tests
# ---------------------------------------------------------------------------


class TestTracedSpanNoOp:
    def test_no_op_when_unavailable(self):
        """traced_span is a no-op when OTel is unavailable (mocked via cache reset)."""
        telemetry_mod._reset_tracer_cache()
        telemetry_mod._otel_unavailable = True

        with traced_span("test.noop") as span:
            assert span is None

        telemetry_mod._reset_tracer_cache()

    def test_no_op_does_not_raise_on_body_exception(self):
        telemetry_mod._reset_tracer_cache()
        telemetry_mod._otel_unavailable = True

        with pytest.raises(ValueError, match="boom"), traced_span("test.noop"):
            raise ValueError("boom")

        telemetry_mod._reset_tracer_cache()


class TestTracedSpanWithOTel:
    def test_emits_span(self, span_exporter: InMemorySpanExporter):
        with traced_span("my.operation"):
            pass
        assert "my.operation" in _span_names(span_exporter)

    def test_span_attributes(self, span_exporter: InMemorySpanExporter):
        with traced_span("op.with.attrs", {"key": "value", "count": 42}):
            pass
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes is not None
        assert spans[0].attributes["key"] == "value"
        assert spans[0].attributes["count"] == 42

    def test_nested_spans(self, span_exporter: InMemorySpanExporter):
        with traced_span("outer"), traced_span("inner"):
            pass
        names = _span_names(span_exporter)
        assert "outer" in names
        assert "inner" in names
        # inner finishes before outer
        assert names.index("inner") < names.index("outer")

    def test_error_recorded_on_exception(self, span_exporter: InMemorySpanExporter):
        from opentelemetry.trace import StatusCode

        with pytest.raises(RuntimeError), traced_span("failing.op"):
            raise RuntimeError("oops")

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].status.status_code == StatusCode.ERROR

    def test_exception_propagates(self, span_exporter: InMemorySpanExporter):
        with pytest.raises(ValueError, match="propagated"), traced_span("will.fail"):
            raise ValueError("propagated")


# ---------------------------------------------------------------------------
# Engine integration: encode + retrieve emit spans
# ---------------------------------------------------------------------------


class TestEngineSpans:
    def _make_engine(self) -> EmotionalMemory:
        return EmotionalMemory(
            store=InMemoryStore(),
            embedder=FixedEmbedder([1.0, 0.0, 0.0]),
        )

    def test_encode_emits_root_span(self, span_exporter: InMemorySpanExporter):
        em = self._make_engine()
        em.encode("hello world")
        assert "emotional_memory.encode" in _span_names(span_exporter)

    def test_encode_emits_embed_child_span(self, span_exporter: InMemorySpanExporter):
        em = self._make_engine()
        em.encode("hello world")
        assert "emotional_memory.embed" in _span_names(span_exporter)

    def test_retrieve_emits_root_span(self, span_exporter: InMemorySpanExporter):
        em = self._make_engine()
        em.encode("hello world")
        em.retrieve("hello", top_k=1)
        assert "emotional_memory.retrieve" in _span_names(span_exporter)

    def test_retrieve_emits_embed_child_span(self, span_exporter: InMemorySpanExporter):
        em = self._make_engine()
        em.encode("seeded memory")
        span_exporter.clear()
        em.retrieve("query", top_k=1)
        names = _span_names(span_exporter)
        assert "emotional_memory.retrieve" in names
        assert "emotional_memory.embed" in names

    def test_observe_emits_span(self, span_exporter: InMemorySpanExporter):
        em = self._make_engine()
        em.observe("just an event")
        assert "emotional_memory.observe" in _span_names(span_exporter)

    def test_encode_batch_emits_root_and_embed_span(self, span_exporter: InMemorySpanExporter):
        em = self._make_engine()
        em.encode_batch(["a", "b", "c"])
        names = _span_names(span_exporter)
        assert "emotional_memory.encode_batch" in names
        assert "emotional_memory.embed" in names

    def test_prune_emits_span(self, span_exporter: InMemorySpanExporter):
        em = self._make_engine()
        em.encode("old memory")
        span_exporter.clear()
        em.prune(threshold=0.0)
        assert "emotional_memory.prune" in _span_names(span_exporter)

    def test_elaborate_emits_span(self, span_exporter: InMemorySpanExporter):
        em = self._make_engine()
        mem = em.encode("some content")
        span_exporter.clear()
        em.elaborate(mem.id)
        assert "emotional_memory.elaborate" in _span_names(span_exporter)

    def test_encode_span_has_content_length_attribute(self, span_exporter: InMemorySpanExporter):
        em = self._make_engine()
        em.encode("hello")
        encode_spans = [
            s for s in span_exporter.get_finished_spans() if s.name == "emotional_memory.encode"
        ]
        assert len(encode_spans) == 1
        assert encode_spans[0].attributes is not None
        assert encode_spans[0].attributes["content_length"] == 5

    def test_retrieve_span_has_top_k_attribute(self, span_exporter: InMemorySpanExporter):
        em = self._make_engine()
        em.encode("memory")
        span_exporter.clear()
        em.retrieve("query", top_k=3)
        retrieve_spans = [
            s for s in span_exporter.get_finished_spans() if s.name == "emotional_memory.retrieve"
        ]
        assert len(retrieve_spans) == 1
        assert retrieve_spans[0].attributes is not None
        assert retrieve_spans[0].attributes["top_k"] == 3

    def test_no_spans_when_otel_unavailable(self):
        """Engine works normally when OTel is not configured (no exporter)."""
        telemetry_mod._reset_tracer_cache()
        telemetry_mod._otel_unavailable = True
        em = self._make_engine()
        mem = em.encode("no tracing")
        results = em.retrieve("no tracing", top_k=1)
        assert len(results) == 1
        assert mem.content == "no tracing"
        telemetry_mod._reset_tracer_cache()
