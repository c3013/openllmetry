"""Test configuration module."""

import pytest
from opentelemetry.instrumentation.mcp import McpInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(scope="session", name="span_exporter")
def fixture_span_exporter():
    exporter = InMemorySpanExporter()
    yield exporter


@pytest.fixture(scope="session", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    yield provider
    provider.shutdown()


@pytest.fixture(scope="session", name="reader")
def fixture_reader():
    reader = InMemoryMetricReader()
    yield reader


@pytest.fixture(scope="session", name="meter_provider")
def fixture_meter_provider(reader):
    provider = MeterProvider(metric_readers=[reader])
    yield provider
    provider.shutdown()


@pytest.fixture(autouse=True)
def instrument_mcp(tracer_provider, meter_provider, span_exporter, reader):
    instrumenter = McpInstrumentor()
    instrumenter.instrument(
        tracer_provider=tracer_provider, meter_provider=meter_provider
    )
    try:
        yield
    finally:
        instrumenter.uninstrument()
        span_exporter.clear()
