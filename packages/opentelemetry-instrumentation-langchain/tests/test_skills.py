import uuid

from langchain_core.agents import AgentAction
from langchain_core.tools import tool
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.instrumentation.langchain.callback_handler import (
    TraceloopCallbackHandler,
)


@tool
def my_skill(query: str) -> str:
    """A skill that processes a query and returns useful information"""
    return f"Processed: {query}"


@tool
def another_skill(text: str) -> str:
    """Another skill for text analysis"""
    return f"Analyzed: {text}"


def test_skill_instrumentation(instrument_legacy, span_exporter):
    """Test that skill (tool) invocations set the correct skill attributes."""
    my_skill.invoke("test query")

    spans = span_exporter.get_finished_spans()

    tool_spans = [span for span in spans if span.name == "my_skill.tool"]
    assert len(tool_spans) == 1, f"Expected 1 my_skill.tool span, got {len(tool_spans)}"

    tool_span = tool_spans[0]

    assert (
        tool_span.attributes.get(SpanAttributes.GEN_AI_OPERATION_NAME) == "load_skill"
    ), "Expected gen_ai.operation.name to be 'load_skill'"

    assert (
        tool_span.attributes.get(SpanAttributes.GEN_AI_SKILL_NAME) == "my_skill"
    ), "Expected gen_ai.skill.name to be 'my_skill'"

    assert tool_span.attributes.get(SpanAttributes.GEN_AI_SKILL_INFO) == (
        "A skill that processes a query and returns useful information"
    ), "Expected gen_ai.skill.info to contain the skill description"


def test_skill_name_attribute(instrument_legacy, span_exporter):
    """Test that gen_ai.skill.name matches the tool name."""
    another_skill.invoke("some text")

    spans = span_exporter.get_finished_spans()

    tool_span = next(
        (span for span in spans if span.name == "another_skill.tool"), None
    )
    assert tool_span is not None, "Expected another_skill.tool span"

    assert (
        tool_span.attributes.get(SpanAttributes.GEN_AI_SKILL_NAME) == "another_skill"
    ), "Expected gen_ai.skill.name to be 'another_skill'"
    assert (
        tool_span.attributes.get(SpanAttributes.GEN_AI_SKILL_INFO)
        == "Another skill for text analysis"
    ), "Expected gen_ai.skill.info to match the skill description"
    assert (
        tool_span.attributes.get(SpanAttributes.GEN_AI_OPERATION_NAME) == "load_skill"
    ), "Expected gen_ai.operation.name to be 'load_skill'"


def test_agent_action_records_skill_name_on_agent_span(
    tracer_provider, span_exporter
):
    """Test that on_agent_action records the selected skill name on the agent span.

    When an agent selects a skill to use, the skill name should be recorded
    at the agent (create_deep_agent) layer, not just in the tool's own span.
    """
    from opentelemetry.sdk.metrics import MeterProvider

    tracer = tracer_provider.get_tracer("test")
    meter_provider = MeterProvider()
    meter = meter_provider.get_meter("test")
    duration_histogram = meter.create_histogram("test.duration")
    token_histogram = meter.create_histogram("test.tokens")

    handler = TraceloopCallbackHandler(tracer, duration_histogram, token_histogram)

    agent_run_id = uuid.uuid4()

    # Simulate on_chain_start for the AgentExecutor (the agent span)
    handler.on_chain_start(
        serialized={"name": "AgentExecutor", "id": ["AgentExecutor"]},
        inputs={"input": "test"},
        run_id=agent_run_id,
        parent_run_id=None,
    )

    # Simulate on_agent_action (when agent selects a skill at the create_deep_agent layer)
    action = AgentAction(
        tool="search_docs",
        tool_input="opentelemetry setup",
        log="I should search for documentation about OpenTelemetry setup.",
    )
    action_run_id = uuid.uuid4()
    handler.on_agent_action(
        action,
        run_id=action_run_id,
        parent_run_id=agent_run_id,
    )

    # Verify that the agent span has the selected skill name recorded
    agent_span_holder = handler.spans[agent_run_id]
    agent_span = agent_span_holder.span

    assert agent_span.attributes.get(SpanAttributes.GEN_AI_SKILL_NAME) == "search_docs", (
        "Expected gen_ai.skill.name to be 'search_docs' on the agent span"
    )

    # Clean up - end the agent chain span
    handler.on_chain_end(
        outputs={"output": "done"},
        run_id=agent_run_id,
    )

    finished_spans = span_exporter.get_finished_spans()
    agent_spans = [s for s in finished_spans if s.name == "AgentExecutor.workflow"]
    assert len(agent_spans) == 1
    assert (
        agent_spans[0].attributes.get(SpanAttributes.GEN_AI_SKILL_NAME) == "search_docs"
    ), "Expected gen_ai.skill.name on the finished AgentExecutor span"
