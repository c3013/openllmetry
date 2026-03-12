import json
import uuid

from langchain_core.agents import AgentAction
from langchain_core.tools import StructuredTool, tool
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


def _make_handler(tracer_provider):
    """Create a TraceloopCallbackHandler for testing."""
    from opentelemetry.sdk.metrics import MeterProvider

    tracer = tracer_provider.get_tracer("test")
    meter = MeterProvider().get_meter("test")
    return TraceloopCallbackHandler(
        tracer,
        meter.create_histogram("test.duration"),
        meter.create_histogram("test.tokens"),
    )


def test_tool_start_propagates_skill_name_to_parent_span(
    tracer_provider, span_exporter
):
    """Test that on_tool_start records the skill name on the parent (agent) span.

    This is the reliable path for all agent types (LCEL, legacy, LangGraph).
    When a tool runs, its skill name is propagated up to the parent agent span
    directly from on_tool_start, without depending on on_agent_action.
    """
    handler = _make_handler(tracer_provider)

    agent_run_id = uuid.uuid4()
    tool_run_id = uuid.uuid4()

    # Simulate on_chain_start for the AgentExecutor workflow span
    handler.on_chain_start(
        serialized={"name": "AgentExecutor", "id": ["AgentExecutor"]},
        inputs={"input": "test"},
        run_id=agent_run_id,
        parent_run_id=None,
    )

    # Simulate on_tool_start — parent_run_id points to the AgentExecutor span
    handler.on_tool_start(
        serialized={
            "name": "search_docs",
            "description": "Search internal documentation",
        },
        input_str="opentelemetry setup",
        run_id=tool_run_id,
        parent_run_id=agent_run_id,
    )

    # The agent span should have gen_ai.skill.name set via on_tool_start propagation
    agent_span = handler.spans[agent_run_id].span
    assert agent_span.attributes.get(SpanAttributes.GEN_AI_SKILL_NAME) == "search_docs", (
        "Expected gen_ai.skill.name on parent agent span to be set from on_tool_start"
    )

    # Finish tool and agent spans
    handler.on_tool_end(output="result", run_id=tool_run_id)
    handler.on_chain_end(outputs={"output": "done"}, run_id=agent_run_id)

    finished_spans = span_exporter.get_finished_spans()
    agent_spans = [s for s in finished_spans if s.name == "AgentExecutor.workflow"]
    assert len(agent_spans) == 1
    assert (
        agent_spans[0].attributes.get(SpanAttributes.GEN_AI_SKILL_NAME) == "search_docs"
    ), "Expected gen_ai.skill.name on finished AgentExecutor span"


def test_agent_action_records_skill_name_on_agent_span(
    tracer_provider, span_exporter
):
    """Test that on_agent_action records the selected skill name on the agent span.

    This path is for legacy langchain AgentExecutor agents only.
    For LCEL and LangGraph agents, use on_tool_start propagation instead.
    """
    handler = _make_handler(tracer_provider)

    agent_run_id = uuid.uuid4()

    # Simulate on_chain_start for the AgentExecutor (the agent span)
    handler.on_chain_start(
        serialized={"name": "AgentExecutor", "id": ["AgentExecutor"]},
        inputs={"input": "test"},
        run_id=agent_run_id,
        parent_run_id=None,
    )

    # Simulate on_agent_action (legacy agent path)
    action = AgentAction(
        tool="search_docs",
        tool_input="opentelemetry setup",
        log="I should search for documentation about OpenTelemetry setup.",
    )
    handler.on_agent_action(
        action,
        run_id=uuid.uuid4(),
        parent_run_id=agent_run_id,
    )

    # Verify that the agent span has the selected skill name recorded
    agent_span = handler.spans[agent_run_id].span
    assert agent_span.attributes.get(SpanAttributes.GEN_AI_SKILL_NAME) == "search_docs", (
        "Expected gen_ai.skill.name to be 'search_docs' on the agent span"
    )

    handler.on_chain_end(outputs={"output": "done"}, run_id=agent_run_id)

    finished_spans = span_exporter.get_finished_spans()
    agent_spans = [s for s in finished_spans if s.name == "AgentExecutor.workflow"]
    assert len(agent_spans) == 1
    assert (
        agent_spans[0].attributes.get(SpanAttributes.GEN_AI_SKILL_NAME) == "search_docs"
    ), "Expected gen_ai.skill.name on the finished AgentExecutor span"


def test_skill_metadata_recorded_on_tool_span(instrument_legacy, span_exporter):
    """Test that skill metadata is recorded on the tool span as gen_ai.skill.metadata."""
    skill = StructuredTool.from_function(
        func=lambda q: f"Processed: {q}",
        name="math_skill",
        description="A skill for math operations",
        metadata={"skill_type": "math", "version": "1.0", "author": "alice"},
    )
    skill.invoke("2+2")

    spans = span_exporter.get_finished_spans()
    tool_span = next((s for s in spans if s.name == "math_skill.tool"), None)
    assert tool_span is not None, "Expected math_skill.tool span"

    raw = tool_span.attributes.get(SpanAttributes.GEN_AI_SKILL_METADATA)
    assert raw is not None, "Expected gen_ai.skill.metadata to be set on tool span"
    metadata = json.loads(raw)
    assert metadata == {"skill_type": "math", "version": "1.0", "author": "alice"}, (
        "Expected gen_ai.skill.metadata to contain the tool's metadata"
    )


def test_skill_metadata_propagated_to_parent_span(tracer_provider, span_exporter):
    """Test that skill metadata is propagated to the parent (agent/workflow) span."""
    handler = _make_handler(tracer_provider)

    agent_run_id = uuid.uuid4()
    tool_run_id = uuid.uuid4()

    handler.on_chain_start(
        serialized={"name": "AgentExecutor", "id": ["AgentExecutor"]},
        inputs={"input": "compute"},
        run_id=agent_run_id,
        parent_run_id=None,
    )

    skill_metadata = {"skill_type": "math", "version": "2.0"}
    handler.on_tool_start(
        serialized={"name": "math_skill", "description": "A skill for math"},
        input_str="2+2",
        run_id=tool_run_id,
        parent_run_id=agent_run_id,
        metadata=skill_metadata,
    )

    # Skill metadata should be set on the parent agent span
    agent_span = handler.spans[agent_run_id].span
    raw = agent_span.attributes.get(SpanAttributes.GEN_AI_SKILL_METADATA)
    assert raw is not None, "Expected gen_ai.skill.metadata on parent agent span"
    assert json.loads(raw) == skill_metadata, (
        "Expected gen_ai.skill.metadata on parent span to match the tool's metadata"
    )

    handler.on_tool_end(output="4", run_id=tool_run_id)
    handler.on_chain_end(outputs={"output": "done"}, run_id=agent_run_id)

    finished_spans = span_exporter.get_finished_spans()
    agent_spans = [s for s in finished_spans if s.name == "AgentExecutor.workflow"]
    assert len(agent_spans) == 1
    raw = agent_spans[0].attributes.get(SpanAttributes.GEN_AI_SKILL_METADATA)
    assert raw is not None, "Expected gen_ai.skill.metadata on finished AgentExecutor span"
    assert json.loads(raw) == skill_metadata
