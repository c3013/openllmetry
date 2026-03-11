from langchain_core.tools import tool
from opentelemetry.semconv_ai import SpanAttributes


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
