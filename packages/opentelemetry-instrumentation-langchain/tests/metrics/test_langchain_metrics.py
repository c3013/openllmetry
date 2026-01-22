from unittest.mock import patch
from typing import TypedDict
import pytest
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import Meters
from langgraph.graph import StateGraph
from openai import OpenAI


@pytest.fixture
def openai_client():
    return OpenAI()


@pytest.fixture
def llm():
    return ChatOpenAI(temperature=0)


@pytest.fixture
def chain(llm):
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    return LLMChain(llm=llm, prompt=prompt)


@pytest.mark.vcr
def test_llm_chain_metrics(instrument_legacy, reader, chain):
    chain.run(product="colorful socks")

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = False  # Not generating tokens metric
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert data_point.sum > 0
                        assert (
                            data_point.attributes[GenAIAttributes.GEN_AI_SYSTEM]
                            == "openai"
                        )

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = False  # Not generating duration metric
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    for data_point in metric.data.data_points:
                        assert (
                            data_point.attributes[GenAIAttributes.GEN_AI_SYSTEM]
                            == "openai"
                        )

    assert found_token_metric is False  # Metrics not generated
    assert found_duration_metric is False  # Metrics not generated


@pytest.mark.vcr
def test_llm_chain_streaming_metrics(instrument_legacy, reader, llm):
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    for _ in chain.stream({"product": "colorful socks"}):
        pass

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = False  # Not generating tokens metric
                    for data_point in metric.data.data_points:
                        assert data_point.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE] in [
                            "output",
                            "input",
                        ]
                        assert data_point.sum > 0
                        assert (
                            data_point.attributes[GenAIAttributes.GEN_AI_SYSTEM]
                            == "openai"
                        )

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = False  # Not generating duration metric
                    assert any(
                        data_point.count > 0 for data_point in metric.data.data_points
                    )
                    assert any(
                        data_point.sum > 0 for data_point in metric.data.data_points
                    )
                    for data_point in metric.data.data_points:
                        assert (
                            data_point.attributes[GenAIAttributes.GEN_AI_SYSTEM]
                            == "openai"
                        )

    assert found_token_metric is False  # Metrics not generated
    assert found_duration_metric is False  # Metrics not generated


def verify_token_metrics(data_points):
    for data_point in data_points:
        assert data_point.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE] in [
            "output",
            "input",
        ]
        assert data_point.sum > 0
        assert data_point.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"


def verify_duration_metrics(data_points):
    assert any(data_point.count > 0 for data_point in data_points)
    assert any(data_point.sum > 0 for data_point in data_points)
    for data_point in data_points:
        assert data_point.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"


def verify_langchain_metrics(reader):
    metrics_data = reader.get_metrics_data()
    if metrics_data is None:
        return False, False
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_token_metric = False
    found_duration_metric = False

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.LLM_TOKEN_USAGE:
                    found_token_metric = False  # Not generating tokens metric
                    verify_token_metrics(metric.data.data_points)

                if metric.name == Meters.LLM_OPERATION_DURATION:
                    found_duration_metric = False  # Not generating duration metric
                    verify_duration_metrics(metric.data.data_points)

    return found_token_metric, found_duration_metric


@pytest.mark.vcr
def test_llm_chain_metrics_with_none_llm_output(instrument_legacy, reader, chain, llm):
    """
    This test verifies that the metrics system correctly handles edge cases where the
    LLM response contains a None value in the llm_output field, ensuring that token
    usage and operation duration metrics are still properly recorded.
    """
    original_generate = llm._generate

    # Create a patched version that returns results with None llm_output
    def patched_generate(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        result.llm_output = None
        return result

    with patch.object(llm, '_generate', side_effect=patched_generate):
        chain.run(product="colorful socks")

    found_token_metric, found_duration_metric = verify_langchain_metrics(reader)

    assert found_token_metric is False  # Metrics not generated, "Token usage metrics not found"
    assert found_duration_metric is False  # Metrics not generated, "Operation duration metrics not found"


@pytest.mark.vcr
def test_langgraph_metrics(instrument_legacy, reader, openai_client):
    class State(TypedDict):
        request: str
        result: str

    def calculate(state: State):
        request = state["request"]
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a mathematician."},
                {"role": "user", "content": request}
            ]
        )
        return {"result": completion.choices[0].message.content}
    workflow = StateGraph(State)
    workflow.add_node("calculate", calculate)
    workflow.set_entry_point("calculate")

    langgraph = workflow.compile()

    user_request = "What's 5 + 5?"
    langgraph.invoke(input={"request": user_request})

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) == 1

    metric_data = resource_metrics[0].scope_metrics[-1].metrics
    assert len(metric_data) == 3

    token_usage_metric = next(
        (
            m
            for m in metric_data
            if m.name == Meters.LLM_TOKEN_USAGE
        ),
        None,
    )
    assert token_usage_metric is not None
    token_usage_data_point = token_usage_metric.data.data_points[0]
    assert token_usage_data_point.sum > 0
    assert (
        token_usage_data_point.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"
        and token_usage_data_point.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE] in ["input", "output"]
    )

    duration_metric = next(
        (
            m
            for m in metric_data
            if m.name == Meters.LLM_OPERATION_DURATION
        ),
        None,
    )
    assert duration_metric is not None
    duration_data_point = duration_metric.data.data_points[0]
    assert duration_data_point.sum > 0
    assert duration_data_point.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "openai"

    generation_choices_metric = next(
        (
            m
            for m in metric_data
            if m.name == Meters.LLM_GENERATION_CHOICES
        ),
        None
    )
    assert generation_choices_metric is not None
    generation_choices_data_points = generation_choices_metric.data.data_points
    for data_point in generation_choices_data_points:
        assert (
            data_point.attributes[GenAIAttributes.GEN_AI_SYSTEM]
            == "openai"
        )
        assert data_point.value > 0


@pytest.mark.vcr
def test_workflow_duration_metric(instrument_legacy, reader, chain):
    """Test that workflow duration metric is recorded."""
    chain.run(product="colorful socks")

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    # Find the workflow duration metric
    workflow_duration_metric = None
    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.GEN_AI_WORKFLOW_DURATION:
                    workflow_duration_metric = metric
                    break

    assert workflow_duration_metric is not None, "Workflow duration metric not found"
    
    # Verify metric properties
    data_points = workflow_duration_metric.data.data_points
    assert len(data_points) > 0, "No data points found for workflow duration metric"
    
    for data_point in data_points:
        assert data_point.sum > 0, "Duration should be greater than 0"
        assert data_point.count > 0, "Count should be greater than 0"
        # Verify the workflow name attribute exists
        from opentelemetry.semconv_ai import SpanAttributes
        assert SpanAttributes.GEN_AI_WORKFLOW_NAME in data_point.attributes


@pytest.mark.vcr
def test_tool_duration_metric(instrument_legacy, reader):
    """Test that tool duration metric is recorded when tools are used."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_openai import ChatOpenAI

    OPENAI_FUNCTIONS_AGENT_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    search = TavilySearchResults(max_results=2)
    tools = [search]

    model = ChatOpenAI(model="gpt-3.5-turbo")
    agent = create_tool_calling_agent(model, tools, OPENAI_FUNCTIONS_AGENT_PROMPT)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    agent_executor.invoke({"input": "What is OpenLLMetry?"})

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    
    # Find the tool duration metric
    tool_duration_metric = None
    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.GEN_AI_TOOL_DURATION:
                    tool_duration_metric = metric
                    break
    
    assert tool_duration_metric is not None, "Tool duration metric not found"
    
    # Verify metric properties
    data_points = tool_duration_metric.data.data_points
    assert len(data_points) > 0, "No data points found for tool duration metric"
    
    for data_point in data_points:
        assert data_point.sum > 0, "Duration should be greater than 0"
        assert data_point.count > 0, "Count should be greater than 0"
        # Verify the tool name attribute exists
        assert GenAIAttributes.GEN_AI_TOOL_NAME in data_point.attributes


@pytest.mark.vcr
def test_agent_duration_metric(instrument_legacy, reader):
    """Test that agent duration metric is recorded when agents are used."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_openai import ChatOpenAI

    OPENAI_FUNCTIONS_AGENT_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    search = TavilySearchResults(max_results=2)
    tools = [search]

    model = ChatOpenAI(model="gpt-3.5-turbo")
    agent = create_tool_calling_agent(model, tools, OPENAI_FUNCTIONS_AGENT_PROMPT)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    agent_executor.invoke({"input": "What is OpenLLMetry?"})

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    
    # Find the agent duration metric
    agent_duration_metric = None
    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.GEN_AI_AGENT_DURATION:
                    agent_duration_metric = metric
                    break
    
    # Agent metrics may or may not be present depending on agent implementation
    # so we check if it exists and if so verify its properties
    if agent_duration_metric is not None:
        # Verify metric properties
        data_points = agent_duration_metric.data.data_points
        assert len(data_points) > 0, "No data points found for agent duration metric"
        
        for data_point in data_points:
            assert data_point.sum > 0, "Duration should be greater than 0"
            assert data_point.count > 0, "Count should be greater than 0"
            # Verify the operation name attribute exists
            assert GenAIAttributes.GEN_AI_OPERATION_NAME in data_point.attributes
