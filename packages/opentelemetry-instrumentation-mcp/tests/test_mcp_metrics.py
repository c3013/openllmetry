"""Tests for MCP metrics implementation."""

import pytest
from opentelemetry.semconv_ai import Meters


@pytest.mark.asyncio
async def test_client_operation_duration_metric(reader) -> None:
    """Test that mcp.client.operation.duration histogram is recorded for MCP operations."""
    from fastmcp import FastMCP, Client

    server = FastMCP("metrics-test-server")

    @server.tool()
    async def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    async with Client(server) as client:
        await client.list_tools()
        await client.call_tool("add_numbers", {"a": 1, "b": 2})

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_client_op_metric = False
    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.MCP_CLIENT_OPERATION_DURATION:
                    found_client_op_metric = True
                    assert len(metric.data.data_points) > 0
                    for dp in metric.data.data_points:
                        assert dp.count > 0
                        assert dp.sum >= 0
                        assert "mcp.method.name" in dp.attributes

    assert found_client_op_metric, (
        f"Expected metric '{Meters.MCP_CLIENT_OPERATION_DURATION}' was not found. "
        f"Available metrics: {[m.name for rm in resource_metrics for sm in rm.scope_metrics for m in sm.metrics]}"
    )


@pytest.mark.asyncio
async def test_client_session_duration_metric(reader) -> None:
    """Test that mcp.client.session.duration histogram is recorded for MCP sessions."""
    from fastmcp import FastMCP, Client

    server = FastMCP("session-metrics-server")

    @server.tool()
    async def ping() -> str:
        """Ping tool."""
        return "pong"

    async with Client(server) as client:
        await client.list_tools()

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_session_metric = False
    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.MCP_CLIENT_SESSION_DURATION:
                    found_session_metric = True
                    assert len(metric.data.data_points) > 0
                    for dp in metric.data.data_points:
                        assert dp.count > 0
                        assert dp.sum >= 0

    assert found_session_metric, (
        f"Expected metric '{Meters.MCP_CLIENT_SESSION_DURATION}' was not found. "
        f"Available metrics: {[m.name for rm in resource_metrics for sm in rm.scope_metrics for m in sm.metrics]}"
    )


@pytest.mark.asyncio
async def test_server_operation_duration_metric(reader) -> None:
    """Test that mcp.server.operation.duration histogram is recorded for server-side tool calls."""
    from fastmcp import FastMCP, Client

    server = FastMCP("server-op-metrics-server")

    @server.tool()
    async def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    async with Client(server) as client:
        await client.call_tool("multiply", {"a": 3, "b": 4})

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_server_op_metric = False
    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.MCP_SERVER_OPERATION_DURATION:
                    found_server_op_metric = True
                    assert len(metric.data.data_points) > 0
                    for dp in metric.data.data_points:
                        assert dp.count > 0
                        assert dp.sum >= 0
                        assert "mcp.method.name" in dp.attributes
                        assert dp.attributes["mcp.method.name"] == "tools/call"
                        assert "gen_ai.tool.name" in dp.attributes

    assert found_server_op_metric, (
        f"Expected metric '{Meters.MCP_SERVER_OPERATION_DURATION}' was not found. "
        f"Available metrics: {[m.name for rm in resource_metrics for sm in rm.scope_metrics for m in sm.metrics]}"
    )


@pytest.mark.asyncio
async def test_server_session_duration_metric(reader) -> None:
    """Test that mcp.server.session.duration histogram is recorded."""
    from fastmcp import FastMCP, Client

    server = FastMCP("server-session-metrics-server")

    @server.tool()
    async def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    async with Client(server) as client:
        await client.call_tool("greet", {"name": "World"})

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) > 0

    found_server_session_metric = False
    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.MCP_SERVER_SESSION_DURATION:
                    found_server_session_metric = True
                    assert len(metric.data.data_points) > 0
                    for dp in metric.data.data_points:
                        assert dp.count > 0
                        assert dp.sum >= 0

    assert found_server_session_metric, (
        f"Expected metric '{Meters.MCP_SERVER_SESSION_DURATION}' was not found. "
        f"Available metrics: {[m.name for rm in resource_metrics for sm in rm.scope_metrics for m in sm.metrics]}"
    )


@pytest.mark.asyncio
async def test_tool_call_metric_has_tool_name(reader) -> None:
    """Test that the mcp.client.operation.duration metric for tools/call has gen_ai.tool.name."""
    from fastmcp import FastMCP, Client

    server = FastMCP("tool-name-metrics-server")

    @server.tool()
    async def subtract(a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b

    async with Client(server) as client:
        await client.call_tool("subtract", {"a": 10, "b": 3})

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    found_subtract_dp = False
    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == Meters.MCP_CLIENT_OPERATION_DURATION:
                    for dp in metric.data.data_points:
                        if (
                            dp.attributes.get("mcp.method.name") == "tools/call"
                            and dp.attributes.get("gen_ai.tool.name") == "subtract"
                        ):
                            found_subtract_dp = True
                            assert dp.count > 0

    assert found_subtract_dp, (
        "Expected a tools/call data point with gen_ai.tool.name='subtract' in "
        f"'{Meters.MCP_CLIENT_OPERATION_DURATION}' metric"
    )


@pytest.mark.asyncio
async def test_all_mcp_metrics_are_histograms(reader) -> None:
    """Test that all 4 MCP metrics are recorded as histograms."""
    from fastmcp import FastMCP, Client
    from opentelemetry.sdk.metrics.export import (
        AggregationTemporality,
        MetricExportResult,
    )

    server = FastMCP("histogram-type-server")

    @server.tool()
    async def echo(msg: str) -> str:
        """Echo a message."""
        return msg

    async with Client(server) as client:
        await client.call_tool("echo", {"msg": "hello"})

    metrics_data = reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    mcp_metric_names = {
        Meters.MCP_CLIENT_OPERATION_DURATION,
        Meters.MCP_SERVER_OPERATION_DURATION,
        Meters.MCP_CLIENT_SESSION_DURATION,
        Meters.MCP_SERVER_SESSION_DURATION,
    }
    found_metrics = set()

    for rm in resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name in mcp_metric_names:
                    found_metrics.add(metric.name)
                    # Verify it's a histogram (has data_points with count and sum)
                    assert hasattr(metric.data, "data_points"), (
                        f"Metric {metric.name} should be a histogram"
                    )
                    for dp in metric.data.data_points:
                        assert hasattr(dp, "count"), f"Metric {metric.name} missing count"
                        assert hasattr(dp, "sum"), f"Metric {metric.name} missing sum"

    assert len(found_metrics) == 4, (
        f"Expected all 4 MCP metrics, found: {found_metrics}"
    )
