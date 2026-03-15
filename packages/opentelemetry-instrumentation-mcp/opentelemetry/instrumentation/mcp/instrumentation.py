from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Collection, Tuple, Union, cast
import json
import logging
import time

from opentelemetry import context, propagate
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import get_meter
from opentelemetry.trace import get_tracer, Tracer
from wrapt import ObjectProxy, register_post_import_hook, wrap_function_wrapper
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.semconv_ai import Meters, SpanAttributes, TraceloopSpanKindValues
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE

from opentelemetry.instrumentation.mcp.version import __version__
from opentelemetry.instrumentation.mcp.utils import dont_throw, Config, is_metrics_enabled
from opentelemetry.instrumentation.mcp.fastmcp_instrumentation import (
    FastMCPInstrumentor,
)

_MCP_OPERATION_DURATION_BUCKETS = [
    0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120, 300
]

_instruments = ("mcp >= 1.6.0",)


class McpInstrumentor(BaseInstrumentor):
    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger
        self._fastmcp_instrumentor = FastMCPInstrumentor()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        if is_metrics_enabled():
            client_operation_duration_histogram = meter.create_histogram(
                name=Meters.MCP_CLIENT_OPERATION_DURATION,
                unit="s",
                description=(
                    "The duration of the MCP request or notification as observed on the sender "
                    "from the time it was sent until the response or ack is received."
                ),
                explicit_bucket_boundaries_advisory=_MCP_OPERATION_DURATION_BUCKETS,
            )
            client_session_duration_histogram = meter.create_histogram(
                name=Meters.MCP_CLIENT_SESSION_DURATION,
                unit="s",
                description="Duration of the MCP client session.",
                explicit_bucket_boundaries_advisory=_MCP_OPERATION_DURATION_BUCKETS,
            )
            server_operation_duration_histogram = meter.create_histogram(
                name=Meters.MCP_SERVER_OPERATION_DURATION,
                unit="s",
                description="The duration of the MCP request or notification as observed on the receiver/server side.",
                explicit_bucket_boundaries_advisory=_MCP_OPERATION_DURATION_BUCKETS,
            )
            server_session_duration_histogram = meter.create_histogram(
                name=Meters.MCP_SERVER_SESSION_DURATION,
                unit="s",
                description="Duration of the MCP server session.",
                explicit_bucket_boundaries_advisory=_MCP_OPERATION_DURATION_BUCKETS,
            )
        else:
            (
                client_operation_duration_histogram,
                client_session_duration_histogram,
                server_operation_duration_histogram,
                server_session_duration_histogram,
            ) = (None, None, None, None)

        # Instrument FastMCP
        self._fastmcp_instrumentor.instrument(
            tracer,
            server_operation_duration_histogram=server_operation_duration_histogram,
            server_session_duration_histogram=server_session_duration_histogram,
        )

        # Instrument FastMCP Client to create a session-level span
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.client",
                "Client.__aenter__",
                self._fastmcp_client_enter_wrapper(
                    tracer, client_session_duration_histogram
                ),
            ),
            "fastmcp.client",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "fastmcp.client",
                "Client.__aexit__",
                self._fastmcp_client_exit_wrapper(
                    tracer, client_session_duration_histogram
                ),
            ),
            "fastmcp.client",
        )

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.sse", "sse_client", self._transport_wrapper(tracer)
            ),
            "mcp.client.sse",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.sse",
                "SseServerTransport.connect_sse",
                self._transport_wrapper(tracer),
            ),
            "mcp.server.sse",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.stdio", "stdio_client", self._transport_wrapper(tracer)
            ),
            "mcp.client.stdio",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.stdio", "stdio_server", self._transport_wrapper(tracer)
            ),
            "mcp.server.stdio",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.session",
                "ServerSession.__init__",
                self._base_session_init_wrapper(tracer),
            ),
            "mcp.server.session",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.session",
                "ServerSession.__aenter__",
                self._server_session_enter_wrapper(server_session_duration_histogram),
            ),
            "mcp.server.session",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.session",
                "ServerSession.__aexit__",
                self._server_session_exit_wrapper(server_session_duration_histogram),
            ),
            "mcp.server.session",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.streamable_http",
                "streamablehttp_client",
                self._transport_wrapper(tracer),
            ),
            "mcp.client.streamable_http",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.streamable_http",
                "StreamableHTTPServerTransport.connect",
                self._transport_wrapper(tracer),
            ),
            "mcp.server.streamable_http",
        )
        wrap_function_wrapper(
            "mcp.shared.session",
            "BaseSession.send_request",
            self.patch_mcp_client(tracer, client_operation_duration_histogram),
        )

    def _uninstrument(self, **kwargs):
        unwrap("mcp.client.stdio", "stdio_client")
        unwrap("mcp.server.stdio", "stdio_server")
        self._fastmcp_instrumentor.uninstrument()

    def _transport_wrapper(self, tracer):
        @asynccontextmanager
        async def traced_method(
            wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
        ) -> AsyncGenerator[
            Union[
                Tuple[InstrumentedStreamReader, InstrumentedStreamWriter],
                Tuple[InstrumentedStreamReader, InstrumentedStreamWriter, Any],
            ],
            None,
        ]:
            async with wrapped(*args, **kwargs) as result:
                try:
                    read_stream, write_stream = result
                    yield InstrumentedStreamReader(
                        read_stream, tracer
                    ), InstrumentedStreamWriter(write_stream, tracer)
                except ValueError:
                    try:
                        read_stream, write_stream, get_session_id_callback = result
                        yield InstrumentedStreamReader(
                            read_stream, tracer
                        ), InstrumentedStreamWriter(
                            write_stream, tracer
                        ), get_session_id_callback
                    except Exception as e:
                        logging.warning(
                            f"mcp instrumentation _transport_wrapper exception: {e}"
                        )
                        yield result
                except Exception as e:
                    logging.warning(
                        f"mcp instrumentation transport_wrapper exception: {e}"
                    )
                    yield result

        return traced_method

    def _base_session_init_wrapper(self, tracer):
        def traced_method(
            wrapped: Callable[..., None], instance: Any, args: Any, kwargs: Any
        ) -> None:
            wrapped(*args, **kwargs)
            reader = getattr(instance, "_incoming_message_stream_reader", None)
            writer = getattr(instance, "_incoming_message_stream_writer", None)
            if reader and writer:
                setattr(
                    instance,
                    "_incoming_message_stream_reader",
                    ContextAttachingStreamReader(reader, tracer),
                )
                setattr(
                    instance,
                    "_incoming_message_stream_writer",
                    ContextSavingStreamWriter(writer, tracer),
                )

        return traced_method

    def _server_session_enter_wrapper(self, server_session_duration_histogram=None):
        """Wrapper for ServerSession.__aenter__ to start tracking server session duration."""

        @dont_throw
        async def traced_method(wrapped, instance, args, kwargs):
            result = await wrapped(*args, **kwargs)
            setattr(instance, "_tracing_session_start_time", time.time())
            setattr(
                instance,
                "_tracing_session_duration_histogram",
                server_session_duration_histogram,
            )
            return result

        return traced_method

    def _server_session_exit_wrapper(self, server_session_duration_histogram=None):
        """Wrapper for ServerSession.__aexit__ to record server session duration."""

        @dont_throw
        async def traced_method(wrapped, instance, args, kwargs):
            session_start = getattr(instance, "_tracing_session_start_time", None)
            histogram = getattr(
                instance,
                "_tracing_session_duration_histogram",
                server_session_duration_histogram,
            )
            if session_start is not None and histogram is not None:
                duration = time.time() - session_start
                exc_type = args[0] if args else None
                if exc_type is not None:
                    histogram.record(
                        duration,
                        attributes={"error.type": exc_type.__name__},
                    )
                else:
                    histogram.record(duration)
            return await wrapped(*args, **kwargs)

        return traced_method

    def patch_mcp_client(self, tracer: Tracer, client_operation_duration_histogram=None):
        @dont_throw
        async def traced_method(wrapped, instance, args, kwargs):
            meta = None
            method = None
            params = None
            if len(args) > 0 and hasattr(args[0].root, "method"):
                method = args[0].root.method
            if len(args) > 0 and hasattr(args[0].root, "params"):
                params = args[0].root.params
            if params:
                if hasattr(args[0].root.params, "meta"):
                    meta = args[0].root.params.meta

            # Handle trace context propagation
            if meta and len(args) > 0:
                carrier = {}
                TraceContextTextMapPropagator().inject(carrier)
                meta.traceparent = carrier["traceparent"]
                args[0].root.params.meta = meta

            # Create different span types based on method
            if method == "tools/call":
                return await self._handle_tool_call(
                    tracer,
                    method,
                    params,
                    args,
                    kwargs,
                    wrapped,
                    client_operation_duration_histogram,
                )
            else:
                return await self._handle_mcp_method(
                    tracer,
                    method,
                    args,
                    kwargs,
                    wrapped,
                    client_operation_duration_histogram,
                )

        return traced_method

    def _fastmcp_client_enter_wrapper(self, tracer, client_session_duration_histogram=None):
        """Wrapper for FastMCP Client.__aenter__ to start a session trace"""

        @dont_throw
        async def traced_method(wrapped, instance, args, kwargs):
            # Start a root span for the MCP client session and make it current
            span_context_manager = tracer.start_as_current_span("mcp.client.session")
            span = span_context_manager.__enter__()
            span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, "session")
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_NAME, "mcp.client.session"
            )

            # Store the span context manager and session start time on the instance
            setattr(instance, "_tracing_session_context_manager", span_context_manager)
            setattr(instance, "_tracing_session_start_time", time.time())
            setattr(
                instance,
                "_tracing_session_duration_histogram",
                client_session_duration_histogram,
            )

            try:
                # Call the original method
                result = await wrapped(*args, **kwargs)
                return result
            except Exception as e:
                span.set_attribute(ERROR_TYPE, type(e).__name__)
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

        return traced_method

    def _fastmcp_client_exit_wrapper(self, tracer, client_session_duration_histogram=None):
        """Wrapper for FastMCP Client.__aexit__ to end the session trace"""

        @dont_throw
        async def traced_method(wrapped, instance, args, kwargs):
            exc_type = args[0] if args else None
            session_start = getattr(instance, "_tracing_session_start_time", None)
            histogram = getattr(
                instance,
                "_tracing_session_duration_histogram",
                client_session_duration_histogram,
            )
            if session_start is not None and histogram is not None:
                duration = time.time() - session_start
                metric_attrs = (
                    {"error.type": exc_type.__name__} if exc_type is not None else {}
                )
                histogram.record(duration, attributes=metric_attrs)

            context_manager = getattr(
                instance, "_tracing_session_context_manager", None
            )
            try:
                result = await wrapped(*args, **kwargs)
                if context_manager:
                    context_manager.__exit__(None, None, None)
                return result
            except Exception as e:
                if context_manager:
                    context_manager.__exit__(type(e), e, e.__traceback__)
                raise

        return traced_method

    async def _handle_tool_call(
        self,
        tracer,
        method,
        params,
        args,
        kwargs,
        wrapped,
        client_operation_duration_histogram=None,
    ):
        """Handle tools/call with tool semantics"""
        # Extract the actual tool name
        entity_name = method
        span_name = f"{method}.tool"
        if params:
            try:
                if hasattr(params, "name"):
                    entity_name = params.name
                    span_name = f"{params.name}.tool"
                elif hasattr(params, "__dict__") and "name" in params.__dict__:
                    entity_name = params.__dict__["name"]
                    span_name = f"{params.__dict__['name']}.tool"
            except Exception:
                pass

        start_time = time.time()
        with tracer.start_as_current_span(span_name) as span:
            # Set tool-specific attributes
            span.set_attribute(
                SpanAttributes.TRACELOOP_SPAN_KIND, TraceloopSpanKindValues.TOOL.value
            )
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)

            # Add input
            clean_input = self._extract_clean_input(method, params)
            if clean_input:
                try:
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_INPUT, json.dumps(clean_input)
                    )
                except (TypeError, ValueError):
                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_INPUT, str(clean_input)
                    )

            try:
                result = await self._execute_and_handle_result(
                    span, method, args, kwargs, wrapped, clean_output=True
                )
            except Exception as exc:
                if client_operation_duration_histogram is not None:
                    duration = time.time() - start_time
                    metric_attrs = {
                        SpanAttributes.MCP_METHOD_NAME: method,
                        "error.type": type(exc).__name__,
                    }
                    if entity_name != method:
                        metric_attrs["gen_ai.tool.name"] = entity_name
                    client_operation_duration_histogram.record(
                        duration, attributes=metric_attrs
                    )
                raise
            if client_operation_duration_histogram is not None:
                duration = time.time() - start_time
                metric_attrs = {SpanAttributes.MCP_METHOD_NAME: method}
                if entity_name != method:
                    metric_attrs["gen_ai.tool.name"] = entity_name
                client_operation_duration_histogram.record(duration, attributes=metric_attrs)
            return result

    async def _handle_mcp_method(
        self,
        tracer,
        method,
        args,
        kwargs,
        wrapped,
        client_operation_duration_histogram=None,
    ):
        """Handle non-tool MCP methods with simple serialization"""
        start_time = time.time()
        with tracer.start_as_current_span(f"{method}.mcp") as span:
            span.set_attribute(
                SpanAttributes.TRACELOOP_ENTITY_INPUT, f"{serialize(args[0])}"
            )
            try:
                result = await self._execute_and_handle_result(
                    span, method, args, kwargs, wrapped, clean_output=False
                )
            except Exception as exc:
                if client_operation_duration_histogram is not None:
                    duration = time.time() - start_time
                    metric_attrs = {
                        SpanAttributes.MCP_METHOD_NAME: method,
                        "error.type": type(exc).__name__,
                    }
                    client_operation_duration_histogram.record(
                        duration, attributes=metric_attrs
                    )
                raise
            if client_operation_duration_histogram is not None:
                duration = time.time() - start_time
                metric_attrs = {SpanAttributes.MCP_METHOD_NAME: method}
                client_operation_duration_histogram.record(duration, attributes=metric_attrs)
            return result

    async def _execute_and_handle_result(
        self, span, method, args, kwargs, wrapped, clean_output=False
    ):
        """Execute the wrapped function and handle the result"""
        try:
            result = await wrapped(*args, **kwargs)
            # Add output
            if clean_output:
                clean_output_data = self._extract_clean_output(method, result)
                if clean_output_data:
                    try:
                        span.set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                            json.dumps(clean_output_data),
                        )
                    except (TypeError, ValueError):
                        span.set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                            str(clean_output_data),
                        )
            else:
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_OUTPUT, serialize(result)
                )
            # Handle errors
            if hasattr(result, "isError") and result.isError:
                if len(result.content) > 0:
                    span.set_status(
                        Status(StatusCode.ERROR, f"{result.content[0].text}")
                    )
            else:
                span.set_status(Status(StatusCode.OK))
            return result
        except Exception as e:
            span.set_attribute(ERROR_TYPE, type(e).__name__)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

    def _extract_clean_input(self, method: str, params: Any) -> dict:
        """Extract clean input parameters for different MCP method types"""
        if not params:
            return {}

        try:
            if method == "tools/call":
                # For tool calls, extract name and arguments
                result = {}
                if hasattr(params, "name"):
                    result["tool_name"] = params.name
                if hasattr(params, "arguments"):
                    if hasattr(params.arguments, "__dict__"):
                        result["arguments"] = params.arguments.__dict__
                    else:
                        result["arguments"] = params.arguments
                elif hasattr(params, "__dict__") and "arguments" in params.__dict__:
                    result["arguments"] = params.__dict__["arguments"]
                return result
            elif method == "tools/list":
                # For list_tools, there are usually no parameters
                return {}
            else:
                # For other methods, try to serialize params cleanly
                if hasattr(params, "__dict__"):
                    # Remove internal fields starting with _ and non-serializable objects
                    clean_params = {}
                    for k, v in params.__dict__.items():
                        if not k.startswith("_"):
                            try:
                                # Test if the value is JSON serializable
                                json.dumps(v)
                                clean_params[k] = v
                            except (TypeError, ValueError):
                                # If not serializable, store a string representation
                                clean_params[k] = str(type(v).__name__)
                    return clean_params
                else:
                    return {"params": str(params)}
        except Exception:
            return {}

    def _extract_clean_output(self, method: str, result: Any) -> dict:
        """Extract clean output for different MCP method types"""
        if not result:
            return {}

        try:
            if method == "tools/call":
                # For tool calls, extract the actual result content
                output = {}
                if hasattr(result, "content") and result.content:
                    if len(result.content) > 0:
                        content_item = result.content[0]
                        if hasattr(content_item, "text"):
                            output["result"] = content_item.text
                        elif hasattr(content_item, "__dict__"):
                            output["result"] = content_item.__dict__
                        else:
                            output["result"] = str(content_item)

                # Check if this is an error response
                if hasattr(result, "isError") and result.isError:
                    output["is_error"] = True

                return output
            elif method == "tools/list":
                # For list_tools, extract tool names and descriptions
                output = {"tools": []}
                if hasattr(result, "tools") and result.tools:
                    for tool in result.tools:
                        tool_info = {}
                        if hasattr(tool, "name"):
                            tool_info["name"] = tool.name
                        if hasattr(tool, "description"):
                            tool_info["description"] = tool.description
                        output["tools"].append(tool_info)
                return output
            else:
                # For other methods, try to serialize result cleanly
                if hasattr(result, "__dict__"):
                    clean_result = {
                        k: v
                        for k, v in result.__dict__.items()
                        if not k.startswith("_")
                    }
                    return clean_result
                else:
                    return {"result": str(result)}
        except Exception:
            return {}


def serialize(request, depth=0, max_depth=4):
    """Serialize input args to MCP server into JSON.
    The function accepts input object and converts into JSON
    keeping depth in mind to prevent creating large nested JSON"""
    if depth > max_depth:
        return {}
    depth += 1

    def is_serializable(request):
        try:
            json.dumps(request)
            return True
        except Exception:
            return False

    if is_serializable(request):
        return json.dumps(request)
    else:
        result = {}
        try:
            if hasattr(request, "model_dump_json"):
                return request.model_dump_json()
            if hasattr(request, "__dict__"):
                for attrib in request.__dict__:
                    if not attrib.startswith("_"):
                        if type(request.__dict__[attrib]) in [
                            bool,
                            str,
                            int,
                            float,
                            type(None),
                        ]:
                            result[str(attrib)] = request.__dict__[attrib]
                        else:
                            result[str(attrib)] = serialize(
                                request.__dict__[attrib], depth
                            )
        except Exception:
            pass
        return json.dumps(result)


class InstrumentedStreamReader(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    def __init__(self, wrapped, tracer):
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    @dont_throw
    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        from mcp.types import JSONRPCMessage, JSONRPCRequest

        async for item in self.__wrapped__:
            # Handle different item types based on what's available
            request = None
            if hasattr(item, "message") and hasattr(item.message, "root"):
                request = item.message.root
            elif type(item) is JSONRPCMessage:
                request = cast(JSONRPCMessage, item).root
            elif hasattr(item, "root"):
                request = item.root
            else:
                yield item
                continue

            if not isinstance(request, JSONRPCRequest):
                yield item
                continue

            if request.params:
                meta = request.params.get("_meta")
                if meta:
                    ctx = propagate.extract(meta)
                    restore = context.attach(ctx)
                    try:
                        yield item
                        continue
                    finally:
                        context.detach(restore)
            yield item


class InstrumentedStreamWriter(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    def __init__(self, wrapped, tracer):
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    @dont_throw
    async def send(self, item: Any) -> Any:
        from mcp.types import JSONRPCMessage, JSONRPCRequest

        # Handle different item types based on what's available
        request = None
        if hasattr(item, "message") and hasattr(item.message, "root"):
            request = item.message.root
        elif type(item) is JSONRPCMessage:
            request = cast(JSONRPCMessage, item).root
        elif hasattr(item, "root"):
            request = item.root
        else:
            return await self.__wrapped__.send(item)

        with self._tracer.start_as_current_span("ResponseStreamWriter") as span:
            if hasattr(request, "result"):
                span.set_attribute(
                    SpanAttributes.MCP_RESPONSE_VALUE, f"{serialize(request.result)}"
                )
                if "isError" in request.result:
                    if request.result["isError"] is True:
                        span.set_status(
                            Status(
                                StatusCode.ERROR,
                                f"{request.result['content'][0]['text']}",
                            )
                        )
            if hasattr(request, "id"):
                span.set_attribute(SpanAttributes.MCP_REQUEST_ID, f"{request.id}")

            if not isinstance(request, JSONRPCRequest):
                return await self.__wrapped__.send(item)
            meta = None
            if not request.params:
                request.params = {}
            meta = request.params.setdefault("_meta", {})

            propagate.get_global_textmap().inject(meta)
            return await self.__wrapped__.send(item)


@dataclass(slots=True, frozen=True)
class ItemWithContext:
    item: Any
    ctx: context.Context


class ContextSavingStreamWriter(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    def __init__(self, wrapped, tracer):
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    @dont_throw
    async def send(self, item: Any) -> Any:
        # Removed RequestStreamWriter span creation - we don't need low-level protocol spans
        ctx = context.get_current()
        return await self.__wrapped__.send(ItemWithContext(item, ctx))


class ContextAttachingStreamReader(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    def __init__(self, wrapped, tracer):
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        async for item in self.__wrapped__:
            item_with_context = cast(ItemWithContext, item)
            restore = context.attach(item_with_context.ctx)
            try:
                yield item_with_context.item
            finally:
                context.detach(restore)
