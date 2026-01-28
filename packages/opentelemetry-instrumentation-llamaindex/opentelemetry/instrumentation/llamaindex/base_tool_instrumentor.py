from importlib.metadata import version as package_version, PackageNotFoundError
import json
import time

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.llamaindex.utils import (
    _with_tracer_wrapper,
    process_request,
    process_response,
    should_send_prompts,
    start_as_current_span_async,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry.trace.status import Status, StatusCode


TO_INSTRUMENT = [
    {
        "class": "FunctionTool",
        "v9_module": "llama_index.tools.function_tool",
        "v10_module": "llama_index.core.tools.function_tool",
        "v10_legacy_module": "llama_index.legacy.tools.function_tool",
    },
    {
        "class": "QueryEngineTool",
        "v9_module": "llama_index.tools.query_engine",
        "v10_module": "llama_index.core.tools.query_engine",
        "v10_legacy_module": "llama_index.legacy.tools.query_engine",
    },
]


class BaseToolInstrumentor:
    def __init__(self, tracer, tool_duration_histogram=None):
        self._tracer = tracer
        self._tool_duration_histogram = tool_duration_histogram

    def instrument(self):
        for module in TO_INSTRUMENT:
            try:
                package_version("llama-index-core")
                self._instrument_module(module["v10_module"], module["class"])
                self._instrument_module(module["v10_legacy_module"], module["class"])

            except PackageNotFoundError:
                self._instrument_module(module["v9_module"], module["class"])

    def _instrument_module(self, module_name, class_name):
        wrap_function_wrapper(
            module_name, f"{class_name}.call", query_wrapper(self._tracer, self._tool_duration_histogram)
        )
        wrap_function_wrapper(
            module_name, f"{class_name}.acall", aquery_wrapper(self._tracer, self._tool_duration_histogram)
        )


@_with_tracer_wrapper
def query_wrapper(tracer, tool_duration_histogram, wrapped, instance, args, kwargs):
    name = instance.__class__.__name__
    with tracer.start_as_current_span(f"{name}.tool") as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TOOL.value,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)
        
        # Add GenAI semantic convention attributes
        span.set_attribute(GenAIAttributes.GEN_AI_OPERATION_NAME, name)
        span.set_attribute(GenAIAttributes.GEN_AI_TOOL_NAME, name)
        span.set_attribute(GenAIAttributes.GEN_AI_TOOL_TYPE, "function")
        span.set_attribute(GenAIAttributes.GEN_AI_TOOL_CALL_ID, str(id(instance)))
        
        # Extract and set tool call arguments
        if should_send_prompts():
            arguments_json = json.dumps({"args": args, "kwargs": kwargs})
            span.set_attribute("gen_ai.tool.call.arguments", arguments_json)

        process_request(span, args, kwargs)
        
        start_time = time.time()
        try:
            res = wrapped(*args, **kwargs)
            duration = time.time() - start_time
            
            # Add GenAI tool call result
            if should_send_prompts() and res:
                result_str = str(res)
                span.set_attribute("gen_ai.tool.call.result", result_str)
            
            process_response(span, res)
            span.set_status(Status(StatusCode.OK))
            
            # Record tool duration metric
            if tool_duration_histogram:
                tool_duration_histogram.record(
                    duration,
                    attributes={
                        GenAIAttributes.GEN_AI_TOOL_NAME: name,
                    },
                )
            
            return res
        except Exception as e:
            duration = time.time() - start_time
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            
            # Record tool duration metric with error
            if tool_duration_histogram:
                tool_duration_histogram.record(
                    duration,
                    attributes={
                        GenAIAttributes.GEN_AI_TOOL_NAME: name,
                        "error.type": type(e).__name__,
                    },
                )
            raise


@_with_tracer_wrapper
async def aquery_wrapper(tracer, tool_duration_histogram, wrapped, instance, args, kwargs):
    name = instance.__class__.__name__
    async with start_as_current_span_async(tracer=tracer, name=f"{name}.tool") as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.TOOL.value,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)
        
        # Add GenAI semantic convention attributes
        span.set_attribute(GenAIAttributes.GEN_AI_OPERATION_NAME, name)
        span.set_attribute(GenAIAttributes.GEN_AI_TOOL_NAME, name)
        span.set_attribute(GenAIAttributes.GEN_AI_TOOL_TYPE, "function")
        span.set_attribute(GenAIAttributes.GEN_AI_TOOL_CALL_ID, str(id(instance)))
        
        # Extract and set tool call arguments
        if should_send_prompts():
            arguments_json = json.dumps({"args": args, "kwargs": kwargs})
            span.set_attribute("gen_ai.tool.call.arguments", arguments_json)

        process_request(span, args, kwargs)
        
        start_time = time.time()
        try:
            res = await wrapped(*args, **kwargs)
            duration = time.time() - start_time
            
            # Add GenAI tool call result
            if should_send_prompts() and res:
                result_str = str(res)
                span.set_attribute("gen_ai.tool.call.result", result_str)
            
            process_response(span, res)
            span.set_status(Status(StatusCode.OK))
            
            # Record tool duration metric
            if tool_duration_histogram:
                tool_duration_histogram.record(
                    duration,
                    attributes={
                        GenAIAttributes.GEN_AI_TOOL_NAME: name,
                    },
                )
            
            return res
        except Exception as e:
            duration = time.time() - start_time
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            
            # Record tool duration metric with error
            if tool_duration_histogram:
                tool_duration_histogram.record(
                    duration,
                    attributes={
                        GenAIAttributes.GEN_AI_TOOL_NAME: name,
                        "error.type": type(e).__name__,
                    },
                )
            raise
