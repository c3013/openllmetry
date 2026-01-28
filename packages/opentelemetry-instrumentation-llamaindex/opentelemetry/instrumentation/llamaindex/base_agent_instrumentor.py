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
        "class": "AgentRunner",
        "v10_module": "llama_index.core.agent.runner.base",
        "v10_legacy_module": "llama_index.legacy.agent.runner.base",
    },
    {
        "class": "OpenAIAssistantAgent",
        "v10_module": "llama_index.agent.openai.openai_assistant_agent",
        "v10_legacy_module": "llama_index.legacy.agent.openai_assistant_agent",
    },
]


class BaseAgentInstrumentor:
    def __init__(self, tracer, agent_duration_histogram=None):
        self._tracer = tracer
        self._agent_duration_histogram = agent_duration_histogram

    def instrument(self):
        for module in TO_INSTRUMENT:
            try:
                package_version("llama-index-core")
                self._instrument_module(module["v10_module"], module["class"])
                self._instrument_module(module["v10_legacy_module"], module["class"])

            except PackageNotFoundError:
                pass  # not supported before v10

    def _instrument_module(self, module_name, class_name):
        wrap_function_wrapper(
            module_name, f"{class_name}.chat", query_wrapper(self._tracer, self._agent_duration_histogram)
        )
        wrap_function_wrapper(
            module_name, f"{class_name}.achat", aquery_wrapper(self._tracer, self._agent_duration_histogram)
        )


@_with_tracer_wrapper
def query_wrapper(tracer, agent_duration_histogram, wrapped, instance, args, kwargs):
    name = instance.__class__.__name__
    with tracer.start_as_current_span(f"{name}.agent") as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.AGENT.value,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)
        
        # Add GenAI semantic convention attributes
        span.set_attribute(GenAIAttributes.GEN_AI_OPERATION_NAME, name)
        span.set_attribute(GenAIAttributes.GEN_AI_AGENT_ID, str(id(instance)))
        
        # Extract and set input messages
        if args and should_send_prompts():
            input_message = str(args[0]) if args else ""
            input_messages_json = json.dumps([{"role": "user", "content": input_message}])
            span.set_attribute(GenAIAttributes.GEN_AI_INPUT_MESSAGES, input_messages_json)

        process_request(span, args, kwargs)
        
        start_time = time.time()
        try:
            res = wrapped(*args, **kwargs)
            duration = time.time() - start_time
            
            # Add GenAI output messages
            if should_send_prompts() and hasattr(res, 'response'):
                output_message = str(res.response)
                output_messages_json = json.dumps([{"role": "assistant", "content": output_message}])
                span.set_attribute(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, output_messages_json)
            
            process_response(span, res)
            span.set_status(Status(StatusCode.OK))
            
            # Record agent duration metric
            if agent_duration_histogram:
                agent_duration_histogram.record(
                    duration,
                    attributes={
                        GenAIAttributes.GEN_AI_OPERATION_NAME: name,
                    },
                )
            
            return res
        except Exception as e:
            duration = time.time() - start_time
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            
            # Record agent duration metric with error
            if agent_duration_histogram:
                agent_duration_histogram.record(
                    duration,
                    attributes={
                        GenAIAttributes.GEN_AI_OPERATION_NAME: name,
                        "error.type": type(e).__name__,
                    },
                )
            raise


@_with_tracer_wrapper
async def aquery_wrapper(tracer, agent_duration_histogram, wrapped, instance, args, kwargs):
    name = instance.__class__.__name__
    async with start_as_current_span_async(tracer=tracer, name=f"{name}.agent") as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.AGENT.value,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)
        
        # Add GenAI semantic convention attributes
        span.set_attribute(GenAIAttributes.GEN_AI_OPERATION_NAME, name)
        span.set_attribute(GenAIAttributes.GEN_AI_AGENT_ID, str(id(instance)))
        
        # Extract and set input messages
        if args and should_send_prompts():
            input_message = str(args[0]) if args else ""
            input_messages_json = json.dumps([{"role": "user", "content": input_message}])
            span.set_attribute(GenAIAttributes.GEN_AI_INPUT_MESSAGES, input_messages_json)

        process_request(span, args, kwargs)
        
        start_time = time.time()
        try:
            res = await wrapped(*args, **kwargs)
            duration = time.time() - start_time
            
            # Add GenAI output messages
            if should_send_prompts() and hasattr(res, 'response'):
                output_message = str(res.response)
                output_messages_json = json.dumps([{"role": "assistant", "content": output_message}])
                span.set_attribute(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, output_messages_json)
            
            process_response(span, res)
            span.set_status(Status(StatusCode.OK))
            
            # Record agent duration metric
            if agent_duration_histogram:
                agent_duration_histogram.record(
                    duration,
                    attributes={
                        GenAIAttributes.GEN_AI_OPERATION_NAME: name,
                    },
                )
            
            return res
        except Exception as e:
            duration = time.time() - start_time
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            
            # Record agent duration metric with error
            if agent_duration_histogram:
                agent_duration_histogram.record(
                    duration,
                    attributes={
                        GenAIAttributes.GEN_AI_OPERATION_NAME: name,
                        "error.type": type(e).__name__,
                    },
                )
            raise
