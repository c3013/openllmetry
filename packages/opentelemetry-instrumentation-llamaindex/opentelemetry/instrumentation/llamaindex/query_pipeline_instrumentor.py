from importlib.metadata import version as package_version, PackageNotFoundError
import json
import time

from wrapt import wrap_function_wrapper
from opentelemetry.context import attach, set_value

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

V10_MODULE_NAME = "llama_index.core.query_pipeline.query"
V10_LEGACY_MODULE_NAME = "llama_index.legacy.query_pipeline.query"

CLASS_NAME = "QueryPipeline"
WORKFLOW_NAME = "llama_index_query_pipeline"


class QueryPipelineInstrumentor:
    def __init__(self, tracer, workflow_duration_histogram=None):
        self._tracer = tracer
        self._workflow_duration_histogram = workflow_duration_histogram

    def instrument(self):
        try:
            package_version("llama-index-core")
            self._instrument_module(V10_MODULE_NAME)
            self._instrument_module(V10_LEGACY_MODULE_NAME)

        except PackageNotFoundError:
            pass  # not supported before v10

    def _instrument_module(self, module_name):
        wrap_function_wrapper(
            module_name, f"{CLASS_NAME}.run", run_wrapper(self._tracer, self._workflow_duration_histogram)
        )
        wrap_function_wrapper(
            module_name, f"{CLASS_NAME}.arun", arun_wrapper(self._tracer, self._workflow_duration_histogram)
        )


def set_workflow_context():
    attach(set_value("workflow_name", WORKFLOW_NAME))


@_with_tracer_wrapper
def run_wrapper(tracer, workflow_duration_histogram, wrapped, instance, args, kwargs):
    set_workflow_context()

    with tracer.start_as_current_span(f"{WORKFLOW_NAME}.workflow") as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.WORKFLOW.value,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, WORKFLOW_NAME)
        
        # Add GenAI semantic convention attributes
        span.set_attribute(SpanAttributes.GEN_AI_WORKFLOW_NAME, WORKFLOW_NAME)
        
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
            if should_send_prompts() and res:
                output_message = str(res)
                output_messages_json = json.dumps([{"role": "assistant", "content": output_message}])
                span.set_attribute(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, output_messages_json)
            
            process_response(span, res)
            span.set_status(Status(StatusCode.OK))
            
            # Record workflow duration metric
            if workflow_duration_histogram:
                workflow_duration_histogram.record(
                    duration,
                    attributes={
                        SpanAttributes.GEN_AI_WORKFLOW_NAME: WORKFLOW_NAME,
                    },
                )
            
            return res
        except Exception as e:
            duration = time.time() - start_time
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            
            # Record workflow duration metric with error
            if workflow_duration_histogram:
                workflow_duration_histogram.record(
                    duration,
                    attributes={
                        SpanAttributes.GEN_AI_WORKFLOW_NAME: WORKFLOW_NAME,
                        "error.type": type(e).__name__,
                    },
                )
            raise


@_with_tracer_wrapper
async def arun_wrapper(tracer, workflow_duration_histogram, wrapped, instance, args, kwargs):
    set_workflow_context()

    async with start_as_current_span_async(
        tracer=tracer, name=f"{WORKFLOW_NAME}.workflow"
    ) as span:
        span.set_attribute(
            SpanAttributes.TRACELOOP_SPAN_KIND,
            TraceloopSpanKindValues.WORKFLOW.value,
        )
        span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, WORKFLOW_NAME)
        
        # Add GenAI semantic convention attributes
        span.set_attribute(SpanAttributes.GEN_AI_WORKFLOW_NAME, WORKFLOW_NAME)
        
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
            if should_send_prompts() and res:
                output_message = str(res)
                output_messages_json = json.dumps([{"role": "assistant", "content": output_message}])
                span.set_attribute(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, output_messages_json)
            
            process_response(span, res)
            span.set_status(Status(StatusCode.OK))
            
            # Record workflow duration metric
            if workflow_duration_histogram:
                workflow_duration_histogram.record(
                    duration,
                    attributes={
                        SpanAttributes.GEN_AI_WORKFLOW_NAME: WORKFLOW_NAME,
                    },
                )
            
            return res
        except Exception as e:
            duration = time.time() - start_time
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            
            # Record workflow duration metric with error
            if workflow_duration_histogram:
                workflow_duration_histogram.record(
                    duration,
                    attributes={
                        SpanAttributes.GEN_AI_WORKFLOW_NAME: WORKFLOW_NAME,
                        "error.type": type(e).__name__,
                    },
                )
            raise
