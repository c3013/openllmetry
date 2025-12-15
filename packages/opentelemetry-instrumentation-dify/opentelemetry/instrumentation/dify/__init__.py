"""OpenTelemetry Dify instrumentation"""

import logging
import time
from typing import Collection

from opentelemetry.instrumentation.dify.config import Config
from opentelemetry.instrumentation.dify.utils import (
    dont_throw,
    get_llm_request_attributes,
    get_llm_response_attributes,
    parse_streaming_response,
    set_span_attribute,
)
from opentelemetry.instrumentation.dify.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.metrics import Histogram, get_meter
from opentelemetry.semconv_ai import LLMRequestTypeValues, Meters, SpanAttributes
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("dify-client >= 0.1.0",)


class DifyInstrumentor(BaseInstrumentor):
    """An instrumentor for Dify SDK."""

    def __init__(
        self,
        exception_logger=None,
        use_legacy_attributes: bool = True,
    ):
        """Create a Dify instrumentor instance.

        Args:
            exception_logger: A callable that takes an Exception as input. This will be
                used to log exceptions that occur during instrumentation. If None, exceptions will not be logged.
            use_legacy_attributes: If True, uses span attributes for Inputs/Outputs instead of events.
        """
        super().__init__()
        Config.exception_logger = exception_logger
        Config.use_legacy_attributes = use_legacy_attributes

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        meter_provider = kwargs.get("meter_provider")
        meter = get_meter(__name__, __version__, meter_provider)

        # Create duration histogram
        duration_histogram = meter.create_histogram(
            name=Meters.LLM_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration",
        )

        # Create token histogram
        token_histogram = meter.create_histogram(
            name=Meters.LLM_TOKEN_USAGE,
            unit="token",
            description="Measures number of input and output tokens used",
        )

        # Wrap CompletionClient.create_completion_message
        wrap_function_wrapper(
            module="dify_client.client",
            name="CompletionClient.create_completion_message",
            wrapper=_wrap_completion_message(
                tracer, duration_histogram, token_histogram
            ),
        )

        # Wrap ChatClient.create_chat_message
        wrap_function_wrapper(
            module="dify_client.client",
            name="ChatClient.create_chat_message",
            wrapper=_wrap_chat_message(tracer, duration_histogram, token_histogram),
        )

    def _uninstrument(self, **kwargs):
        unwrap("dify_client.client", "CompletionClient.create_completion_message")
        unwrap("dify_client.client", "ChatClient.create_chat_message")


def _wrap_completion_message(tracer, duration_histogram: Histogram, token_histogram: Histogram):
    """Wrapper for CompletionClient.create_completion_message"""

    @dont_throw
    def wrapper(wrapped, instance, args, kwargs):
        # Check if instrumentation is suppressed
        if _SUPPRESS_INSTRUMENTATION_KEY in kwargs:
            return wrapped(*args, **kwargs)

        inputs = kwargs.get("inputs", {})
        response_mode = kwargs.get("response_mode", "blocking")
        user = kwargs.get("user")
        files = kwargs.get("files")

        span_name = "dify.completion"
        
        with tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
        ) as span:
            start_time = time.time()
            
            # Set span attributes
            span.set_attribute(SpanAttributes.LLM_SYSTEM, "Dify")
            span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value)
            
            # Set request attributes
            request_attributes = get_llm_request_attributes(
                inputs=inputs,
                user=user,
                response_mode=response_mode,
                files=files,
            )
            for key, value in request_attributes.items():
                set_span_attribute(span, key, value)

            try:
                response = wrapped(*args, **kwargs)
                
                # Handle streaming vs blocking
                if response_mode == "streaming":
                    # For streaming, we wrap the response
                    return _StreamingResponseWrapper(
                        response,
                        span,
                        start_time,
                        duration_histogram,
                        token_histogram,
                    )
                else:
                    # For blocking, process the response
                    duration = time.time() - start_time
                    
                    try:
                        response_json = response.json()
                        
                        # Set response attributes
                        response_attributes = get_llm_response_attributes(response_json)
                        for key, value in response_attributes.items():
                            set_span_attribute(span, key, value)
                        
                        # Record metrics
                        _record_metrics(
                            duration_histogram,
                            token_histogram,
                            duration,
                            response_json,
                            span_name,
                        )
                        
                        span.set_status(Status(StatusCode.OK))
                    except Exception as e:
                        logger.debug("Error processing Dify response: %s", e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    
                    return response
                    
            except Exception as e:
                duration = time.time() - start_time
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                
                # Record error metrics
                duration_histogram.record(
                    duration,
                    attributes={
                        "gen_ai.operation.name": span_name,
                        "error.type": type(e).__name__,
                    },
                )
                raise

    return wrapper


def _wrap_chat_message(tracer, duration_histogram: Histogram, token_histogram: Histogram):
    """Wrapper for ChatClient.create_chat_message"""

    @dont_throw
    def wrapper(wrapped, instance, args, kwargs):
        # Check if instrumentation is suppressed
        if _SUPPRESS_INSTRUMENTATION_KEY in kwargs:
            return wrapped(*args, **kwargs)

        inputs = kwargs.get("inputs", {})
        query = kwargs.get("query", "")
        response_mode = kwargs.get("response_mode", "blocking")
        user = kwargs.get("user")
        conversation_id = kwargs.get("conversation_id")
        files = kwargs.get("files")

        span_name = "dify.chat"
        
        with tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
        ) as span:
            start_time = time.time()
            
            # Set span attributes
            span.set_attribute(SpanAttributes.LLM_SYSTEM, "Dify")
            span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.CHAT.value)
            
            if conversation_id:
                span.set_attribute("gen_ai.conversation_id", conversation_id)
            
            # Set request attributes
            request_attributes = get_llm_request_attributes(
                inputs=inputs,
                query=query,
                user=user,
                response_mode=response_mode,
                files=files,
            )
            for key, value in request_attributes.items():
                set_span_attribute(span, key, value)

            try:
                response = wrapped(*args, **kwargs)
                
                # Handle streaming vs blocking
                if response_mode == "streaming":
                    # For streaming, we wrap the response
                    return _StreamingResponseWrapper(
                        response,
                        span,
                        start_time,
                        duration_histogram,
                        token_histogram,
                    )
                else:
                    # For blocking, process the response
                    duration = time.time() - start_time
                    
                    try:
                        response_json = response.json()
                        
                        # Set response attributes
                        response_attributes = get_llm_response_attributes(response_json)
                        for key, value in response_attributes.items():
                            set_span_attribute(span, key, value)
                        
                        # Record metrics
                        _record_metrics(
                            duration_histogram,
                            token_histogram,
                            duration,
                            response_json,
                            span_name,
                        )
                        
                        span.set_status(Status(StatusCode.OK))
                    except Exception as e:
                        logger.debug("Error processing Dify response: %s", e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    
                    return response
                    
            except Exception as e:
                duration = time.time() - start_time
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                
                # Record error metrics
                duration_histogram.record(
                    duration,
                    attributes={
                        "gen_ai.operation.name": span_name,
                        "error.type": type(e).__name__,
                    },
                )
                raise

    return wrapper


class _StreamingResponseWrapper:
    """Wrapper for streaming responses to collect metrics and attributes."""

    def __init__(
        self,
        response,
        span,
        start_time,
        duration_histogram: Histogram,
        token_histogram: Histogram,
    ):
        self._response = response
        self._span = span
        self._start_time = start_time
        self._duration_histogram = duration_histogram
        self._token_histogram = token_histogram
        self._first_chunk = True
        self._response_text = []
        self._response_json = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            line = next(self._response.iter_lines(decode_unicode=True))
            
            if line:
                parsed = parse_streaming_response(line)
                if parsed:
                    # Store for final processing
                    self._response_json = parsed
                    
                    # Collect answer chunks
                    if "answer" in parsed:
                        self._response_text.append(parsed["answer"])
            
            return line
        except StopIteration:
            # Stream ended, finalize span
            self._finalize()
            raise

    def iter_lines(self, decode_unicode=True):
        """Mimic requests response iter_lines method."""
        for line in self._response.iter_lines(decode_unicode=decode_unicode):
            if line:
                parsed = parse_streaming_response(line)
                if parsed:
                    self._response_json = parsed
                    if "answer" in parsed:
                        self._response_text.append(parsed["answer"])
            
            yield line
        
        # Finalize when iteration is complete
        self._finalize()

    def _finalize(self):
        """Finalize the span with collected metrics and attributes."""
        duration = time.time() - self._start_time
        
        if self._response_json:
            try:
                # Set response attributes
                response_attributes = get_llm_response_attributes(self._response_json)
                for key, value in response_attributes.items():
                    set_span_attribute(self._span, key, value)
                
                # Record metrics
                span_name = self._span.name
                _record_metrics(
                    self._duration_histogram,
                    self._token_histogram,
                    duration,
                    self._response_json,
                    span_name,
                )
                
                self._span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.debug("Error finalizing streaming response: %s", e)
                self._span.set_status(Status(StatusCode.ERROR, str(e)))
        else:
            self._span.set_status(Status(StatusCode.OK))
        
        self._span.end()

    def raise_for_status(self):
        """Proxy to the underlying response."""
        return self._response.raise_for_status()

    def json(self):
        """Proxy to the underlying response."""
        return self._response.json()


def _record_metrics(
    duration_histogram: Histogram,
    token_histogram: Histogram,
    duration: float,
    response_json: dict,
    operation_name: str,
):
    """Record metrics for a completed LLM operation."""
    attributes = {"gen_ai.operation.name": operation_name}
    
    # Record duration
    duration_histogram.record(duration, attributes=attributes)
    
    # Record token usage
    metadata = response_json.get("metadata", {})
    usage = metadata.get("usage", {})
    
    if "prompt_tokens" in usage:
        token_histogram.record(
            usage["prompt_tokens"],
            attributes={
                **attributes,
                "gen_ai.token.type": "input",
            },
        )
    
    if "completion_tokens" in usage:
        token_histogram.record(
            usage["completion_tokens"],
            attributes={
                **attributes,
                "gen_ai.token.type": "output",
            },
        )
