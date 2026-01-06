"""OpenTelemetry Dify instrumentation"""

import logging
import time
from typing import Collection

from opentelemetry.instrumentation.dify.config import Config
from opentelemetry.instrumentation.dify.utils import (
    dont_throw,
    get_llm_request_attributes,
    get_llm_response_attributes,
    is_dify_server_available,
    is_package_available,
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
    """An instrumentor for Dify SDK and Server.

    This instrumentor can work with:
    - dify-client SDK (client-side API calls)
    - Dify server v1.8.1+ (server-side operations)

    It automatically detects which components are available and instruments them accordingly.
    """

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

        # Instrument dify-client SDK if available
        if is_package_available("dify_client"):
            self._instrument_client(tracer, duration_histogram, token_histogram)

        # Instrument Dify server if available
        if is_dify_server_available():
            self._instrument_server(tracer, duration_histogram, token_histogram)

    def _instrument_client(self, tracer, duration_histogram, token_histogram):
        """Instrument dify-client SDK."""
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

    def _instrument_server(self, tracer, duration_histogram, token_histogram):
        """Instrument Dify server-side modules."""
        # Instrument LLM model runtime
        self._instrument_llm_runtime(tracer, duration_histogram, token_histogram)

        # Instrument workflow execution
        self._instrument_workflow(tracer, duration_histogram, token_histogram)

        # Instrument RAG/retrieval
        self._instrument_rag(tracer, duration_histogram, token_histogram)

        # Instrument embeddings
        self._instrument_embedding(tracer, duration_histogram, token_histogram)

        # Instrument tools
        self._instrument_tools(tracer, duration_histogram, token_histogram)

    def _instrument_llm_runtime(self, tracer, duration_histogram, token_histogram):
        """Instrument LLM model runtime."""
        try:
            wrap_function_wrapper(
                module="core.model_runtime.model_providers.__base.large_language_model",
                name="LargeLanguageModel.invoke",
                wrapper=_wrap_llm_invoke(tracer, duration_histogram, token_histogram),
            )
        except Exception as e:
            logger.debug("Failed to instrument LLM invoke: %s", e)

    def _instrument_workflow(self, tracer, duration_histogram, token_histogram):
        """Instrument workflow execution."""
        try:
            wrap_function_wrapper(
                module="core.workflow.graph_engine.graph_engine",
                name="GraphEngine.run",
                wrapper=_wrap_workflow_run(tracer),
            )
        except Exception as e:
            logger.debug("Failed to instrument workflow: %s", e)

    def _instrument_rag(self, tracer, duration_histogram, token_histogram):
        """Instrument RAG/knowledge base retrieval."""
        try:
            wrap_function_wrapper(
                module="core.rag.retrieval.retrieval_methods.retrieval_base",
                name="RetrievalBase.retrieve",
                wrapper=_wrap_rag_retrieve(tracer),
            )
        except Exception as e:
            logger.debug("Failed to instrument RAG: %s", e)

    def _instrument_embedding(self, tracer, duration_histogram, token_histogram):
        """Instrument embedding generation."""
        try:
            wrap_function_wrapper(
                module="core.model_runtime.model_providers.__base.text_embedding_model",
                name="TextEmbeddingModel.invoke",
                wrapper=_wrap_embedding_invoke(tracer, token_histogram),
            )
        except Exception as e:
            logger.debug("Failed to instrument embedding: %s", e)

    def _instrument_tools(self, tracer, duration_histogram, token_histogram):
        """Instrument tool execution."""
        try:
            wrap_function_wrapper(
                module="core.tools.tool_engine",
                name="ToolEngine.invoke",
                wrapper=_wrap_tool_invoke(tracer),
            )
        except Exception as e:
            logger.debug("Failed to instrument tools: %s", e)

    def _uninstrument(self, **kwargs):
        # Uninstrument client SDK
        if is_package_available("dify_client"):
            try:
                unwrap("dify_client.client", "CompletionClient.create_completion_message")
                unwrap("dify_client.client", "ChatClient.create_chat_message")
            except Exception as e:
                logger.debug("Failed to uninstrument client: %s", e)

        # Uninstrument server
        if is_dify_server_available():
            try:
                unwrap("core.model_runtime.model_providers.__base.large_language_model", "LargeLanguageModel.invoke")
            except Exception:
                pass

            try:
                unwrap("core.workflow.graph_engine.graph_engine", "GraphEngine.run")
            except Exception:
                pass

            try:
                unwrap("core.rag.retrieval.retrieval_methods.retrieval_base", "RetrievalBase.retrieve")
            except Exception:
                pass

            try:
                unwrap("core.model_runtime.model_providers.__base.text_embedding_model", "TextEmbeddingModel.invoke")
            except Exception:
                pass

            try:
                unwrap("core.tools.tool_engine", "ToolEngine.invoke")
            except Exception:
                pass


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


# Server-side wrappers for Dify 1.8.1+

def _wrap_llm_invoke(tracer, duration_histogram: Histogram, token_histogram: Histogram):
    """Wrapper for LLM model invocations in Dify server."""

    @dont_throw
    def wrapper(wrapped, instance, args, kwargs):
        if _SUPPRESS_INSTRUMENTATION_KEY in kwargs:
            return wrapped(*args, **kwargs)

        model_name = getattr(instance, 'model', 'unknown')
        provider_name = getattr(instance, 'provider', 'unknown')

        with tracer.start_as_current_span(
            "dify.llm.invoke",
            kind=SpanKind.CLIENT,
        ) as span:
            start_time = time.time()

            span.set_attribute(SpanAttributes.LLM_SYSTEM, "Dify")
            span.set_attribute(SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value)
            span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("gen_ai.provider", provider_name)

            try:
                result = wrapped(*args, **kwargs)

                duration = time.time() - start_time
                duration_histogram.record(
                    duration,
                    attributes={"gen_ai.operation.name": "dify.llm.invoke"},
                )

                # Extract token usage if available
                if hasattr(result, 'usage'):
                    usage = result.usage
                    if hasattr(usage, 'prompt_tokens'):
                        token_histogram.record(
                            usage.prompt_tokens,
                            attributes={
                                "gen_ai.operation.name": "dify.llm.invoke",
                                "gen_ai.token.type": "input",
                            },
                        )
                    if hasattr(usage, 'completion_tokens'):
                        token_histogram.record(
                            usage.completion_tokens,
                            attributes={
                                "gen_ai.operation.name": "dify.llm.invoke",
                                "gen_ai.token.type": "output",
                            },
                        )

                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                duration = time.time() - start_time
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                duration_histogram.record(
                    duration,
                    attributes={
                        "gen_ai.operation.name": "dify.llm.invoke",
                        "error.type": type(e).__name__,
                    },
                )
                raise

    return wrapper


def _wrap_workflow_run(tracer):
    """Wrapper for workflow execution in Dify server."""

    @dont_throw
    def wrapper(wrapped, instance, args, kwargs):
        workflow_id = kwargs.get('workflow_id') or (args[0] if args else 'unknown')

        with tracer.start_as_current_span(
            "dify.workflow",
            kind=SpanKind.SERVER,
        ) as span:
            span.set_attribute(SpanAttributes.LLM_SYSTEM, "Dify")
            span.set_attribute("gen_ai.operation.type", "workflow")
            span.set_attribute("gen_ai.workflow.id", str(workflow_id))

            try:
                result = wrapped(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def _wrap_rag_retrieve(tracer):
    """Wrapper for RAG/knowledge base retrieval in Dify server."""

    @dont_throw
    def wrapper(wrapped, instance, args, kwargs):
        query = kwargs.get('query') or (args[0] if args else '')
        top_k = kwargs.get('top_k', 5)

        with tracer.start_as_current_span(
            "dify.rag.retrieve",
            kind=SpanKind.CLIENT,
        ) as span:
            span.set_attribute(SpanAttributes.LLM_SYSTEM, "Dify")
            span.set_attribute("gen_ai.operation.type", "rag")
            span.set_attribute("gen_ai.rag.top_k", top_k)

            if query:
                set_span_attribute(span, "gen_ai.rag.query", query)

            try:
                result = wrapped(*args, **kwargs)

                # Try to get document count
                if hasattr(result, '__len__'):
                    span.set_attribute("gen_ai.rag.document_count", len(result))

                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def _wrap_embedding_invoke(tracer, token_histogram: Histogram):
    """Wrapper for embedding generation in Dify server."""

    @dont_throw
    def wrapper(wrapped, instance, args, kwargs):
        texts = kwargs.get('texts') or (args[0] if args else [])
        text_count = len(texts) if isinstance(texts, (list, tuple)) else 1

        with tracer.start_as_current_span(
            "dify.embedding.generate",
            kind=SpanKind.CLIENT,
        ) as span:
            span.set_attribute(SpanAttributes.LLM_SYSTEM, "Dify")
            span.set_attribute("gen_ai.operation.type", "embedding")
            span.set_attribute("gen_ai.embedding.text_count", text_count)

            model_name = getattr(instance, 'model', 'unknown')
            span.set_attribute("gen_ai.request.model", model_name)

            try:
                result = wrapped(*args, **kwargs)

                # Try to get dimensions
                if hasattr(result, '__len__') and len(result) > 0:
                    first_embedding = result[0] if isinstance(result, (list, tuple)) else result
                    if hasattr(first_embedding, '__len__'):
                        span.set_attribute("gen_ai.embedding.dimensions", len(first_embedding))

                # Record token usage for embeddings
                if hasattr(result, 'usage') and hasattr(result.usage, 'total_tokens'):
                    token_histogram.record(
                        result.usage.total_tokens,
                        attributes={
                            "gen_ai.operation.name": "dify.embedding.generate",
                            "gen_ai.token.type": "input",
                        },
                    )

                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper


def _wrap_tool_invoke(tracer):
    """Wrapper for tool execution in Dify server."""

    @dont_throw
    def wrapper(wrapped, instance, args, kwargs):
        tool_name = getattr(instance, 'name', 'unknown')

        with tracer.start_as_current_span(
            f"dify.tool.{tool_name}",
            kind=SpanKind.CLIENT,
        ) as span:
            span.set_attribute(SpanAttributes.LLM_SYSTEM, "Dify")
            span.set_attribute("gen_ai.operation.type", "tool")
            span.set_attribute("gen_ai.tool.name", tool_name)

            # Try to capture tool parameters
            if kwargs:
                set_span_attribute(span, "gen_ai.tool.parameters", kwargs)

            try:
                result = wrapped(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    return wrapper
