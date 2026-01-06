"""Utility functions for Dify instrumentation"""

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"


def is_dify_server_available():
    """Check if Dify server modules are available."""
    try:
        import core  # noqa: F401
        return True
    except ImportError:
        return False


def is_package_available(package_name):
    """Check if a package is available for import."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def should_send_prompts():
    """Check if prompts should be sent to the tracer."""
    return os.getenv(TRACELOOP_TRACE_CONTENT, "true").lower() == "true"


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug("Error in Dify instrumentation: %s", e, exc_info=True)
            return None

    return wrapper


def set_span_attribute(span, name, value):
    """Set span attribute if value is not None."""
    if value is not None:
        if isinstance(value, (dict, list)):
            span.set_attribute(name, json.dumps(value))
        else:
            span.set_attribute(name, value)


def get_llm_request_attributes(
    inputs: Dict[str, Any],
    query: Optional[str] = None,
    user: Optional[str] = None,
    response_mode: str = "blocking",
    files: Optional[list] = None,
) -> Dict[str, Any]:
    """Extract request attributes for LLM call."""
    attributes = {}

    if should_send_prompts():
        if query:
            attributes["gen_ai.prompt"] = query
        if inputs:
            attributes["gen_ai.inputs"] = json.dumps(inputs)

    attributes["gen_ai.request.response_mode"] = response_mode

    if user:
        attributes["gen_ai.user.id"] = user

    if files:
        attributes["gen_ai.request.has_files"] = True
        attributes["gen_ai.request.file_count"] = len(files)

    return attributes


def get_llm_response_attributes(response_json: Dict[str, Any]) -> Dict[str, Any]:
    """Extract response attributes from Dify API response."""
    attributes = {}

    # Common attributes
    if "message_id" in response_json:
        attributes["gen_ai.response.id"] = response_json["message_id"]

    if "conversation_id" in response_json:
        attributes["gen_ai.response.conversation_id"] = response_json["conversation_id"]

    if "mode" in response_json:
        attributes["gen_ai.response.mode"] = response_json["mode"]

    # Response content
    if should_send_prompts():
        if "answer" in response_json:
            attributes["gen_ai.completion"] = response_json["answer"]

    # Usage/token information
    metadata = response_json.get("metadata", {})
    usage = metadata.get("usage", {})

    if "total_tokens" in usage:
        attributes["gen_ai.usage.completion_tokens"] = usage.get("completion_tokens", 0)
        attributes["gen_ai.usage.prompt_tokens"] = usage.get("prompt_tokens", 0)
        attributes["gen_ai.usage.total_tokens"] = usage["total_tokens"]

    # Model information
    if "model" in metadata:
        attributes["gen_ai.response.model"] = metadata["model"]

    return attributes


def parse_streaming_response(line: str) -> Optional[Dict[str, Any]]:
    """Parse a streaming response line."""
    try:
        if line.startswith("data:"):
            line = line[5:].strip()
        if line:
            return json.loads(line)
    except json.JSONDecodeError:
        logger.debug("Failed to parse streaming response: %s", line)
    return None
