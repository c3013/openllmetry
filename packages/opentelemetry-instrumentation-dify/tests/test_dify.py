"""Tests for Dify instrumentation"""

import pytest
import requests_mock
from dify_client import ChatClient, CompletionClient


@pytest.mark.usefixtures("instrument_legacy")
def test_completion_blocking(span_exporter):
    """Test completion with blocking response mode."""
    api_key = "test-api-key"
    client = CompletionClient(api_key)

    # Mock the response
    with requests_mock.Mocker() as m:
        mock_response = {
            "message_id": "msg-123",
            "mode": "completion",
            "answer": "This is a test completion response.",
            "metadata": {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
                "model": "gpt-3.5-turbo",
            },
        }
        m.post("https://api.dify.ai/v1/completion-messages", json=mock_response)

        response = client.create_completion_message(
            inputs={"query": "Test query"},
            response_mode="blocking",
            user="test-user",
        )

    # Check response
    assert response.json()["answer"] == "This is a test completion response."

    # Check spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "dify.completion"
    assert span.attributes["gen_ai.system"] == "Dify"
    assert span.attributes["llm.request.type"] == "completion"
    assert span.attributes["gen_ai.response.id"] == "msg-123"
    assert span.attributes["gen_ai.usage.total_tokens"] == 30


@pytest.mark.usefixtures("instrument_legacy")
def test_chat_blocking(span_exporter):
    """Test chat with blocking response mode."""
    api_key = "test-api-key"
    client = ChatClient(api_key)

    # Mock the response
    with requests_mock.Mocker() as m:
        mock_response = {
            "message_id": "msg-456",
            "conversation_id": "conv-789",
            "mode": "chat",
            "answer": "Hello! How can I help you?",
            "metadata": {
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 25,
                    "total_tokens": 40,
                },
                "model": "gpt-4",
            },
        }
        m.post("https://api.dify.ai/v1/chat-messages", json=mock_response)

        response = client.create_chat_message(
            inputs={},
            query="Hello",
            user="test-user",
            response_mode="blocking",
        )

    # Check response
    assert response.json()["answer"] == "Hello! How can I help you?"

    # Check spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "dify.chat"
    assert span.attributes["gen_ai.system"] == "Dify"
    assert span.attributes["llm.request.type"] == "chat"
    assert span.attributes["gen_ai.response.id"] == "msg-456"
    assert span.attributes["gen_ai.response.conversation_id"] == "conv-789"
    assert span.attributes["gen_ai.usage.total_tokens"] == 40


@pytest.mark.usefixtures("instrument_legacy")
def test_chat_with_conversation_id(span_exporter):
    """Test chat with existing conversation ID."""
    api_key = "test-api-key"
    client = ChatClient(api_key)

    # Mock the response
    with requests_mock.Mocker() as m:
        mock_response = {
            "message_id": "msg-999",
            "conversation_id": "conv-existing",
            "mode": "chat",
            "answer": "Continuing our conversation.",
            "metadata": {
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 30,
                    "total_tokens": 50,
                },
                "model": "gpt-4",
            },
        }
        m.post("https://api.dify.ai/v1/chat-messages", json=mock_response)

        response = client.create_chat_message(
            inputs={},
            query="Continue",
            user="test-user",
            response_mode="blocking",
            conversation_id="conv-existing",
        )

    # Check response
    assert response.json()["conversation_id"] == "conv-existing"

    # Check spans
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes["gen_ai.conversation_id"] == "conv-existing"


@pytest.mark.usefixtures("instrument_legacy")
def test_trace_content_disabled(span_exporter, monkeypatch):
    """Test that prompts and completions are not logged when TRACELOOP_TRACE_CONTENT is false."""
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")

    api_key = "test-api-key"
    client = ChatClient(api_key)

    # Mock the response
    with requests_mock.Mocker() as m:
        mock_response = {
            "message_id": "msg-secret",
            "conversation_id": "conv-secret",
            "mode": "chat",
            "answer": "Secret answer",
            "metadata": {
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "total_tokens": 15,
                },
            },
        }
        m.post("https://api.dify.ai/v1/chat-messages", json=mock_response)

        _ = client.create_chat_message(
            inputs={},
            query="Secret query",
            user="test-user",
            response_mode="blocking",
        )

    # Check that content is not in attributes
    spans = span_exporter.get_finished_spans()
    span = spans[0]

    # These should not be present when content tracing is disabled
    assert "gen_ai.prompt" not in span.attributes
    assert "gen_ai.completion" not in span.attributes

    # But metadata should still be present
    assert span.attributes["gen_ai.response.id"] == "msg-secret"
