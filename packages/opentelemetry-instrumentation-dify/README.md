# OpenTelemetry Dify Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-dify/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-dify.svg">
</a>

This library allows tracing LLM applications built with [Dify](https://github.com/langgenius/dify) using the Dify Python client SDK.

## Installation

```bash
pip install opentelemetry-instrumentation-dify
```

## Example usage

```python
from opentelemetry.instrumentation.dify import DifyInstrumentor

DifyInstrumentor().instrument()
```

## Usage with Dify Client

After instrumentation, all Dify client calls will be automatically traced:

```python
from dify_client import ChatClient, CompletionClient
from opentelemetry.instrumentation.dify import DifyInstrumentor

# Initialize instrumentation
DifyInstrumentor().instrument()

# Use Dify Chat Client
api_key = "your_api_key"
chat_client = ChatClient(api_key)

# Create a chat message (automatically traced)
response = chat_client.create_chat_message(
    inputs={},
    query="Hello, how are you?",
    user="user_id",
    response_mode="blocking"
)

print(response.json().get('answer'))

# Use Dify Completion Client
completion_client = CompletionClient(api_key)

# Create a completion message (automatically traced)
response = completion_client.create_completion_message(
    inputs={"query": "What's the weather like today?"},
    response_mode="blocking",
    user="user_id"
)

print(response.json().get('answer'))
```

## Streaming Support

The instrumentation also supports streaming responses:

```python
import json
from dify_client import ChatClient

chat_client = ChatClient(api_key)

# Streaming response (automatically traced)
response = chat_client.create_chat_message(
    inputs={},
    query="Tell me a story",
    user="user_id",
    response_mode="streaming"
)

for line in response.iter_lines(decode_unicode=True):
    line = line.split('data:', 1)[-1]
    if line.strip():
        data = json.loads(line.strip())
        print(data.get('answer', ''), end='', flush=True)
```

## Privacy

**By default, this instrumentation logs prompts, completions, and embeddings to span attributes**. This gives you a clear visibility into how your LLM application is working, and can make it easy to debug and evaluate the quality of the outputs.

However, you may want to disable this logging for privacy reasons, as they may contain highly sensitive data from your users. You may also simply want to reduce the size of your traces.

To disable logging, set the `TRACELOOP_TRACE_CONTENT` environment variable to `false`.

```bash
TRACELOOP_TRACE_CONTENT=false
```

## Captured Span Attributes

The instrumentation captures the following span attributes:

- `gen_ai.system`: "Dify"
- `gen_ai.request.type`: "chat" or "completion"
- `gen_ai.prompt`: The input query (if tracing content is enabled)
- `gen_ai.completion`: The response answer (if tracing content is enabled)
- `gen_ai.usage.prompt_tokens`: Number of tokens in the prompt
- `gen_ai.usage.completion_tokens`: Number of tokens in the completion
- `gen_ai.usage.total_tokens`: Total number of tokens used
- `gen_ai.response.id`: The message ID from Dify
- `gen_ai.response.conversation_id`: The conversation ID (for chat messages)
- `gen_ai.response.model`: The model used for generation

## Metrics

The instrumentation also records the following metrics:

- `gen_ai.client.operation.duration`: Duration of the LLM operation
- `gen_ai.client.token.usage`: Token usage for input and output
