# OpenTelemetry Dify Instrumentation

<a href="https://pypi.org/project/opentelemetry-instrumentation-dify/">
    <img src="https://badge.fury.io/py/opentelemetry-instrumentation-dify.svg">
</a>

This library allows tracing LLM applications built with [Dify](https://github.com/langgenius/dify), supporting both:
- **Dify Client SDK** (`dify-client` package) - for client-side API calls
- **Dify Server v1.8.1+** - for server-side operations including LLM calls, workflows, RAG, embeddings, and tools

## Installation

```bash
pip install opentelemetry-instrumentation-dify
```

## Quick Start

```python
from opentelemetry.instrumentation.dify import DifyInstrumentor

# Initialize instrumentation - automatically detects available components
DifyInstrumentor().instrument()
```

## Usage with Dify Client SDK

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

## Usage with Dify Server (v1.8.1+)

When running Dify server, the instrumentation automatically traces server-side operations:

### LLM Model Invocations

All LLM calls made through Dify's model runtime are automatically traced:

```python
from opentelemetry.instrumentation.dify import DifyInstrumentor

# Initialize instrumentation
DifyInstrumentor().instrument()

# Your Dify server code - LLM calls are automatically traced
# Example: core.model_runtime.model_providers.openai.llm.OpenAILargeLanguageModel
```

**Captured attributes:**
- Model name and provider
- Token usage (input/output)
- Request parameters
- Response metadata

### Workflow Execution

Workflow runs and node executions are traced:

```python
# Workflows are automatically traced when executed
# Span name: dify.workflow
# Attributes: workflow ID, node types, execution path
```

### RAG/Knowledge Base Retrieval

Document retrieval and vector search operations are traced:

```python
# RAG operations are automatically traced
# Span name: dify.rag.retrieve
# Attributes: query, top_k, document count, retrieval method
```

### Embedding Generation

Text embedding operations are traced:

```python
# Embedding generation is automatically traced
# Span name: dify.embedding.generate
# Attributes: model, text count, dimensions, token usage
```

### Tool Execution

Tool and agent invocations are traced:

```python
# Tool calls are automatically traced
# Span name: dify.tool.{tool_name}
# Attributes: tool name, parameters, results
```

## Server-Side Span Attributes

In addition to client SDK attributes, server-side instrumentation captures:

### LLM Operations
- `gen_ai.request.model`: Model name (e.g., "gpt-4")
- `gen_ai.provider`: Model provider (e.g., "openai")
- `gen_ai.usage.prompt_tokens`: Input tokens
- `gen_ai.usage.completion_tokens`: Output tokens

### Workflow Operations
- `gen_ai.operation.type`: "workflow"
- `gen_ai.workflow.id`: Workflow identifier
- Nested spans for each workflow node

### RAG Operations
- `gen_ai.operation.type`: "rag"
- `gen_ai.rag.query`: Search query
- `gen_ai.rag.top_k`: Number of documents requested
- `gen_ai.rag.document_count`: Number of documents retrieved

### Embedding Operations
- `gen_ai.operation.type`: "embedding"
- `gen_ai.embedding.text_count`: Number of texts embedded
- `gen_ai.embedding.dimensions`: Embedding vector dimensions

### Tool Operations
- `gen_ai.operation.type`: "tool"
- `gen_ai.tool.name`: Tool name
- `gen_ai.tool.parameters`: Tool input parameters

