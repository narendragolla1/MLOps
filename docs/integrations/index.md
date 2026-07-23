# Integrations

Model providers available behind the [`ChatModel`](../concepts/chat_models.md) interface. All support native tool calling, [structured output](../how_to/structured_output.md), and the [agent executor](../tutorials/build_an_agent.md).

| Provider | Class | Notes |
| --- | --- | --- |
| [OpenAI](openai.md) | `OpenAIChatModel` | Also covers **any OpenAI-compatible endpoint** (Azure gateways, Ollama, vLLM/SGLang servers) via `base_url`. |
| [Anthropic](anthropic.md) | `AnthropicChatModel` | Messages API with full tool_use / tool_result translation. |
| [Self-hosted](self_hosted.md) | `EngineChatModel` | Your own vLLM/SGLang deployment through `ModelEngine`, with its reliability layer and LoRA routing. |

Something else? [Write a custom chat model](../how_to/custom_chat_model.md) — it's one class with one method.
