"""ChatModel adapter over the self-hosted ModelEngine.

Delegates to :class:`omniai.engine.ModelEngine`, inheriting its circuit
breaker, retries, backpressure semaphore, token metrics, and LoRA routing —
the graph/agent layer sees the same ChatModel surface as cloud providers.
"""

from __future__ import annotations

from typing import Any

from omniai.engine.engine import ModelEngine
from omniai.graph.tools import Tool
from omniai.models.base import ChatModel, ChatResult, parse_openai_tool_calls


class EngineChatModel(ChatModel):
    def __init__(self, engine: ModelEngine):
        self.engine = engine

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if tools:
            kwargs["tools"] = [t.to_openai() for t in tools]
        data = await self.engine.chat(messages, **kwargs)
        choice = data["choices"][0]
        message = choice.get("message", {})
        return ChatResult(
            content=message.get("content") or "",
            tool_calls=parse_openai_tool_calls(message),
            usage=data.get("usage") or {},
            stop_reason=choice.get("finish_reason"),
            raw=data,
        )
