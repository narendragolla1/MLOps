"""ChatModel adapter for OpenAI and any OpenAI-compatible endpoint.

Covers api.openai.com, Azure-style gateways, and self-hosted OpenAI-clone
servers (vLLM, SGLang, Ollama, ...) — pass the matching ``base_url``.
"""

from __future__ import annotations

from typing import Any

import httpx

from omniai.engine.resilience import with_retries
from omniai.graph.tools import Tool
from omniai.models.base import ChatModel, ChatResult, parse_openai_tool_calls


class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        retries: int = 3,
        client: httpx.AsyncClient | None = None,
        **default_params: Any,
    ):
        self.model = model
        self.retries = retries
        self.default_params = default_params
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )
        # Injected clients (tests, custom transports) still get auth headers.
        self.client.headers.update(headers)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            **self.default_params,
            **kwargs,
        }
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]

        async def attempt() -> httpx.Response:
            resp = await self.client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            return resp

        data = (await with_retries(attempt, attempts=self.retries)).json()
        choice = data["choices"][0]
        message = choice.get("message", {})
        return ChatResult(
            content=message.get("content") or "",
            tool_calls=parse_openai_tool_calls(message),
            usage=data.get("usage") or {},
            stop_reason=choice.get("finish_reason"),
            raw=data,
        )
