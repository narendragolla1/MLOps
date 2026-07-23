"""Provider-neutral chat model interface.

Every provider adapter (OpenAI-compatible, Anthropic, the self-hosted
ModelEngine) implements :class:`ChatModel`, so graphs, agents, and
structured-output wrappers work identically across providers.

The canonical wire format for ``messages`` is the OpenAI chat shape:
``{"role": ..., "content": ...}``, assistant turns may carry ``tool_calls``,
and tool results use ``{"role": "tool", "tool_call_id": ..., "content": ...}``.
Adapters translate this to their provider's native format.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from omniai.protocol import OmniMessage, Role, ToolCall

if TYPE_CHECKING:
    from pydantic import BaseModel

    from omniai.graph.tools import Tool
    from omniai.models.structured import StructuredOutput


@dataclass
class ChatResult:
    """Normalized provider response."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    raw: Any = None


class ChatModel(abc.ABC):
    """Abstract chat model: one async call surface for every provider."""

    @abc.abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Run one chat completion; ``tools`` enables native tool calling."""

    def with_structured_output(
        self, schema: type[BaseModel], max_retries: int = 2
    ) -> StructuredOutput:
        """Wrap this model so responses are validated ``schema`` instances."""
        from omniai.models.structured import StructuredOutput

        return StructuredOutput(self, schema, max_retries=max_retries)


def omni_to_openai(messages: list[OmniMessage]) -> list[dict[str, Any]]:
    """Convert OmniMessage history into the canonical chat wire format."""
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role is Role.TOOL:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": message.metadata.get("tool_call_id", ""),
                    "content": message.content,
                }
            )
        elif message.role is Role.ASSISTANT and message.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )
        else:
            out.append({"role": message.role.value, "content": message.content})
    return out


def parse_openai_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Parse OpenAI-format tool_calls into protocol ToolCall objects."""
    calls: list[ToolCall] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        raw_args = fn.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            arguments = {"__unparsed__": raw_args}
        calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=arguments))
    return calls
