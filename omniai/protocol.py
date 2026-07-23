"""Canonical internal message protocol shared by every gateway channel.

Every inbound payload (REST JSON, Discord webhook, WebSocket frame) is
normalized into an :class:`OmniMessage` before it touches the graph, and every
graph output is an ``OmniMessage`` until a channel adapter re-encodes it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Channel(StrEnum):
    REST = "rest"
    WEBSOCKET = "websocket"
    DISCORD = "discord"
    INTERNAL = "internal"


class ToolCall(BaseModel):
    """A structured request from the model to invoke a registered tool."""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class OmniMessage(BaseModel):
    """The canonical message that flows through gateways, graphs, and memory."""

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    session_id: str = "default"
    channel: Channel = Channel.INTERNAL
    role: Role = Role.USER
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def reply(self, content: str, role: Role = Role.ASSISTANT) -> OmniMessage:
        """Build a response message bound to the same session and channel."""
        return OmniMessage(
            session_id=self.session_id,
            channel=self.channel,
            role=role,
            content=content,
            metadata={"in_reply_to": self.id},
        )

    def to_openai(self) -> dict[str, Any]:
        """Render as an OpenAI chat-completions message dict."""
        return {"role": self.role.value, "content": self.content}
