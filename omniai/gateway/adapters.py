"""Channel adapters: translate native payloads to/from OmniMessage.

Each adapter is a pure codec — no I/O — which keeps the gateway routes thin
and makes the translation logic trivially testable.
"""

from __future__ import annotations

import abc
from typing import Any

from omniai.protocol import Channel, OmniMessage, Role


class ChannelAdapter(abc.ABC):
    """Converts a channel's native payload format to/from OmniMessage."""

    channel: Channel

    @abc.abstractmethod
    def to_omni(self, payload: dict[str, Any]) -> OmniMessage:
        """Decode an inbound native payload into the canonical message."""

    @abc.abstractmethod
    def from_omni(self, message: OmniMessage) -> dict[str, Any]:
        """Encode an outbound canonical message into the native format."""


class RestAdapter(ChannelAdapter):
    """Plain JSON: ``{"content": ..., "session_id": ..., "metadata": ...}``."""

    channel = Channel.REST

    def to_omni(self, payload: dict[str, Any]) -> OmniMessage:
        return OmniMessage(
            channel=self.channel,
            role=Role.USER,
            content=payload.get("content", ""),
            session_id=payload.get("session_id", "default"),
            metadata=payload.get("metadata", {}),
        )

    def from_omni(self, message: OmniMessage) -> dict[str, Any]:
        return {
            "id": message.id,
            "session_id": message.session_id,
            "role": message.role.value,
            "content": message.content,
            "metadata": message.metadata,
        }


class WebSocketAdapter(RestAdapter):
    """Same JSON shape as REST, delivered over a persistent socket."""

    channel = Channel.WEBSOCKET


class DiscordAdapter(ChannelAdapter):
    """Discord interaction/webhook payloads."""

    channel = Channel.DISCORD

    def to_omni(self, payload: dict[str, Any]) -> OmniMessage:
        author = payload.get("author", {}) or payload.get("member", {}).get("user", {})
        return OmniMessage(
            channel=self.channel,
            role=Role.USER,
            content=payload.get("content", ""),
            session_id=str(payload.get("channel_id", "discord")),
            metadata={
                "discord_message_id": payload.get("id"),
                "author_id": author.get("id"),
                "author_name": author.get("username"),
                "guild_id": payload.get("guild_id"),
            },
        )

    def from_omni(self, message: OmniMessage) -> dict[str, Any]:
        # Discord hard-caps message content at 2000 characters.
        return {"content": message.content[:2000]}
