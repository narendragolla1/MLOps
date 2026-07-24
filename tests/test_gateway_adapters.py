"""Unit tests for gateway channel adapters with error handling."""

import pytest

from omniai.gateway.adapters import (
    ChannelAdapter,
    DiscordAdapter,
    RestAdapter,
    WebSocketAdapter,
)
from omniai.protocol import Channel, OmniMessage, Role


# -- RestAdapter Tests ---------------------------------------------------


class TestRestAdapter:
    """Unit tests for RestAdapter."""

    def setup_method(self):
        self.adapter = RestAdapter()

    def test_channel_attribute(self):
        """Verify adapter has correct channel attribute."""
        assert self.adapter.channel is Channel.REST

    def test_to_omni_with_all_fields(self):
        """Verify to_omni handles all fields correctly."""
        payload = {
            "content": "hello",
            "session_id": "session123",
            "metadata": {"user_id": "alice", "version": 1},
        }
        message = self.adapter.to_omni(payload)
        assert message.content == "hello"
        assert message.session_id == "session123"
        assert message.metadata == {"user_id": "alice", "version": 1}
        assert message.role is Role.USER
        assert message.channel is Channel.REST

    def test_to_omni_missing_content(self):
        """Verify to_omni handles missing content."""
        payload = {"session_id": "s1"}
        message = self.adapter.to_omni(payload)
        assert message.content == ""
        assert message.session_id == "s1"

    def test_to_omni_missing_session_id(self):
        """Verify to_omni defaults session_id to 'default'."""
        payload = {"content": "hello"}
        message = self.adapter.to_omni(payload)
        assert message.session_id == "default"

    def test_to_omni_missing_metadata(self):
        """Verify to_omni defaults metadata to empty dict."""
        payload = {"content": "hello"}
        message = self.adapter.to_omni(payload)
        assert message.metadata == {}

    def test_to_omni_empty_payload(self):
        """Verify to_omni handles empty payload gracefully."""
        message = self.adapter.to_omni({})
        assert message.content == ""
        assert message.session_id == "default"
        assert message.metadata == {}

    def test_to_omni_null_fields(self):
        """Verify to_omni handles null fields."""
        payload = {"content": None, "session_id": None, "metadata": None}
        message = self.adapter.to_omni(payload)
        # None gets passed through, but defaults are used if falsy
        assert message.content is None or message.content == ""
        assert message.session_id == "default"
        assert message.metadata == {}

    def test_to_omni_extra_fields(self):
        """Verify to_omni ignores extra fields."""
        payload = {
            "content": "hello",
            "session_id": "s1",
            "metadata": {"key": "value"},
            "extra_field": "ignored",
            "another_extra": 123,
        }
        message = self.adapter.to_omni(payload)
        assert message.content == "hello"
        assert message.session_id == "s1"
        assert "extra_field" not in message.metadata

    def test_from_omni_basic(self):
        """Verify from_omni converts message correctly."""
        message = OmniMessage(
            content="response",
            session_id="s1",
            role=Role.ASSISTANT,
            metadata={"key": "value"},
        )
        payload = self.adapter.from_omni(message)
        assert payload["content"] == "response"
        assert payload["session_id"] == "s1"
        assert payload["role"] == "assistant"
        assert payload["metadata"] == {"key": "value"}
        assert "id" in payload

    def test_from_omni_preserves_message_id(self):
        """Verify from_omni includes message ID."""
        message = OmniMessage(content="hello", session_id="s1")
        payload = self.adapter.from_omni(message)
        assert payload["id"] == message.id

    def test_from_omni_with_complex_metadata(self):
        """Verify from_omni preserves complex metadata structures."""
        message = OmniMessage(
            content="hello",
            session_id="s1",
            metadata={
                "nested": {"data": {"deep": "value"}},
                "list": [1, 2, 3],
                "bool": True,
            },
        )
        payload = self.adapter.from_omni(message)
        assert payload["metadata"]["nested"]["data"]["deep"] == "value"
        assert payload["metadata"]["list"] == [1, 2, 3]
        assert payload["metadata"]["bool"] is True

    def test_round_trip_preservation(self):
        """Verify round-trip preserves data."""
        original_payload = {
            "content": "test message",
            "session_id": "session_abc",
            "metadata": {"source": "test"},
        }
        message = self.adapter.to_omni(original_payload)
        reconstructed = self.adapter.from_omni(message)
        assert reconstructed["content"] == original_payload["content"]
        assert reconstructed["session_id"] == original_payload["session_id"]
        assert reconstructed["metadata"] == original_payload["metadata"]

    def test_to_omni_large_content(self):
        """Verify to_omni handles large content."""
        large_content = "x" * 100000
        payload = {"content": large_content}
        message = self.adapter.to_omni(payload)
        assert len(message.content) == 100000

    def test_to_omni_special_characters(self):
        """Verify to_omni preserves special characters."""
        special_content = "Hello\nWorld\t🚀\n日本語\r\nSpecial: @#$%^&*()"
        payload = {"content": special_content}
        message = self.adapter.to_omni(payload)
        assert message.content == special_content

    def test_to_omni_unicode_session_id(self):
        """Verify to_omni handles unicode in session_id."""
        payload = {"content": "hello", "session_id": "session_🔑_café"}
        message = self.adapter.to_omni(payload)
        assert message.session_id == "session_🔑_café"


# -- WebSocketAdapter Tests -----------------------------------------------


class TestWebSocketAdapter:
    """Unit tests for WebSocketAdapter."""

    def setup_method(self):
        self.adapter = WebSocketAdapter()

    def test_channel_attribute(self):
        """Verify WebSocketAdapter has correct channel."""
        assert self.adapter.channel is Channel.WEBSOCKET

    def test_inherits_rest_behavior(self):
        """Verify WebSocketAdapter inherits RestAdapter functionality."""
        assert isinstance(self.adapter, RestAdapter)
        payload = {"content": "ws message", "session_id": "ws_session"}
        message = self.adapter.to_omni(payload)
        assert message.content == "ws message"
        assert message.session_id == "ws_session"
        assert message.channel is Channel.WEBSOCKET

    def test_from_omni_same_as_rest(self):
        """Verify from_omni behaves identically to RestAdapter."""
        message = OmniMessage(content="response", session_id="s1")
        ws_payload = self.adapter.from_omni(message)

        rest_adapter = RestAdapter()
        rest_payload = rest_adapter.from_omni(message)

        # Should be identical except for channel
        assert ws_payload["content"] == rest_payload["content"]
        assert ws_payload["session_id"] == rest_payload["session_id"]


# -- DiscordAdapter Tests -----------------------------------------------


class TestDiscordAdapter:
    """Unit tests for DiscordAdapter."""

    def setup_method(self):
        self.adapter = DiscordAdapter()

    def test_channel_attribute(self):
        """Verify adapter has correct channel attribute."""
        assert self.adapter.channel is Channel.DISCORD

    def test_to_omni_with_author_field(self):
        """Verify to_omni extracts author from author field."""
        payload = {
            "content": "discord message",
            "channel_id": "12345",
            "author": {"id": "user123", "username": "alice"},
        }
        message = self.adapter.to_omni(payload)
        assert message.content == "discord message"
        assert message.session_id == "12345"
        assert message.metadata["author_id"] == "user123"
        assert message.metadata["author_name"] == "alice"

    def test_to_omni_with_member_field(self):
        """Verify to_omni extracts author from member field."""
        payload = {
            "content": "interaction message",
            "channel_id": "12345",
            "member": {"user": {"id": "user456", "username": "bob"}},
        }
        message = self.adapter.to_omni(payload)
        assert message.metadata["author_id"] == "user456"
        assert message.metadata["author_name"] == "bob"

    def test_to_omni_author_precedence(self):
        """Verify author field takes precedence over member field."""
        payload = {
            "content": "message",
            "channel_id": "12345",
            "author": {"id": "author_id", "username": "author_name"},
            "member": {"user": {"id": "member_id", "username": "member_name"}},
        }
        message = self.adapter.to_omni(payload)
        # author field should be used
        assert message.metadata["author_id"] == "author_id"
        assert message.metadata["author_name"] == "author_name"

    def test_to_omni_missing_author(self):
        """Verify to_omni handles missing author gracefully."""
        payload = {"content": "message", "channel_id": "12345"}
        message = self.adapter.to_omni(payload)
        assert message.metadata["author_id"] is None
        assert message.metadata["author_name"] is None

    def test_to_omni_missing_guild_id(self):
        """Verify to_omni handles missing guild_id."""
        payload = {
            "content": "message",
            "channel_id": "12345",
            "author": {"id": "user1", "username": "alice"},
        }
        message = self.adapter.to_omni(payload)
        assert message.metadata["guild_id"] is None

    def test_to_omni_with_guild_id(self):
        """Verify to_omni includes guild_id."""
        payload = {
            "content": "message",
            "channel_id": "12345",
            "guild_id": "guild_xyz",
            "author": {"id": "user1", "username": "alice"},
        }
        message = self.adapter.to_omni(payload)
        assert message.metadata["guild_id"] == "guild_xyz"

    def test_to_omni_channel_id_conversion(self):
        """Verify to_omni converts channel_id to string."""
        payload = {
            "content": "message",
            "channel_id": 123456789,  # numeric
            "author": {"id": "user1", "username": "alice"},
        }
        message = self.adapter.to_omni(payload)
        assert message.session_id == "123456789"
        assert isinstance(message.session_id, str)

    def test_to_omni_missing_channel_id(self):
        """Verify to_omni defaults channel_id to 'discord'."""
        payload = {
            "content": "message",
            "author": {"id": "user1", "username": "alice"},
        }
        message = self.adapter.to_omni(payload)
        assert message.session_id == "discord"

    def test_to_omni_discord_message_id(self):
        """Verify to_omni captures Discord message ID."""
        payload = {
            "id": "msg_123",
            "content": "message",
            "channel_id": "12345",
            "author": {"id": "user1", "username": "alice"},
        }
        message = self.adapter.to_omni(payload)
        assert message.metadata["discord_message_id"] == "msg_123"

    def test_to_omni_empty_author(self):
        """Verify to_omni handles empty author dict."""
        payload = {
            "content": "message",
            "channel_id": "12345",
            "author": {},
        }
        message = self.adapter.to_omni(payload)
        assert message.metadata["author_id"] is None
        assert message.metadata["author_name"] is None

    def test_to_omni_null_author(self):
        """Verify to_omni handles null author."""
        payload = {
            "content": "message",
            "channel_id": "12345",
            "author": None,
        }
        message = self.adapter.to_omni(payload)
        # None or {} fallback
        assert message.metadata["author_id"] is None or message.metadata["author_id"] is None

    def test_from_omni_truncates_at_2000_chars(self):
        """Verify from_omni enforces Discord's 2000 character limit."""
        long_content = "x" * 3000
        message = OmniMessage(content=long_content, session_id="s1")
        payload = self.adapter.from_omni(message)
        assert len(payload["content"]) == 2000
        assert payload["content"] == "x" * 2000

    def test_from_omni_exact_2000_chars(self):
        """Verify from_omni preserves exactly 2000 characters."""
        content_2000 = "a" * 2000
        message = OmniMessage(content=content_2000, session_id="s1")
        payload = self.adapter.from_omni(message)
        assert payload["content"] == content_2000
        assert len(payload["content"]) == 2000

    def test_from_omni_under_2000_chars(self):
        """Verify from_omni doesn't truncate content under 2000 chars."""
        content = "Hello World" * 50  # 550 chars
        message = OmniMessage(content=content, session_id="s1")
        payload = self.adapter.from_omni(message)
        assert payload["content"] == content
        assert len(payload["content"]) < 2000

    def test_from_omni_truncates_unicode_safely(self):
        """Verify from_omni truncates unicode content safely."""
        # Mix of ASCII and emoji
        content = ("Hello " * 300) + "🚀" * 100  # Will exceed 2000 when combined
        message = OmniMessage(content=content, session_id="s1")
        payload = self.adapter.from_omni(message)
        assert len(payload["content"]) == 2000
        # Should not be broken UTF-8
        try:
            payload["content"].encode("utf-8").decode("utf-8")
        except UnicodeDecodeError:
            pytest.fail("Truncated content is not valid UTF-8")

    def test_to_omni_metadata_keys_present(self):
        """Verify to_omni always includes all metadata keys."""
        payload = {"content": "x", "channel_id": "1", "id": "2"}
        message = self.adapter.to_omni(payload)
        required_keys = {
            "discord_message_id",
            "author_id",
            "author_name",
            "guild_id",
        }
        assert required_keys.issubset(message.metadata.keys())

    def test_to_omni_missing_content(self):
        """Verify to_omni handles missing content."""
        payload = {
            "channel_id": "12345",
            "author": {"id": "user1", "username": "alice"},
        }
        message = self.adapter.to_omni(payload)
        assert message.content == ""

    def test_to_omni_role_is_user(self):
        """Verify to_omni always sets role to USER."""
        payload = {
            "content": "message",
            "channel_id": "12345",
            "author": {"id": "user1", "username": "alice"},
        }
        message = self.adapter.to_omni(payload)
        assert message.role is Role.USER

    def test_from_omni_only_returns_content(self):
        """Verify from_omni only returns content field (for Discord webhook)."""
        message = OmniMessage(
            content="response",
            session_id="s1",
            metadata={"key": "value"},
            role=Role.ASSISTANT,
        )
        payload = self.adapter.from_omni(message)
        # Discord webhook response only needs content
        assert "content" in payload
        assert len(payload) == 1
        assert payload["content"] == "response"

    def test_special_discord_fields(self):
        """Verify Discord-specific fields are captured."""
        payload = {
            "id": "msg_id_123",
            "content": "message",
            "channel_id": "chan_123",
            "guild_id": "guild_123",
            "author": {"id": "user_123", "username": "alice"},
        }
        message = self.adapter.to_omni(payload)
        assert message.metadata["discord_message_id"] == "msg_id_123"
        assert message.metadata["author_id"] == "user_123"
        assert message.session_id == "chan_123"
        assert message.metadata["guild_id"] == "guild_123"


# -- Adapter Integration Tests -------------------------------------------


class TestAdapterComparison:
    """Compare adapter behaviors."""

    def test_rest_and_websocket_same_except_channel(self):
        """Verify REST and WebSocket differ only in channel."""
        payload = {
            "content": "message",
            "session_id": "s1",
            "metadata": {"key": "value"},
        }

        rest_adapter = RestAdapter()
        ws_adapter = WebSocketAdapter()

        rest_msg = rest_adapter.to_omni(payload)
        ws_msg = ws_adapter.to_omni(payload)

        assert rest_msg.channel is Channel.REST
        assert ws_msg.channel is Channel.WEBSOCKET
        assert rest_msg.content == ws_msg.content
        assert rest_msg.session_id == ws_msg.session_id
        assert rest_msg.metadata == ws_msg.metadata

    def test_discord_extraction_edge_cases(self):
        """Verify Discord adapter handles edge cases."""
        test_cases = [
            {
                "payload": {},
                "expected_session_id": "discord",
                "expected_author_id": None,
            },
            {
                "payload": {"channel_id": "0"},
                "expected_session_id": "0",
                "expected_author_id": None,
            },
            {
                "payload": {
                    "channel_id": "123",
                    "author": {"id": "", "username": ""},
                },
                "expected_session_id": "123",
                "expected_author_id": "",
            },
        ]

        adapter = DiscordAdapter()
        for case in test_cases:
            message = adapter.to_omni(case["payload"])
            assert message.session_id == case["expected_session_id"]
            assert message.metadata["author_id"] == case["expected_author_id"]
