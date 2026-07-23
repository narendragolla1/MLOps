from omniai.protocol import Channel, OmniMessage, Role, ToolCall


def test_message_defaults():
    msg = OmniMessage(content="hello")
    assert msg.role is Role.USER
    assert msg.channel is Channel.INTERNAL
    assert msg.id.startswith("msg_")
    assert msg.session_id == "default"
    assert msg.tool_calls == []


def test_reply_binds_session_and_channel():
    msg = OmniMessage(content="hi", session_id="s1", channel=Channel.DISCORD)
    reply = msg.reply("hello back")
    assert reply.session_id == "s1"
    assert reply.channel is Channel.DISCORD
    assert reply.role is Role.ASSISTANT
    assert reply.metadata["in_reply_to"] == msg.id


def test_openai_rendering():
    msg = OmniMessage(content="x", role=Role.SYSTEM)
    assert msg.to_openai() == {"role": "system", "content": "x"}


def test_tool_call_ids_unique():
    a, b = ToolCall(name="t"), ToolCall(name="t")
    assert a.id != b.id


def test_json_round_trip():
    msg = OmniMessage(content="hi", tool_calls=[ToolCall(name="search", arguments={"q": "x"})])
    restored = OmniMessage.model_validate_json(msg.model_dump_json())
    assert restored == msg
