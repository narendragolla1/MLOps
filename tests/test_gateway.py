from fastapi.testclient import TestClient

from omniai.gateway import GatewayRouter
from omniai.gateway.router import GuardrailViolation
from omniai.protocol import Channel, OmniMessage, Role


def echo_handler(message: OmniMessage) -> OmniMessage:
    return message.reply(f"echo: {message.content}")


def make_router(**kwargs) -> GatewayRouter:
    return GatewayRouter(handler=echo_handler, **kwargs)


def test_health():
    client = TestClient(make_router().app)
    assert client.get("/health").json() == {"status": "ok"}


def test_rest_round_trip():
    client = TestClient(make_router().app)
    resp = client.post("/v1/messages", json={"content": "hi", "session_id": "s9"})
    body = resp.json()
    assert resp.status_code == 200
    assert body["content"] == "echo: hi"
    assert body["session_id"] == "s9"
    assert body["role"] == "assistant"


def test_websocket_round_trip():
    client = TestClient(make_router().app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"content": "ping"})
        assert ws.receive_json()["content"] == "echo: ping"
        ws.send_json({"content": "again"})
        assert ws.receive_json()["content"] == "echo: again"


def test_discord_adapter_translation():
    client = TestClient(make_router().app)
    payload = {
        "id": "111",
        "content": "hello bot",
        "channel_id": "222",
        "author": {"id": "333", "username": "alice"},
    }
    resp = client.post("/discord/webhook", json=payload)
    assert resp.json() == {"content": "echo: hello bot"}


async def test_async_handler_supported():
    async def async_handler(message: OmniMessage) -> OmniMessage:
        return message.reply("async!")

    router = GatewayRouter(handler=async_handler)
    reply = await router.dispatch(OmniMessage(content="x"))
    assert reply.content == "async!"


def test_interceptor_blocks_message():
    def guard(message: OmniMessage) -> OmniMessage:
        if "attack" in message.content:
            raise GuardrailViolation("blocked: injection attempt")
        return message

    client = TestClient(make_router(interceptors=[guard]).app)
    resp = client.post("/v1/messages", json={"content": "attack payload"})
    assert resp.status_code == 400
    assert "blocked" in resp.json()["detail"]
    assert client.post("/v1/messages", json={"content": "benign"}).status_code == 200


def test_observers_see_inbound_and_outbound():
    seen: list[tuple[Role, str]] = []
    router = make_router(observers=[lambda m: seen.append((m.role, m.content))])
    client = TestClient(router.app)
    client.post("/v1/messages", json={"content": "hi"})
    assert (Role.USER, "hi") in seen
    assert (Role.ASSISTANT, "echo: hi") in seen


def test_channel_tagging():
    captured: list[OmniMessage] = []

    def handler(message: OmniMessage) -> OmniMessage:
        captured.append(message)
        return message.reply("ok")

    client = TestClient(GatewayRouter(handler=handler).app)
    client.post("/v1/messages", json={"content": "a"})
    client.post("/discord/webhook", json={"content": "b", "channel_id": "1"})
    assert captured[0].channel is Channel.REST
    assert captured[1].channel is Channel.DISCORD
