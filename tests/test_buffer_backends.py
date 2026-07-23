"""DB-agnosticism tests: same buffer suite against multiple backends.

Locally runs on SQLite; in CI a second pass runs against a real Postgres
service container via OMNIAI_TEST_DATABASE_URL.
"""

import os

from omniai.memory import InteractionBuffer
from omniai.protocol import OmniMessage, Role


def _backends(tmp_path):
    urls = [f"sqlite+aiosqlite:///{tmp_path}/agnostic.db"]
    if extra := os.environ.get("OMNIAI_TEST_DATABASE_URL"):
        urls.append(extra)
    return urls


async def test_round_trip_on_every_backend(tmp_path):
    for url in _backends(tmp_path):
        buffer = InteractionBuffer(url)
        try:
            await buffer.log(OmniMessage(content="q", session_id="bk", role=Role.USER))
            await buffer.log(OmniMessage(content="a", session_id="bk", role=Role.ASSISTANT))
            rows = await buffer.fetch(session_id="bk")
            assert [r["role"] for r in rows] == ["user", "assistant"], url
            assert rows[0]["content"] == "q"
            assert await buffer.count() >= 2
        finally:
            await buffer.aclose()


async def test_threshold_on_every_backend(tmp_path):
    for url in _backends(tmp_path):
        fired = []
        buffer = InteractionBuffer(url, threshold=2, on_threshold=lambda: fired.append(1))
        try:
            for i in range(4):
                await buffer.log(OmniMessage(content=str(i), session_id="thr"))
            assert len(fired) == 2, url
        finally:
            await buffer.aclose()


async def test_plain_path_is_treated_as_sqlite(tmp_path):
    buffer = InteractionBuffer(tmp_path / "legacy.db")
    assert buffer.database_url.startswith("sqlite+aiosqlite:///")
    await buffer.log(OmniMessage(content="x"))
    assert await buffer.count() == 1
    await buffer.aclose()


async def test_merge_makes_relogging_idempotent(tmp_path):
    buffer = InteractionBuffer(tmp_path / "idem.db")
    msg = OmniMessage(content="once")
    await buffer.log(msg)
    await buffer.log(msg)  # same primary key: upsert, not duplicate
    assert await buffer.count() == 1
    await buffer.aclose()
