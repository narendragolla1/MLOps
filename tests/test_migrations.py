import pathlib
import sqlite3

from alembic import command
from alembic.config import Config

from omniai.settings import reset_settings_cache

REPO_ROOT = pathlib.Path(__file__).parent.parent


def _alembic_config() -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    return cfg


def test_upgrade_head_creates_schema(tmp_path, monkeypatch):
    db = tmp_path / "migrated.db"
    monkeypatch.setenv("OMNIAI_DATABASE_URL", f"sqlite+aiosqlite:///{db}")
    reset_settings_cache()
    try:
        command.upgrade(_alembic_config(), "head")
    finally:
        reset_settings_cache()

    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "interactions" in tables
    assert "training_state" in tables
    assert "alembic_version" in tables
    columns = {r[1] for r in conn.execute("PRAGMA table_info(interactions)")}
    assert columns == {
        "id",
        "session_id",
        "channel",
        "role",
        "content",
        "tool_calls",
        "metadata_json",
        "created_at",
    }
    training_columns = {r[1] for r in conn.execute("PRAGMA table_info(training_state)")}
    assert training_columns == {"key", "value"}
    conn.close()


def test_migrated_schema_matches_model(tmp_path, monkeypatch):
    """The buffer must work against a migration-created database."""
    import asyncio

    from omniai.memory import InteractionBuffer
    from omniai.protocol import OmniMessage

    db = tmp_path / "compat.db"
    monkeypatch.setenv("OMNIAI_DATABASE_URL", f"sqlite+aiosqlite:///{db}")
    reset_settings_cache()
    try:
        command.upgrade(_alembic_config(), "head")
    finally:
        reset_settings_cache()

    async def use_buffer():
        buffer = InteractionBuffer(f"sqlite+aiosqlite:///{db}")
        await buffer.log(OmniMessage(content="post-migration"))
        rows = await buffer.fetch()
        await buffer.aclose()
        return rows

    rows = asyncio.run(use_buffer())
    assert rows[0]["content"] == "post-migration"


def test_watermark_persists_against_migrated_schema(tmp_path, monkeypatch):
    """The continuous-learning watermark must round-trip against training_state."""
    import asyncio
    from datetime import datetime

    from omniai.memory import InteractionBuffer

    db = tmp_path / "watermark.db"
    monkeypatch.setenv("OMNIAI_DATABASE_URL", f"sqlite+aiosqlite:///{db}")
    reset_settings_cache()
    try:
        command.upgrade(_alembic_config(), "head")
    finally:
        reset_settings_cache()

    async def use_buffer():
        buffer = InteractionBuffer(f"sqlite+aiosqlite:///{db}")
        assert await buffer.get_watermark() is None
        stamp = datetime(2026, 1, 1, 12, 0, 0)
        await buffer.set_watermark(stamp)
        watermark = await buffer.get_watermark()
        await buffer.aclose()
        return watermark

    assert asyncio.run(use_buffer()) == datetime(2026, 1, 1, 12, 0, 0)
