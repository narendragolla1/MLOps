"""Interaction buffer: async SQL logging of every message through the gateway.

Designed to be attached as a GatewayRouter observer — it receives every
inbound user message, tool output, and LLM response. SQLite writes run in a
worker thread so the event loop never blocks; the schema is plain SQL so a
Postgres DSN can back the same interface later.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Awaitable, Callable

from omniai.protocol import OmniMessage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_calls TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions (session_id, created_at);
"""


class InteractionBuffer:
    """Async-friendly interaction log with a training-trigger threshold."""

    def __init__(
        self,
        db_path: str | Path = "interactions.db",
        threshold: int | None = None,
        on_threshold: Callable[[], Awaitable[Any] | Any] | None = None,
    ):
        self.db_path = str(db_path)
        self.threshold = threshold
        self.on_threshold = on_threshold
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._threshold_fired_at = self._count_sync()

    # -- sync core (runs in worker threads) ---------------------------------

    def _count_sync(self) -> int:
        with self._lock:
            (n,) = self._conn.execute("SELECT COUNT(*) FROM interactions").fetchone()
        return n

    def _insert_sync(self, message: OmniMessage) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO interactions "
                "(id, session_id, channel, role, content, tool_calls, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message.id,
                    message.session_id,
                    message.channel.value,
                    message.role.value,
                    message.content,
                    json.dumps([tc.model_dump() for tc in message.tool_calls]),
                    json.dumps(message.metadata, default=str),
                    message.created_at.isoformat(),
                ),
            )
            self._conn.commit()

    def _fetch_sync(self, session_id: str | None, limit: int | None) -> list[dict[str, Any]]:
        query = "SELECT * FROM interactions"
        params: list[Any] = []
        if session_id is not None:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # -- async API ----------------------------------------------------------

    async def log(self, message: OmniMessage) -> None:
        """Persist a message; fires ``on_threshold`` when the bar is crossed."""
        await asyncio.to_thread(self._insert_sync, message)
        if self.threshold is None or self.on_threshold is None:
            return
        count = await asyncio.to_thread(self._count_sync)
        if count - self._threshold_fired_at >= self.threshold:
            self._threshold_fired_at = count
            result = self.on_threshold()
            if asyncio.iscoroutine(result):
                await result

    async def count(self) -> int:
        return await asyncio.to_thread(self._count_sync)

    async def fetch(
        self, session_id: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._fetch_sync, session_id, limit)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # Allows: GatewayRouter(observers=[buffer])
    async def __call__(self, message: OmniMessage) -> None:
        await self.log(message)
