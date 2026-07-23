"""Interaction buffer: async, database-agnostic logging of gateway traffic.

Backed by SQLModel over SQLAlchemy's async engine, so the storage backend is
selected purely by URL — ``postgresql+asyncpg://`` in production,
``sqlite+aiosqlite://`` for zero-config dev, or any other async dialect.
Plain file paths are accepted for backward compatibility and treated as
SQLite databases.

Designed to be attached as a GatewayRouter observer — it receives every
inbound user message, tool output, and LLM response without blocking the
event loop.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from omniai.memory.models import Interaction, TrainingState
from omniai.protocol import OmniMessage

_WATERMARK_KEY = "watermark"


def _to_url(target: str | Path) -> str:
    target = str(target)
    if "://" in target:
        return target
    return f"sqlite+aiosqlite:///{target}"


class InteractionBuffer:
    """Async interaction log with a training-trigger threshold."""

    def __init__(
        self,
        database_url: str | Path = "sqlite+aiosqlite:///interactions.db",
        threshold: int | None = None,
        on_threshold: Callable[[], Awaitable[Any] | Any] | None = None,
    ):
        self.database_url = _to_url(database_url)
        self.threshold = threshold
        self.on_threshold = on_threshold
        self._engine: AsyncEngine | None = None
        self._ready_lock = asyncio.Lock()
        self._threshold_fired_at: int | None = None
        # Incrementally maintained so the threshold check never needs a
        # COUNT(*) round-trip per logged message. Seeded from the DB once;
        # slight drift on upserts is harmless for threshold purposes.
        self._approx_count: int = 0

    async def _ensure_ready(self) -> AsyncEngine:
        if self._engine is None:
            async with self._ready_lock:
                if self._engine is None:
                    engine = create_async_engine(self.database_url)
                    async with engine.begin() as conn:
                        await conn.run_sync(SQLModel.metadata.create_all)
                    self._engine = engine
        if self._threshold_fired_at is None:
            self._approx_count = await self._count()
            self._threshold_fired_at = self._approx_count
        return self._engine

    async def _count(self) -> int:
        from sqlalchemy import func

        async with AsyncSession(self._engine) as session:
            result = await session.exec(select(func.count()).select_from(Interaction))
            return int(result.one())

    @staticmethod
    def _to_row(message: OmniMessage) -> Interaction:
        # Store naive UTC: portable across TIMESTAMP flavors (asyncpg rejects
        # tz-aware values for TIMESTAMP WITHOUT TIME ZONE).
        created = message.created_at
        if created.tzinfo is not None:
            created = created.astimezone(UTC).replace(tzinfo=None)
        return Interaction(
            id=message.id,
            session_id=message.session_id,
            channel=message.channel.value,
            role=message.role.value,
            content=message.content,
            tool_calls=json.dumps([tc.model_dump() for tc in message.tool_calls]),
            metadata_json=json.dumps(message.metadata, default=str),
            created_at=created,
        )

    async def log(self, message: OmniMessage) -> None:
        """Persist a message; fires ``on_threshold`` when the bar is crossed."""
        await self._ensure_ready()
        async with AsyncSession(self._engine) as session:
            await session.merge(self._to_row(message))
            await session.commit()
        self._approx_count += 1
        if self.threshold is None or self.on_threshold is None:
            return
        if self._approx_count - (self._threshold_fired_at or 0) >= self.threshold:
            self._threshold_fired_at = self._approx_count
            result = self.on_threshold()
            if asyncio.iscoroutine(result):
                await result

    async def count(self) -> int:
        await self._ensure_ready()
        return await self._count()

    async def fetch(
        self,
        session_id: str | None = None,
        limit: int | None = None,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Rows as plain dicts, oldest first (keys match the legacy schema).

        ``since`` filters to rows created strictly after the given (naive
        UTC) timestamp — the incremental-training high-water mark.
        """
        await self._ensure_ready()
        query = select(Interaction)
        if session_id is not None:
            query = query.where(Interaction.session_id == session_id)
        if since is not None:
            if since.tzinfo is not None:
                since = since.astimezone(UTC).replace(tzinfo=None)
            query = query.where(Interaction.created_at > since)
        # SQLModel types class-level field access as the field's Python type
        # (datetime) rather than the SQLAlchemy column expression it actually
        # is at runtime; order_by needs the latter.
        query = query.order_by(Interaction.created_at, Interaction.id)  # type: ignore[arg-type]
        if limit is not None:
            query = query.limit(limit)
        async with AsyncSession(self._engine) as session:
            rows = (await session.exec(query)).all()
        return [
            {
                "id": r.id,
                "session_id": r.session_id,
                "channel": r.channel,
                "role": r.role,
                "content": r.content,
                "tool_calls": r.tool_calls,
                "metadata": r.metadata_json,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]

    async def get_watermark(self) -> datetime | None:
        """Timestamp of the last interaction consumed by a successful
        training cycle; None before the first cycle."""
        await self._ensure_ready()
        async with AsyncSession(self._engine) as session:
            row = await session.get(TrainingState, _WATERMARK_KEY)
        return datetime.fromisoformat(row.value) if row is not None else None

    async def set_watermark(self, timestamp: datetime) -> None:
        """Advance the watermark; persisted in the same database so restarts
        never re-train on already-consumed data."""
        await self._ensure_ready()
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(UTC).replace(tzinfo=None)
        async with AsyncSession(self._engine) as session:
            row = await session.get(TrainingState, _WATERMARK_KEY)
            if row is None:
                row = TrainingState(key=_WATERMARK_KEY, value=timestamp.isoformat())
            else:
                row.value = timestamp.isoformat()
            session.add(row)
            await session.commit()

    async def aclose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    def close(self) -> None:
        """Sync-friendly close; schedules disposal if a loop is running."""
        if self._engine is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
        else:
            loop.create_task(self.aclose())

    # Allows: GatewayRouter(observers=[buffer])
    async def __call__(self, message: OmniMessage) -> None:
        await self.log(message)
