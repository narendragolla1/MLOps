"""SQLModel table definitions for the memory subsystem.

Only portable column types are used, so the schema works unchanged on any
SQLAlchemy async dialect — Postgres (asyncpg) in production, SQLite
(aiosqlite) for zero-config dev, MySQL (asyncmy), etc. The backend is chosen
purely by the database URL.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class Interaction(SQLModel, table=True):
    __tablename__ = "interactions"

    id: str = Field(primary_key=True)
    session_id: str = Field(index=True)
    channel: str
    role: str
    content: str
    tool_calls: str = "[]"  # JSON-encoded; TEXT keeps the schema portable
    # "metadata" is reserved by SQLModel/SQLAlchemy, hence the _json suffix.
    metadata_json: str = "{}"
    created_at: datetime = Field(index=True)


class TrainingState(SQLModel, table=True):
    """Small key/value store for continuous-learning bookkeeping.

    Holds the incremental-training watermark (the timestamp of the last
    interaction consumed by a successful cycle) so a restarted process never
    re-trains on already-consumed data.
    """

    __tablename__ = "training_state"

    key: str = Field(primary_key=True)
    value: str
