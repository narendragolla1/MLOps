"""Initial schema: interactions table.

Revision ID: 0001
Revises:
Create Date: 2026-07-23

"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "interactions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("channel", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("tool_calls", sa.String(), nullable=False, server_default="[]"),
        sa.Column("metadata_json", sa.String(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_interactions_session_id", "interactions", ["session_id"])
    op.create_index("ix_interactions_created_at", "interactions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_interactions_created_at", table_name="interactions")
    op.drop_index("ix_interactions_session_id", table_name="interactions")
    op.drop_table("interactions")
