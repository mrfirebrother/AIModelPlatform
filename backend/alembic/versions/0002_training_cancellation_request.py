"""Add cancellation_requested field to training_tasks

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-24
"""
from alembic import op
import sqlalchemy as sa


revision = "0002"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "training_tasks",
        sa.Column(
            "cancellation_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )


def downgrade() -> None:
    op.drop_column("training_tasks", "cancellation_requested")
