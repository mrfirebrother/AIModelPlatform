"""Add code field to model_nodes

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-29
"""
from alembic import op
import sqlalchemy as sa


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add code column (nullable first for backfill)
    op.add_column(
        "model_nodes",
        sa.Column("code", sa.String(20), nullable=True),
    )

    # Backfill existing rows with sequential codes
    conn = op.get_bind()
    rows = conn.execute(sa.text("SELECT id FROM model_nodes ORDER BY created_at")).fetchall()
    for i, (model_id,) in enumerate(rows, start=1):
        conn.execute(
            sa.text("UPDATE model_nodes SET code = :code WHERE id = :id"),
            {"code": f"M{i:03d}", "id": model_id},
        )

    # Make non-nullable and add unique constraint
    op.alter_column("model_nodes", "code", nullable=False)
    op.create_unique_constraint("uq_model_node_code", "model_nodes", ["code"])


def downgrade() -> None:
    op.drop_constraint("uq_model_node_code", "model_nodes", type_="unique")
    op.drop_column("model_nodes", "code")
