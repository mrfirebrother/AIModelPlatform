"""Make release/instance model references nullable for binding teardown

Binding teardown (DELETE /api/bindings/{id}) removes a binding's release history, and
model deletion clears historical (non-active) references — both need the FK columns to
accept NULL. Active releases and serving instances still block model deletion.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-19
"""
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("model_binding_releases", "model_node_id", nullable=True)
    op.alter_column("runtime_instances", "model_node_id", nullable=True)


def downgrade() -> None:
    # Fails on purpose if NULLs have accumulated - a downgrade must clean them first.
    op.alter_column("model_binding_releases", "model_node_id", nullable=False)
    op.alter_column("runtime_instances", "model_node_id", nullable=False)
