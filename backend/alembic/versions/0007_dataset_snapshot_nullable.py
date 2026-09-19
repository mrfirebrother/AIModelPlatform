"""Make snapshot references nullable for dataset deletion

Deleting a dataset clears historical references (training tasks, checkpoints,
evaluations, model nodes keep their rows; the snapshot pointer becomes NULL).
The FK columns must accept NULL for that.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-19
"""
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("training_tasks", "dataset_snapshot_id", nullable=True)
    op.alter_column("checkpoints", "dataset_snapshot_id", nullable=True)
    op.alter_column("evaluations", "dataset_snapshot_id", nullable=True)


def downgrade() -> None:
    # Fails on purpose if NULLs have accumulated - a downgrade must clean them first.
    op.alter_column("training_tasks", "dataset_snapshot_id", nullable=False)
    op.alter_column("checkpoints", "dataset_snapshot_id", nullable=False)
    op.alter_column("evaluations", "dataset_snapshot_id", nullable=False)
