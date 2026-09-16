"""Drop the evaluation review columns (no review mechanism in this platform)

The product owner's call: the quality gate produced a pass/fail label nobody acted on and
nobody owns a reviewer role, so the human-review workflow was removed (see the 2026-09-16
note in AGENTS.md — the ``POST /api/evaluations/{id}/review`` endpoint and
``evaluation_service.record_human_review`` are gone). ``Evaluation`` no longer declares these
columns, and leaving them in the schema invites future readers to assume a review step exists.

Dropped: ``human_status``, ``reviewer``, ``human_conclusion``, ``human_comments``,
``reviewed_at`` and the ``evaluation_human_status`` CHECK constraint.

``auto_status`` is deliberately kept — it is the evaluation pipeline state consumed by
``evaluation_worker``, the ``sweep-pending-evaluations`` job and ``model_diff``.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-16
"""
from alembic import op
import sqlalchemy as sa


revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

UTC = sa.DateTime(timezone=True)


def upgrade() -> None:
    # The CHECK constraint ended up double-prefixed in a migrated database
    # (`ck_evaluations_ck_evaluations_evaluation_human_status`, because 0001 names it
    # `ck_evaluations_…` and the metadata naming convention adds its own prefix). Drop every
    # plausible spelling, otherwise this revision fails on databases created either way.
    for name in (
        "ck_evaluations_ck_evaluations_evaluation_human_status",
        "ck_evaluations_evaluation_human_status",
        "evaluation_human_status",
    ):
        op.execute(f"ALTER TABLE evaluations DROP CONSTRAINT IF EXISTS {name}")

    op.drop_column("evaluations", "human_status")
    op.drop_column("evaluations", "reviewer")
    op.drop_column("evaluations", "human_conclusion")
    op.drop_column("evaluations", "human_comments")
    op.drop_column("evaluations", "reviewed_at")


def downgrade() -> None:
    # Restores the columns so the revision is reversible; the data they held is gone, so
    # human_status comes back with the old default and nothing repopulates it.
    op.add_column(
        "evaluations",
        sa.Column("human_status", sa.String(32), nullable=False, server_default="pending"),
    )
    op.add_column("evaluations", sa.Column("reviewer", sa.String(255), nullable=True))
    op.add_column("evaluations", sa.Column("human_conclusion", sa.String(64), nullable=True))
    op.add_column("evaluations", sa.Column("human_comments", sa.Text(), nullable=True))
    op.add_column("evaluations", sa.Column("reviewed_at", UTC, nullable=True))
    op.execute(
        "ALTER TABLE evaluations ADD CONSTRAINT "
        "ck_evaluations_ck_evaluations_evaluation_human_status "
        "CHECK (human_status IN ('pending', 'passed', 'failed'))"
    )
