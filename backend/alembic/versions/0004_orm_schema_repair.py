"""Repair ORM/migration drift: model_nodes.name, datasets.source_path, alerts, backup_records

The SQLAlchemy models gained columns and tables that never received a migration, so a
database created by ``alembic upgrade head`` was missing them and several routes failed
with UndefinedColumn / UndefinedTable. This revision brings the migrated schema back in
line with ``backend/app/models``.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-14
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


UUID = sa.Uuid(as_uuid=True)
JSONB = postgresql.JSONB()
UTC = sa.DateTime(timezone=True)
NOW = sa.text("now()")


def upgrade() -> None:
    # --- columns the ORM declares but the migrations never added -------------
    # model_nodes.name  (ModelNode.name, model-name feature)
    op.add_column("model_nodes", sa.Column("name", sa.String(255), nullable=True))
    op.execute("UPDATE model_nodes SET name = '' WHERE name IS NULL")
    op.alter_column("model_nodes", "name", nullable=False)

    # datasets.source_path  (Dataset.source_path, consumed by the async import flow)
    op.add_column("datasets", sa.Column("source_path", sa.String(1024), nullable=True))

    # --- tables the ORM declares but no migration created --------------------
    op.create_table(
        "alerts",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("alert_type", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(32), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("resource_type", sa.String(64)),
        sa.Column("resource_id", sa.String(255)),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("metadata_json", JSONB, nullable=False),
        sa.Column("acknowledged_at", UTC),
        sa.Column("acknowledged_by", sa.String(255)),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.Column("updated_at", UTC, server_default=NOW, nullable=False),
    )

    op.create_table(
        "backup_records",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("backup_type", sa.String(32), nullable=False),
        sa.Column("scope_json", JSONB, nullable=False),
        sa.Column("artifact_path", sa.String(1024)),
        sa.Column("artifact_hash", sa.String(255)),
        sa.Column("metadata_json", JSONB, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("verified", sa.Boolean(), nullable=False),
        sa.Column("verified_at", UTC),
        sa.Column("verify_hash", sa.String(255)),
        sa.Column("restore_target_id", UUID),
        sa.Column("error_summary", sa.Text()),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.Column("updated_at", UTC, server_default=NOW, nullable=False),
        sa.UniqueConstraint("name", name="uq_backup_records_name"),
        sa.ForeignKeyConstraint(
            ["restore_target_id"],
            ["backup_records.id"],
            name="fk_backup_records_restore_target_id_backup_records",
            ondelete="SET NULL",
        ),
    )


def downgrade() -> None:
    op.drop_table("backup_records")
    op.drop_table("alerts")
    op.drop_column("datasets", "source_path")
    op.drop_column("model_nodes", "name")
