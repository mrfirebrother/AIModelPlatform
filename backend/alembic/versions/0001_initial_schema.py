"""Create the PostgreSQL metadata schema for the model platform."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial_schema"
down_revision = "0000_empty"
branch_labels = None
depends_on = None


UUID = sa.Uuid(as_uuid=True)
JSONB = postgresql.JSONB()
UTC = sa.DateTime(timezone=True)
NOW = sa.text("now()")


def upgrade() -> None:
    # Cross-table semantic gates are intentionally not encoded here. Task 4/5/7
    # services must validate model/snapshot schemas, checkpoint/task snapshots,
    # and binding activation pointers in one transaction. Direct database writes
    # that bypass those services are unsupported even when foreign keys pass.
    op.create_table(
        "label_schemas",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("parent_schema_id", UUID),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("definition_sealed", sa.Boolean(), nullable=False),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.Column("updated_at", UTC, server_default=NOW, nullable=False),
        sa.CheckConstraint(
            "status IN ('active', 'archived')", name="ck_label_schemas_label_schema_status"
        ),
        sa.ForeignKeyConstraint(
            ["parent_schema_id"], ["label_schemas.id"],
            name="fk_label_schemas_parent_schema_id_label_schemas",
            ondelete="RESTRICT",
        ),
    )
    op.create_table(
        "label_schema_classes",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("schema_id", UUID, nullable=False),
        sa.Column("class_id", sa.Integer(), nullable=False),
        sa.Column("semantic_key", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("definition_sealed", sa.Boolean(), nullable=False),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.UniqueConstraint(
            "schema_id", "class_id", name="uq_label_class_schema_class_id"
        ),
        sa.UniqueConstraint(
            "schema_id", "semantic_key", name="uq_label_class_schema_semantic"
        ),
        sa.CheckConstraint(
            "class_id >= 0", name="ck_label_schema_classes_label_class_id_nonnegative"
        ),
        sa.ForeignKeyConstraint(
            ["schema_id"], ["label_schemas.id"],
            name="fk_label_schema_classes_schema_id_label_schemas",
            ondelete="RESTRICT",
        ),
    )
    op.create_table(
        "datasets",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("metadata_json", JSONB, nullable=False),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.Column("updated_at", UTC, server_default=NOW, nullable=False),
        sa.UniqueConstraint("name", name="uq_datasets_name"),
    )
    op.create_table(
        "model_bindings",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("external_ref", sa.String(255)),
        sa.Column("current_release_id", UUID),
        sa.Column("current_runtime_instance_id", UUID),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.CheckConstraint(
            "status IN ('unbound', 'bound', 'archived')", name="ck_model_bindings_binding_status"
        ),
    )
    op.create_table(
        "dataset_snapshots",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("dataset_id", UUID, nullable=False),
        sa.Column("label_schema_id", UUID, nullable=False),
        sa.Column("train_manifest_json", JSONB, nullable=False),
        sa.Column("val_manifest_json", JSONB, nullable=False),
        sa.Column("test_manifest_json", JSONB, nullable=False),
        sa.Column("manifest_path", sa.String(1024), nullable=False),
        sa.Column("manifest_hash", sa.String(255), nullable=False),
        sa.Column("train_positive_count", sa.Integer(), nullable=False),
        sa.Column("val_positive_count", sa.Integer(), nullable=False),
        sa.Column("test_positive_count", sa.Integer(), nullable=False),
        sa.Column("train_negative_count", sa.Integer(), nullable=False),
        sa.Column("val_negative_count", sa.Integer(), nullable=False),
        sa.Column("test_negative_count", sa.Integer(), nullable=False),
        sa.Column("quality_json", JSONB, nullable=False),
        sa.Column("source_path", sa.String(1024)),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.CheckConstraint(
            "train_positive_count >= 0 AND val_positive_count >= 0 AND test_positive_count >= 0",
            name="ck_dataset_snapshots_dataset_snapshot_positive_counts",
        ),
        sa.CheckConstraint(
            "train_negative_count >= 0 AND val_negative_count >= 0 AND test_negative_count >= 0",
            name="ck_dataset_snapshots_dataset_snapshot_negative_counts",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["datasets.id"],
            name="fk_dataset_snapshots_dataset_id_datasets",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["label_schema_id"], ["label_schemas.id"],
            name="fk_dataset_snapshots_label_schema_id_label_schemas",
            ondelete="RESTRICT",
        ),
    )
    op.create_table(
        "model_nodes",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("parent_id", UUID),
        sa.Column("label_schema_id", UUID),
        sa.Column("dataset_snapshot_id", UUID),
        sa.Column("training_attempt_id", UUID),
        sa.Column("task_type", sa.String(64), nullable=False),
        sa.Column("model_family", sa.String(128), nullable=False),
        sa.Column("artifact_path", sa.String(1024), nullable=False),
        sa.Column("artifact_hash", sa.String(255), nullable=False),
        sa.Column("framework", sa.String(64), nullable=False),
        sa.Column("artifact_format", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("metadata_json", JSONB, nullable=False),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.Column("updated_at", UTC, server_default=NOW, nullable=False),
        sa.CheckConstraint(
            "status IN ('candidate', 'approved', 'rejected', 'archived')",
            name="ck_model_nodes_model_node_status",
        ),
        sa.UniqueConstraint(
            "training_attempt_id", name="uq_model_node_training_attempt"
        ),
        sa.UniqueConstraint(
            "id", "training_attempt_id", name="uq_model_node_id_training_attempt"
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"], ["model_nodes.id"],
            name="fk_model_nodes_parent_id_model_nodes",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["label_schema_id"], ["label_schemas.id"],
            name="fk_model_nodes_label_schema_id_label_schemas",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_snapshot_id"], ["dataset_snapshots.id"],
            name="fk_model_nodes_dataset_snapshot_id_dataset_snapshots",
            ondelete="RESTRICT",
        ),
    )
    op.create_table(
        "model_binding_releases",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("binding_id", UUID, nullable=False),
        sa.Column("revision_no", sa.Integer(), nullable=False),
        sa.Column("model_node_id", UUID, nullable=False),
        sa.Column("inference_config_json", JSONB, nullable=False),
        sa.Column("inference_config_hash", sa.String(255), nullable=False),
        sa.Column("release_type", sa.String(32), nullable=False),
        sa.Column("rollback_target_release_id", UUID),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("reason", sa.Text()),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.UniqueConstraint(
            "binding_id", "id", name="uq_binding_release_binding_id_id"
        ),
        sa.UniqueConstraint(
            "binding_id", "id", "model_node_id",
            name="uq_binding_release_binding_id_model_node_id",
        ),
        sa.UniqueConstraint(
            "binding_id", "revision_no", name="uq_binding_release_revision"
        ),
        sa.CheckConstraint("revision_no > 0", name="ck_model_binding_releases_release_revision_positive"),
        sa.CheckConstraint(
            "release_type IN ('normal', 'rollback')", name="ck_model_binding_releases_release_type"
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'preparing', 'active', 'failed', 'superseded')",
            name="ck_model_binding_releases_release_status",
        ),
        sa.CheckConstraint(
            "(release_type = 'rollback' AND rollback_target_release_id IS NOT NULL) OR "
            "(release_type = 'normal' AND rollback_target_release_id IS NULL)",
            name="ck_model_binding_releases_release_rollback_target_required",
        ),
        sa.ForeignKeyConstraint(
            ["binding_id"], ["model_bindings.id"],
            name="fk_model_binding_releases_binding_id_model_bindings",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["model_node_id"], ["model_nodes.id"],
            name="fk_model_binding_releases_model_node_id_model_nodes",
            ondelete="RESTRICT",
        ),
    )
    op.create_table(
        "training_tasks",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("parent_model_node_id", UUID),
        sa.Column("dataset_snapshot_id", UUID, nullable=False),
        sa.Column("target_binding_id", UUID),
        sa.Column("task_type", sa.String(64), nullable=False),
        sa.Column("model_family", sa.String(128), nullable=False),
        sa.Column("training_config_json", JSONB, nullable=False),
        sa.Column("resource_config_json", JSONB, nullable=False),
        sa.Column("evaluation_policy_json", JSONB, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_by", sa.String(255)),
        sa.Column("failure_reason", sa.Text()),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.Column("updated_at", UTC, server_default=NOW, nullable=False),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_training_tasks_training_task_status",
        ),
        sa.ForeignKeyConstraint(
            ["parent_model_node_id"], ["model_nodes.id"],
            name="fk_training_tasks_parent_model_node_id_model_nodes",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_snapshot_id"], ["dataset_snapshots.id"],
            name="fk_training_tasks_dataset_snapshot_id_dataset_snapshots",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["target_binding_id"], ["model_bindings.id"],
            name="fk_training_tasks_target_binding_id_model_bindings",
            ondelete="RESTRICT",
        ),
    )
    op.create_table(
        "training_attempts",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("task_id", UUID, nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("gpu_device", sa.String(64)),
        sa.Column("current_epoch", sa.Integer(), nullable=False),
        sa.Column("log_path", sa.String(1024)),
        sa.Column("latest_checkpoint_id", UUID),
        sa.Column("best_checkpoint_id", UUID),
        sa.Column("lease_token", UUID),
        sa.Column("fencing_token", sa.BigInteger()),
        sa.Column("heartbeat_at", UTC),
        sa.Column("lease_expires_at", UTC),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text()),
        sa.Column("started_at", UTC),
        sa.Column("finished_at", UTC),
        sa.UniqueConstraint("task_id", "attempt_no", name="uq_training_attempt_task_no"),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'recovering', 'completed', 'failed', 'cancelled')",
            name="ck_training_attempts_training_attempt_status",
        ),
        sa.CheckConstraint(
            "attempt_no >= 0 AND current_epoch >= 0 AND retry_count >= 0",
            name="ck_training_attempts_training_attempt_counters_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["task_id"], ["training_tasks.id"],
            name="fk_training_attempts_task_id_training_tasks",
            ondelete="RESTRICT",
        ),
    )
    op.create_table(
        "checkpoints",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("attempt_id", UUID, nullable=False),
        sa.Column("parent_artifact_hash", sa.String(255), nullable=False),
        sa.Column("dataset_snapshot_id", UUID, nullable=False),
        sa.Column("training_config_hash", sa.String(255), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("artifact_path", sa.String(1024), nullable=False),
        sa.Column("artifact_hash", sa.String(255), nullable=False),
        sa.Column("metrics_json", JSONB, nullable=False),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.UniqueConstraint("attempt_id", "id", name="uq_checkpoint_attempt_id_id"),
        sa.CheckConstraint("epoch >= 0", name="ck_checkpoints_checkpoint_epoch_nonnegative"),
        sa.ForeignKeyConstraint(
            ["attempt_id"], ["training_attempts.id"],
            name="fk_checkpoints_attempt_id_training_attempts",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_snapshot_id"], ["dataset_snapshots.id"],
            name="fk_checkpoints_dataset_snapshot_id_dataset_snapshots",
            ondelete="RESTRICT",
        ),
    )
    op.create_table(
        "evaluations",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("model_node_id", UUID, nullable=False),
        sa.Column("dataset_snapshot_id", UUID, nullable=False),
        sa.Column("auto_status", sa.String(32), nullable=False),
        sa.Column("human_status", sa.String(32), nullable=False),
        sa.Column("evaluation_policy_json", JSONB, nullable=False),
        sa.Column("auto_metrics_json", JSONB, nullable=False),
        sa.Column("report_path", sa.String(1024)),
        sa.Column("report_hash", sa.String(255)),
        sa.Column("reviewer", sa.String(255)),
        sa.Column("human_conclusion", sa.String(64)),
        sa.Column("human_comments", sa.Text()),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("attempt_status", sa.String(32), nullable=False),
        sa.Column("lease_token", UUID),
        sa.Column("fencing_token", sa.BigInteger()),
        sa.Column("heartbeat_at", UTC),
        sa.Column("lease_expires_at", UTC),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text()),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.Column("reviewed_at", UTC),
        sa.CheckConstraint(
            "auto_status IN ('pending', 'running', 'passed', 'failed')",
            name="ck_evaluations_evaluation_auto_status",
        ),
        sa.CheckConstraint(
            "human_status IN ('pending', 'passed', 'failed')",
            name="ck_evaluations_evaluation_human_status",
        ),
        sa.CheckConstraint(
            "attempt_status IN ('pending', 'running', 'passed', 'failed', 'retrying', 'cancelled')",
            name="ck_evaluations_evaluation_attempt_status",
        ),
        sa.CheckConstraint(
            "attempt_no >= 0 AND retry_count >= 0",
            name="ck_evaluations_evaluation_counters_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["model_node_id"], ["model_nodes.id"],
            name="fk_evaluations_model_node_id_model_nodes",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_snapshot_id"], ["dataset_snapshots.id"],
            name="fk_evaluations_dataset_snapshot_id_dataset_snapshots",
            ondelete="RESTRICT",
        ),
    )
    op.create_table(
        "runtime_instances",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("binding_id", UUID, nullable=False),
        sa.Column("release_id", UUID, nullable=False),
        sa.Column("model_node_id", UUID, nullable=False),
        sa.Column("config_hash", sa.String(255), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("gpu_device", sa.String(64), nullable=False),
        sa.Column("reserved_memory_mb", sa.Integer(), nullable=False),
        sa.Column("actual_memory_mb", sa.Integer()),
        sa.Column("worker_id", sa.String(255)),
        sa.Column("heartbeat_at", UTC),
        sa.Column("failure_reason", sa.String(2048)),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.Column("stopped_at", UTC),
        sa.UniqueConstraint("binding_id", "id", name="uq_runtime_binding_id_id"),
        sa.CheckConstraint(
            "status IN ('loading', 'preparing', 'ready', 'serving', 'draining', 'stopped', 'failed')",
            name="ck_runtime_instances_runtime_status",
        ),
        sa.CheckConstraint(
            "generation >= 0 AND reserved_memory_mb >= 0 "
            "AND (actual_memory_mb IS NULL OR actual_memory_mb >= 0)",
            name="ck_runtime_instances_runtime_memory_generation_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["binding_id"], ["model_bindings.id"],
            name="fk_runtime_instances_binding_id_model_bindings",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["model_node_id"], ["model_nodes.id"],
            name="fk_runtime_instances_model_node_id_model_nodes",
            ondelete="RESTRICT",
        ),
    )
    op.create_table(
        "gpu_resources",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("gpu_device", sa.String(64), nullable=False),
        sa.Column("capacity_memory_mb", sa.Integer(), nullable=False),
        sa.Column("reserved_memory_mb", sa.Integer(), nullable=False),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.Column("updated_at", UTC, server_default=NOW, nullable=False),
        sa.UniqueConstraint("gpu_device", name="uq_gpu_resources_gpu_device"),
        sa.CheckConstraint(
            "capacity_memory_mb >= 0 AND reserved_memory_mb >= 0 "
            "AND reserved_memory_mb <= capacity_memory_mb",
            name="ck_gpu_resources_gpu_resource_memory_nonnegative",
        ),
    )
    op.create_table(
        "gpu_resource_leases",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("gpu_resource_id", UUID, nullable=False),
        sa.Column("gpu_device", sa.String(64), nullable=False),
        sa.Column("owner_type", sa.String(64), nullable=False),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("reserved_memory_mb", sa.Integer(), nullable=False),
        sa.Column("actual_memory_mb", sa.Integer()),
        sa.Column("lease_token", UUID, nullable=False),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False),
        sa.Column("heartbeat_at", UTC),
        sa.Column("lease_expires_at", UTC, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
        sa.CheckConstraint(
            "status IN ('active', 'released', 'expired', 'failed')",
            name="ck_gpu_resource_leases_gpu_lease_status",
        ),
        sa.CheckConstraint(
            "reserved_memory_mb >= 0 AND (actual_memory_mb IS NULL OR actual_memory_mb >= 0)",
            name="ck_gpu_resource_leases_gpu_lease_memory_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["gpu_resource_id"], ["gpu_resources.id"],
            name="fk_gpu_resource_leases_gpu_resource_id_gpu_resources",
            ondelete="RESTRICT",
        ),
    )
    op.create_table(
        "model_residency_plans",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("binding_id", UUID, nullable=False),
        sa.Column("release_id", UUID, nullable=False),
        sa.Column("model_node_id", UUID, nullable=False),
        sa.Column("gpu_device", sa.String(64), nullable=False),
        sa.Column("reserved_memory_mb", sa.Integer(), nullable=False),
        sa.Column("desired_state", sa.String(32), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("created_by", sa.String(255)),
        sa.Column("updated_at", UTC, server_default=NOW, nullable=False),
        sa.CheckConstraint(
            "desired_state IN ('resident', 'removed')",
            name="ck_model_residency_plans_residency_desired_state",
        ),
        sa.CheckConstraint(
            "reserved_memory_mb >= 0",
            name="ck_model_residency_plans_residency_reserved_memory_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["binding_id"], ["model_bindings.id"],
            name="fk_model_residency_plans_binding_id_model_bindings",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["model_node_id"], ["model_nodes.id"],
            name="fk_model_residency_plans_model_node_id_model_nodes",
            ondelete="RESTRICT",
        ),
    )
    op.create_table(
        "operation_logs",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("operation_type", sa.String(128), nullable=False),
        sa.Column("actor", sa.String(255)),
        sa.Column("request_id", UUID),
        sa.Column("resource_type", sa.String(64)),
        sa.Column("resource_id", UUID),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("summary_json", JSONB, nullable=False),
        sa.Column("error_summary", sa.Text()),
        sa.Column("created_at", UTC, server_default=NOW, nullable=False),
    )

    op.create_foreign_key(
        "fk_model_nodes_training_attempt_id_training_attempts",
        "model_nodes",
        "training_attempts",
        ["training_attempt_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_training_attempt_latest_checkpoint_same_attempt",
        "training_attempts",
        "checkpoints",
        ["id", "latest_checkpoint_id"],
        ["attempt_id", "id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_training_attempt_best_checkpoint_same_attempt",
        "training_attempts",
        "checkpoints",
        ["id", "best_checkpoint_id"],
        ["attempt_id", "id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_release_rollback_target",
        "model_binding_releases",
        "model_binding_releases",
        ["binding_id", "rollback_target_release_id"],
        ["binding_id", "id"],
    )
    op.create_foreign_key(
        "fk_runtime_binding_release",
        "runtime_instances",
        "model_binding_releases",
        ["binding_id", "release_id", "model_node_id"],
        ["binding_id", "id", "model_node_id"],
    )
    op.create_foreign_key(
        "fk_residency_binding_release",
        "model_residency_plans",
        "model_binding_releases",
        ["binding_id", "release_id", "model_node_id"],
        ["binding_id", "id", "model_node_id"],
    )
    op.create_foreign_key(
        "fk_binding_current_release",
        "model_bindings",
        "model_binding_releases",
        ["id", "current_release_id"],
        ["binding_id", "id"],
    )
    op.create_foreign_key(
        "fk_binding_current_runtime",
        "model_bindings",
        "runtime_instances",
        ["id", "current_runtime_instance_id"],
        ["binding_id", "id"],
    )

    op.create_index(
        "uq_training_attempt_active_task",
        "training_attempts",
        ["task_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('running', 'recovering')"),
    )
    op.create_index(
        "uq_runtime_serving_binding",
        "runtime_instances",
        ["binding_id"],
        unique=True,
        postgresql_where=sa.text("status = 'serving'"),
    )
    op.create_index(
        "uq_runtime_candidate_binding",
        "runtime_instances",
        ["binding_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('loading', 'preparing', 'ready')"),
    )
    op.create_index(
        "uq_gpu_lease_owner_active",
        "gpu_resource_leases",
        ["owner_type", "owner_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.drop_index("uq_gpu_lease_owner_active", table_name="gpu_resource_leases")
    op.drop_index("uq_runtime_candidate_binding", table_name="runtime_instances")
    op.drop_index("uq_runtime_serving_binding", table_name="runtime_instances")
    op.drop_index("uq_training_attempt_active_task", table_name="training_attempts")

    op.drop_constraint("fk_binding_current_runtime", "model_bindings", type_="foreignkey")
    op.drop_constraint("fk_binding_current_release", "model_bindings", type_="foreignkey")
    op.drop_constraint("fk_residency_binding_release", "model_residency_plans", type_="foreignkey")
    op.drop_constraint("fk_runtime_binding_release", "runtime_instances", type_="foreignkey")
    op.drop_constraint("fk_release_rollback_target", "model_binding_releases", type_="foreignkey")
    op.drop_constraint("fk_training_attempt_best_checkpoint_same_attempt", "training_attempts", type_="foreignkey")
    op.drop_constraint("fk_training_attempt_latest_checkpoint_same_attempt", "training_attempts", type_="foreignkey")
    op.drop_constraint("fk_model_nodes_training_attempt_id_training_attempts", "model_nodes", type_="foreignkey")

    for table in (
        "operation_logs",
        "model_residency_plans",
        "gpu_resource_leases",
        "gpu_resources",
        "runtime_instances",
        "evaluations",
        "checkpoints",
        "training_attempts",
        "training_tasks",
        "model_binding_releases",
        "model_nodes",
        "dataset_snapshots",
        "model_bindings",
        "datasets",
        "label_schema_classes",
        "label_schemas",
    ):
        op.drop_table(table)
