from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


MIGRATION = Path(__file__).parents[2] / "alembic" / "versions" / "0001_initial_schema.py"


def test_initial_migration_contains_explicit_ddl() -> None:
    source = MIGRATION.read_text(encoding="utf-8-sig")
    assert "op.create_table" in source
    assert "op.create_index" in source
    assert "op.create_foreign_key" in source
    assert "Base.metadata.create_all" not in source
    assert "Base.metadata.drop_all" not in source


def test_initial_migration_upgrade_and_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url or not database_url.startswith("postgresql"):
        pytest.skip("requires PostgreSQL TEST_DATABASE_URL")

    config = Config()
    config.set_main_option("script_location", str(MIGRATION.parents[1]))
    config.set_main_option("prepend_sys_path", str(MIGRATION.parents[2]))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    engine = create_engine(database_url)
    monkeypatch.setenv("POSTGRES_URL", database_url)
    try:
        command.upgrade(config, "head")
        inspector = inspect(engine)
        expected_tables = {
            "label_schemas",
            "label_schema_classes",
            "datasets",
            "dataset_snapshots",
            "model_nodes",
            "model_bindings",
            "model_binding_releases",
            "training_tasks",
            "training_attempts",
            "checkpoints",
            "evaluations",
            "runtime_instances",
            "gpu_resources",
            "gpu_resource_leases",
            "model_residency_plans",
            "operation_logs",
        }
        assert expected_tables <= set(inspector.get_table_names())
        assert {"id", "training_attempt_id", "label_schema_id"} <= {
            column["name"] for column in inspector.get_columns("model_nodes")
        }
        assert {"uq_training_attempt_active_task", "uq_runtime_serving_binding"} <= {
            index["name"] for index in inspector.get_indexes("training_attempts")
        } | {index["name"] for index in inspector.get_indexes("runtime_instances")}
        assert any(
            set(foreign_key["constrained_columns"])
            == {"binding_id", "release_id", "model_node_id"}
            for foreign_key in inspector.get_foreign_keys("runtime_instances")
        )
        suite = MIGRATION.parents[2] / "tests" / "integration" / "test_schema_constraints.py"
        environment = os.environ.copy()
        environment["TEST_DATABASE_URL"] = database_url
        environment["POSTGRES_URL"] = database_url
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-c",
                "NUL",
                "-p",
                "no:cacheprovider",
                str(suite),
                "-q",
            ],
            cwd=str(MIGRATION.parents[2].parent),
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        command.downgrade(config, "base")
        assert not inspect(engine).get_table_names()
        command.upgrade(config, "head")
        inspector = inspect(engine)
        assert {"model_nodes", "training_attempts"} <= set(
            inspector.get_table_names()
        )
        command.downgrade(config, "0000_empty")
        assert not inspect(engine).get_table_names()
    finally:
        engine.dispose()
