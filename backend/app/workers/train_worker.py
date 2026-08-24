from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.models import TrainingAttempt, TrainingTask
from backend.app.training.checkpoint_manager import CheckpointManager
from backend.app.training.resource_scheduler import (
    InsufficientGPUError,
    ResourceScheduler,
)
from backend.app.training.yolo_dataset import YoloDataset
from backend.app.training.yolo_trainer import YoloTrainer, TrainResult

from backend.app.workers.attempt_lease import heartbeat as attempt_heartbeat

from .celery_app import celery_app

logger = logging.getLogger(__name__)


def _resolve_dataset_snapshot(
    train_manifest_json: list[dict[str, Any]],
    val_manifest_json: list[dict[str, Any]],
    test_manifest_json: list[dict[str, Any]],
    nc: int,
    names: list[str],
) -> dict[str, Any]:
    return {
        "train_files": train_manifest_json,
        "val_files": val_manifest_json,
        "test_files": test_manifest_json,
        "nc": nc,
        "names": names,
    }


@celery_app.task(name="platform.training.run_training")
def run_training(task_id: str) -> str:
    """Execute a real YOLO training run via ultralytics.

    Expected flow:
    1. Load task config from DB
    2. Create training attempt
    3. Reserve GPU via ResourceScheduler
    4. Prepare YOLO-format dataset from snapshot manifest
    5. Run ultralytics YOLO.train()
    6. Save checkpoints (best + latest)
    7. Release GPU lease
    8. Return result summary
    """
    from backend.app.workers.db_session import worker_session
    from backend.app.services.training_service import start_attempt, complete_attempt, fail_attempt

    logger.info("Starting training for task %s", task_id)

    try:
        task_uuid = UUID(task_id)
    except ValueError:
        logger.error("Invalid task ID: %s", task_id)
        return f"invalid task id: {task_id}"

    with worker_session() as session:
        task = session.get(TrainingTask, task_uuid)
        if task is None:
            logger.error("Task %s not found", task_id)
            return f"task {task_id} not found"

        if task.status in ("completed", "cancelled"):
            logger.info("Task %s already in terminal state: %s", task_id, task.status)
            return f"task {task_id} already {task.status}"

        if task.status == "failed":
            logger.info("Task %s already failed", task_id)
            return f"task {task_id} already failed"

        if task.cancellation_requested:
            logger.info("Task %s has cancellation requested, aborting", task_id)
            task.status = "cancelled"
            session.commit()
            return f"task {task_id} cancelled"

        if task.status == "running":
            active_attempt = None
            for attempt in task.attempts:
                if attempt.status in ("running", "recovering"):
                    active_attempt = attempt
                    break
            if active_attempt is not None:
                logger.info("Task %s already running with attempt %s", task_id, active_attempt.attempt_no)
                return f"task {task_id} already running"

        try:
            attempt = start_attempt(session, task_uuid)
            session.commit()
        except ValueError as exc:
            task.status = "failed"
            task.failure_reason = str(exc)
            session.commit()
            logger.error("Failed to start attempt for task %s: %s", task_id, exc)
            return f"failed to start attempt: {exc}"

        output_dir = Path(f"/data/checkpoints/{task_id}/attempt_{attempt.attempt_no}")
        output_dir.mkdir(parents=True, exist_ok=True)

        training_config = task.training_config_json or {}
        parent_model_path = None
        if task.parent_model_node_id is not None:
            parent_model = session.get(type(task).parent_model_node.property.mapper.class_, task.parent_model_node_id)
            if parent_model is not None:
                parent_model_path = parent_model.artifact_path

        required_memory_mb = 4096
        resource_config = task.resource_config_json or {}
        if "memory_mb" in resource_config:
            required_memory_mb = resource_config["memory_mb"]
        elif "gpu_memory_mb" in resource_config:
            required_memory_mb = resource_config["gpu_memory_mb"]

        dataset_snapshot = task.dataset_snapshot
        if dataset_snapshot is None:
            fail_attempt(session, attempt.id, error="Dataset snapshot not found")
            session.commit()
            return f"dataset snapshot not found for task {task_id}"

        label_schema = dataset_snapshot.label_schema
        nc = len(label_schema.classes) if label_schema else 0
        names = [c.semantic_key for c in label_schema.classes] if label_schema else []

        dataset_manifest = {
            "train_files": dataset_snapshot.train_manifest_json or [],
            "val_files": dataset_snapshot.val_manifest_json or [],
            "test_files": dataset_snapshot.test_manifest_json or [],
            "nc": nc,
            "names": names,
        }

        try:
            result = execute_training(
                session=session,
                task=task,
                attempt=attempt,
                training_config=training_config,
                dataset_manifest=dataset_manifest,
                output_dir=output_dir,
                parent_model_path=parent_model_path,
                required_memory_mb=required_memory_mb,
            )

            session.refresh(attempt)

            if result.success:
                if result.best_model_path and Path(result.best_model_path).exists():
                    import hashlib
                    with open(result.best_model_path, "rb") as f:
                        artifact_hash = f"sha256:{hashlib.sha256(f.read()).hexdigest()}"

                    checkpoint_info = {
                        "epoch": result.epochs_completed,
                        "is_best": True,
                        "metrics": result.metrics,
                    }

                    model = complete_attempt(
                        session,
                        attempt.id,
                        artifact_path=result.best_model_path,
                        artifact_hash=artifact_hash,
                        metadata_json={"metrics": result.metrics, "epochs": result.epochs_completed},
                        checkpoint_info=checkpoint_info,
                    )
                    session.commit()
                    logger.info("Task %s completed, candidate model %s created", task_id, model.id)
                    return f"task {task_id} completed, candidate {model.id}"
                else:
                    fail_attempt(session, attempt.id, error="No best model artifact produced")
                    session.commit()
                    return f"task {task_id} failed: no best model artifact"
            else:
                fail_attempt(session, attempt.id, error=result.error or "Training failed")
                session.commit()
                logger.error("Task %s failed: %s", task_id, result.error)
                return f"task {task_id} failed: {result.error}"

        except Exception as exc:
            try:
                session.refresh(attempt)
                fail_attempt(session, attempt.id, error=str(exc))
                session.commit()
            except Exception:
                session.rollback()
            logger.exception("Task %s failed with exception", task_id)
            return f"task {task_id} failed: {exc}"


def execute_training(
    *,
    session: Session,
    task: TrainingTask,
    attempt: TrainingAttempt,
    training_config: dict[str, Any],
    dataset_manifest: dict[str, Any],
    output_dir: Path,
    parent_model_path: str | None = None,
    required_memory_mb: int = 4096,
) -> TrainResult:
    """Run YOLO training synchronously with GPU lease management.

    This function:
    1. Reserves GPU memory
    2. Runs training
    3. Releases GPU memory

    Does NOT mutate attempt.status -- run_training uses
    complete_attempt/fail_attempt for that.
    """
    device = training_config.get("device", "cpu")
    cpu_mode = device == "cpu"
    scheduler = ResourceScheduler(session=session, cpu_mode=cpu_mode)
    lease = None

    try:
        if device != "cpu":
            lease = scheduler.reserve_gpu(
                attempt_id=attempt.id,
                required_memory_mb=required_memory_mb,
                gpu_device=attempt.gpu_device,
            )
            attempt.lease_token = lease.lease_token
            attempt.fencing_token = lease.fencing_token
            session.flush()

        dataset_dir = output_dir / "dataset"
        checkpoint_dir = output_dir / "checkpoints"

        YoloDataset.from_manifest(dataset_manifest, dataset_dir)

        try:
            attempt_heartbeat(session, attempt.id, attempt.lease_token)
            session.flush()
        except Exception:
            logger.warning("Heartbeat failed for attempt %s", attempt.id)

        trainer = YoloTrainer(training_config)
        result = trainer.train(
            dataset_dir, checkpoint_dir, parent_model_path=parent_model_path
        )

        return result

    except InsufficientGPUError as exc:
        return TrainResult(
            success=False,
            epochs_completed=0,
            best_model_path=None,
            latest_model_path=None,
            error=str(exc),
        )

    except Exception as exc:
        return TrainResult(
            success=False,
            epochs_completed=0,
            best_model_path=None,
            latest_model_path=None,
            error=str(exc),
        )

    finally:
        if lease is not None:
            scheduler.release_gpu(attempt.id)


@celery_app.task(name="platform.training.recover_attempt")
def recover_attempt(attempt_id: str) -> str:
    """Placeholder for attempt recovery after worker crash."""
    return f"recovery for attempt {attempt_id} not implemented"
