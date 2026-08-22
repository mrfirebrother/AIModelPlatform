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
    logger.info("Starting training for task %s", task_id)
    return f"training task {task_id} dispatched"


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
    4. Updates attempt status
    """
    scheduler = ResourceScheduler(session=session, cpu_mode=True)
    lease = None

    try:
        lease = scheduler.reserve_gpu(
            attempt_id=attempt.id,
            required_memory_mb=required_memory_mb,
            gpu_device=attempt.gpu_device,
        )

        attempt.status = "running"
        attempt.lease_token = lease.lease_token
        attempt.fencing_token = lease.fencing_token
        session.flush()

        dataset_dir = output_dir / "dataset"
        checkpoint_dir = output_dir / "checkpoints"

        YoloDataset.from_manifest(dataset_manifest, dataset_dir)

        trainer = YoloTrainer(training_config)
        result = trainer.train(
            dataset_dir, checkpoint_dir, parent_model_path=parent_model_path
        )

        if result.success:
            attempt.status = "completed"
        else:
            attempt.status = "failed"
            attempt.last_error = result.error

        return result

    except InsufficientGPUError as exc:
        attempt.status = "failed"
        attempt.last_error = f"GPU resource error: {exc}"
        session.flush()
        return TrainResult(
            success=False,
            epochs_completed=0,
            best_model_path=None,
            latest_model_path=None,
            error=str(exc),
        )

    except Exception as exc:
        attempt.status = "failed"
        attempt.last_error = str(exc)
        session.flush()
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
