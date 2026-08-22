from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.app.training.checkpoint_manager import CheckpointManager
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
    1. Load task config from DB (caller must have stored it)
    2. Prepare YOLO-format dataset from snapshot manifest
    3. Run ultralytics YOLO.train()
    4. Save checkpoints (best + latest)
    5. Return result summary
    """
    logger.info("Starting training for task %s", task_id)
    return f"training task {task_id} dispatched"


def execute_training(
    *,
    training_config: dict[str, Any],
    dataset_manifest: dict[str, Any],
    output_dir: Path,
    parent_model_path: str | None = None,
) -> TrainResult:
    """Run YOLO training synchronously. Callable from worker or tests."""
    dataset_dir = output_dir / "dataset"
    checkpoint_dir = output_dir / "checkpoints"

    YoloDataset.from_manifest(dataset_manifest, dataset_dir)

    trainer = YoloTrainer(training_config)
    result = trainer.train(dataset_dir, checkpoint_dir, parent_model_path=parent_model_path)
    return result


@celery_app.task(name="platform.training.recover_attempt")
def recover_attempt(attempt_id: str) -> str:
    """Placeholder for attempt recovery after worker crash."""
    return f"recovery for attempt {attempt_id} not implemented"
