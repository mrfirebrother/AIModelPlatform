from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.models import TrainingAttempt, TrainingTask
from backend.app.training.resource_scheduler import (
    InsufficientGPUError,
    ResourceScheduler,
)
from backend.app.training.training_params import defaults_from_settings, resolve_config
from backend.app.training.yolo_dataset import DatasetPreparationCancelled, YoloDataset
from backend.app.training.yolo_trainer import YoloTrainer, TrainResult
from backend.app.storage.artifacts import compute_file_hash
from backend.app.services.resource_lease import heartbeat_lease

from backend.app.workers.attempt_lease import heartbeat as attempt_heartbeat

from .celery_app import celery_app

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECONDS = 60


def _manifest_source_dir(manifest_path: Path) -> str:
    """Return the ``source_dir`` recorded in a snapshot manifest.

    Snapshot manifests store per-split entries as *relative* paths and keep the
    directory they are rooted at in a top-level ``source_dir`` field.
    """
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001 - a broken manifest must not abort the merge
        logger.warning("Could not read source_dir from manifest %s", manifest_path)
        return ""
    value = data.get("source_dir")
    return str(value) if value else ""


def _with_absolute_paths(entries: list[dict[str, Any]], source_dir: str) -> list[dict[str, Any]]:
    """Resolve manifest entries against *source_dir*.

    ``YoloDataset.from_manifest`` copies from ``*_stored_path`` when present and
    only falls back to ``source_dir`` + relative path otherwise. Populating the
    stored paths here keeps multi-snapshot merges correct, since entries from
    different snapshots are rooted at different directories.
    """
    if not source_dir:
        return entries
    base = Path(source_dir)
    resolved: list[dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        for rel_key, stored_key in (("image", "image_stored_path"), ("label", "label_stored_path")):
            relative = item.get(rel_key)
            if not item.get(stored_key) and relative and not Path(relative).is_absolute():
                item[stored_key] = str(base / relative)
        resolved.append(item)
    return resolved


def _resolve_training_device(training_config: dict[str, Any]) -> str:
    """Pick the training device: explicit task config wins, else ``GPU_DEVICE``.

    Tasks created from the UI carry no ``device`` field, so without this fallback
    they would always train on CPU even when ``GPU_DEVICE`` points at a usable GPU.
    ``YoloDataset.resolve_device`` degrades to CPU when CUDA is unavailable.
    """
    requested = training_config.get("device") or get_settings().gpu_device
    return YoloDataset.resolve_device(str(requested))


def _build_trainer_config(training_config: dict[str, Any], device: str) -> dict[str, Any]:
    """Resolve the ultralytics arguments for one training run.

    The create route already stores a complete, normalized parameter snapshot in
    ``training_config_json`` (which is immutable after insert), so this normally just adds the
    resolved ``device``. Tasks created before a default existed - or hand-written API payloads -
    still get ``TRAINING_DEFAULT_*`` filled in here, which is why the resolution lives in
    ``training_params.resolve_config`` and is shared with the route instead of duplicated.
    """
    settings = get_settings()
    try:
        resolved, warnings = resolve_config(
            training_config, defaults=defaults_from_settings(settings)
        )
    except ValueError as exc:  # e.g. a task created before a validation rule existed
        logger.warning("Training config rejected by validation (%s); using it as-is", exc)
        resolved = {**defaults_from_settings(settings), **training_config}
        warnings = []
    for warning in warnings:
        logger.warning("Training config: %s", warning)
    resolved["device"] = device
    return resolved


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
    from backend.app.services.training_service import start_attempt, complete_attempt, cancel_task, fail_attempt

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
            cancel_task(session, task.id)
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

        settings = get_settings()
        output_dir = Path(settings.checkpoint_dir) / task_id / f"attempt_{attempt.attempt_no}"
        output_dir.mkdir(parents=True, exist_ok=True)

        training_config = task.training_config_json or {}

        if _resolve_training_device(training_config) != "cpu":
            try:
                import torch
                if not torch.cuda.is_available():
                    fail_attempt(session, attempt.id, error="CUDA requested but not available")
                    session.commit()
                    return f"task {task_id} failed: CUDA requested but not available"
            except ImportError:
                fail_attempt(session, attempt.id, error="CUDA requested but torch not installed")
                session.commit()
                return f"task {task_id} failed: CUDA requested but torch not installed"
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

        # Support multiple snapshots: merge manifests from all snapshot IDs in config
        extra_snapshot_ids = training_config.get("dataset_snapshot_ids", [])
        all_snapshots = [dataset_snapshot]
        if extra_snapshot_ids and len(extra_snapshot_ids) > 1:
            for sid in extra_snapshot_ids:
                snap = session.get(DatasetSnapshot, sid)
                if snap is not None and snap.id != dataset_snapshot.id:
                    all_snapshots.append(snap)

        merged_train = []
        merged_val = []
        merged_test = []
        nc = 0
        names = []
        source_dirs = []

        for snap in all_snapshots:
            if snap.manifest_hash is None:
                continue
            manifest_path = Path(snap.manifest_path)
            if not manifest_path.exists():
                continue

            label_schema = snap.label_schema
            if label_schema:
                nc = max(nc, len(label_schema.classes))
                for c in label_schema.classes:
                    if c.semantic_key not in names:
                        names.append(c.semantic_key)

            snap_source_dir = _manifest_source_dir(manifest_path)
            snap_train = _with_absolute_paths(snap.train_manifest_json or [], snap_source_dir)
            snap_val = _with_absolute_paths(snap.val_manifest_json or [], snap_source_dir)
            snap_test = _with_absolute_paths(snap.test_manifest_json or [], snap_source_dir)
            merged_train.extend(snap_train)
            merged_val.extend(snap_val)
            merged_test.extend(snap_test)

            if snap.source_path:
                source_dirs.append(snap.source_path)

        if not merged_train and not merged_val and not merged_test:
            fail_attempt(session, attempt.id, error="No valid dataset snapshots found")
            session.commit()
            return f"no valid dataset snapshots for task {task_id}"

        dataset_manifest = {
            "train_files": merged_train,
            "val_files": merged_val,
            "test_files": merged_test,
            "nc": nc,
            "names": names,
        }

        resolved_device = _resolve_training_device(training_config)
        attempt.gpu_device = None if resolved_device == "cpu" else resolved_device
        session.flush()

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
            session.refresh(task)

            if task.cancellation_requested:
                logger.info("Task %s was cancelled during training, cancelling task", task_id)
                cancel_task(session, task.id)
                if attempt.status in ("running", "recovering"):
                    attempt.status = "cancelled"
                    attempt.finished_at = datetime.now(timezone.utc)
                task.status = "cancelled"
                session.commit()
                return f"task {task_id} cancelled during training"

            if result.success:
                if result.best_model_path and Path(result.best_model_path).exists():
                    with open(result.best_model_path, "rb") as f:
                        artifact_hash = f"sha256:{hashlib.sha256(f.read()).hexdigest()}"

                    checkpoint_info = {
                        "epoch": result.epochs_completed,
                        "is_best": True,
                        "metrics": result.metrics,
                    }

                    label_schema_id = (
                        dataset_snapshot.label_schema_id
                        if dataset_snapshot is not None
                        else None
                    )

                    model = complete_attempt(
                        session,
                        attempt.id,
                        artifact_path=result.best_model_path,
                        artifact_hash=artifact_hash,
                        metadata_json={"metrics": result.metrics, "epochs": result.epochs_completed},
                        checkpoint_info=checkpoint_info,
                        label_schema_id=label_schema_id,
                    )
                    # auto-create pending evaluation so it appears in 评估收件箱
                    try:
                        from backend.app.evaluation.policy import EvaluationPolicy
                        from backend.app.services.evaluation_service import create_evaluation
                        # An all-zero policy lets every model "pass" (0.0 >= 0.0), which makes the
                        # gate meaningless. Default to the configured threshold and let the task
                        # payload override any individual field.
                        default_policy = EvaluationPolicy(
                            min_mAP50=settings.training_default_min_map50,
                            min_precision=0.0,
                            min_recall=0.0,
                            max_regression_ratio=1.0,
                            require_per_class_coverage=False,
                        ).to_dict()
                        policy = EvaluationPolicy.from_dict(
                            {**default_policy, **(task.evaluation_policy_json or {})}
                        )
                        create_evaluation(session, model_node_id=model.id, dataset_snapshot_id=task.dataset_snapshot_id, policy=policy)
                    except Exception as eval_exc:
                        logger.warning("Failed to auto-create evaluation for model %s: %s", model.id, eval_exc)
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
    heartbeat_interval: int = HEARTBEAT_INTERVAL_SECONDS,
) -> TrainResult:
    """Run YOLO training synchronously with GPU lease management.

    This function:
    1. Reserves GPU memory
    2. Runs training
    3. Releases GPU memory

    Does NOT mutate attempt.status -- run_training uses
    complete_attempt/fail_attempt for that.
    """
    device = _resolve_training_device(training_config)
    cpu_mode = device == "cpu"
    scheduler = ResourceScheduler(session=session, cpu_mode=cpu_mode)
    lease = None
    heartbeat_thread: _HeartbeatThread | None = None
    cancel_watcher: _CancelWatcher | None = None
    trainer: YoloTrainer | None = None

    try:
        if device != "cpu":
            lease = scheduler.reserve_gpu(
                attempt_id=attempt.id,
                required_memory_mb=required_memory_mb,
                gpu_device=attempt.gpu_device,
            )
            session.flush()

        def read_cancel_requested() -> bool:
            cancel_marker = (
                Path(get_settings().checkpoint_dir)
                / str(task.id)
                / "cancel.requested"
            )
            if cancel_marker.exists():
                return True
            try:
                with worker_session() as s:
                    t = s.get(TrainingTask, task_id)
                    return bool(t.cancellation_requested) if t else False
            except Exception:
                return False

        cancel_watcher = _CancelWatcher(read_cancel_requested)
        cancel_watcher.start()

        cancel_marker = (
            Path(get_settings().checkpoint_dir)
            / str(task.id)
            / "cancel.requested"
        )

        def is_cancel_requested() -> bool:
            return cancel_watcher.cancelled or cancel_marker.exists()

        if is_cancel_requested():
            return TrainResult(success=False, epochs_completed=0, best_model_path=None, latest_model_path=None, error="Cancelled before start")

        dataset_dir = output_dir / "dataset"
        checkpoint_dir = output_dir / "checkpoints"

        try:
            YoloDataset.from_manifest(
                dataset_manifest, dataset_dir, should_cancel=is_cancel_requested
            )
        except DatasetPreparationCancelled:
            return TrainResult(
                success=False,
                epochs_completed=0,
                best_model_path=None,
                latest_model_path=None,
                error="Training cancelled",
            )

        if is_cancel_requested():
            return TrainResult(success=False, epochs_completed=0, best_model_path=None, latest_model_path=None, error="Cancelled during dataset preparation")

        # Parameter snapshot: persist the resolved parameters onto the task so the UI can show
        # what this run actually used. This gets its OWN commit: coupling it to the heartbeat
        # below lost the snapshot silently, because that call's `with_for_update` can raise and
        # its `rollback()` discarded the assignment (observed - the log said "resolved training
        # parameters" and the row still held the pre-resolution config).
        attempt_id_value = attempt.id
        lease_token_value = attempt.lease_token
        gpu_lease_id_value = lease.id if lease is not None else None
        gpu_lease_token_value = lease.lease_token if lease is not None else None

        trainer_config = _build_trainer_config(training_config, device)
        # Log, do not persist: `training_config_json` is immutable after insert
        # (install_immutable_guard(TrainingTask, …)), so the parameter snapshot is written by
        # the create route. Attempting it here raises ImmutableFieldError - verified.
        logger.info(
            "resolved training parameters for task %s: %s",
            task.id,
            {k: trainer_config.get(k) for k in (
                "epochs", "imgsz", "batch", "patience", "cache", "augmentation", "device", "recipe",
            ) if k in trainer_config},
        )

        # This transaction MUST be committed before the (minutes-to-hours) training call.
        # attempt_heartbeat() takes the attempt row with SELECT ... FOR UPDATE, so leaving
        # the transaction open for the whole run blocks every other writer on the
        # task/attempt rows: the API's cancel endpoint waits forever (cancellation never
        # reaches the database, so the run cannot be stopped from the UI), the heartbeat
        # thread blocks on its own heartbeat, and the scheduler's lease reconciliation is
        # frozen. Verified against pg_stat_activity: the worker backend sat "idle in
        # transaction" with the heartbeat UPDATE as its last statement until training ended.
        try:
            attempt_heartbeat(session, attempt_id_value, lease_token_value)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning("Heartbeat failed for attempt %s: %s", attempt_id_value, exc)

        heartbeat_thread = _HeartbeatThread(
            session=session,
            attempt_id=attempt_id_value,
            lease_token=lease_token_value,
            interval=heartbeat_interval,
            cpu_mode=cpu_mode,
            gpu_lease_id=gpu_lease_id_value,
            gpu_lease_token=gpu_lease_token_value,
        )
        heartbeat_thread.start()

        trainer = YoloTrainer(trainer_config, check_cancel=is_cancel_requested)
        result = trainer.train(
            dataset_dir, checkpoint_dir, parent_model_path=parent_model_path
        )

        # Check for cancellation after training
        session.refresh(task)
        if task.cancellation_requested:
            trainer.stop()
            cancel_task(session, task_id)
            session.commit()
            return TrainResult(success=False, epochs_completed=0, best_model_path=None, latest_model_path=None, error="Training cancelled")

        return result

    except InsufficientGPUError as exc:
        return TrainResult(
            success=False,
            epochs_completed=0,
            best_model_path=None,
            latest_model_path=None,
            error=str(exc),
        )

    finally:
        if trainer is not None:
            trainer.stop()  # Stop subprocess if still running
        if heartbeat_thread is not None:
            heartbeat_thread.stop()
            heartbeat_thread.join(timeout=5)
        if cancel_watcher is not None:
            cancel_watcher.stop()
            cancel_watcher.join(timeout=5)
        if lease is not None:
            scheduler.release_gpu(attempt.id)


class _HeartbeatThread(threading.Thread):
    """Daemon thread that sends periodic heartbeats for a training attempt.

    Uses its own independent DB session so it does not share state with the
    main training session, avoiding race conditions and detached-instance errors.
    """

    def __init__(
        self,
        session: Session,
        attempt_id: UUID,
        lease_token: UUID | None,
        interval: int = HEARTBEAT_INTERVAL_SECONDS,
        cpu_mode: bool = False,
        gpu_lease_id: UUID | None = None,
        gpu_lease_token: UUID | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self._session = session
        self._attempt_id = attempt_id
        self._lease_token = lease_token
        self._interval = interval
        self._stop_event = threading.Event()
        self._cpu_mode = cpu_mode
        self._gpu_lease_id = gpu_lease_id
        self._gpu_lease_token = gpu_lease_token

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        from backend.app.workers.db_session import worker_session

        while not self._stop_event.is_set():
            self._stop_event.wait(self._interval)
            if self._stop_event.is_set():
                break
            try:
                with worker_session() as hb_session:
                    attempt_heartbeat(hb_session, self._attempt_id, self._lease_token)
                    if self._gpu_lease_id is not None and self._gpu_lease_token is not None:
                        heartbeat_lease(
                            hb_session,
                            self._gpu_lease_id,
                            self._gpu_lease_token,
                        )
            except Exception:
                logger.warning(
                    "Heartbeat failed for attempt %s", self._attempt_id
                )


class _CancelWatcher(threading.Thread):
    """Poll cancellation independently so long file copies remain interruptible."""

    def __init__(self, read_cancel, interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self._read_cancel = read_cancel
        self._interval = interval
        self._stop_event = threading.Event()
        self.cancelled = False

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._read_cancel():
                    self.cancelled = True
                    return
            except Exception:
                logger.warning("Cancellation poll failed")
            self._stop_event.wait(self._interval)


@celery_app.task(name="platform.training.recover_attempt")
def recover_attempt(attempt_id: str) -> str:
    """Placeholder for attempt recovery after worker crash."""
    return f"recovery for attempt {attempt_id} not implemented"
