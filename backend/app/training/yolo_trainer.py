from __future__ import annotations

import logging
import multiprocessing
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .yolo_dataset import YoloDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    success: bool
    epochs_completed: int
    best_model_path: str | None
    latest_model_path: str | None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _train_worker(
    config: dict[str, Any],
    dataset_dir: str,
    checkpoint_dir: str,
    parent_model_path: str | None,
    result_queue: multiprocessing.Queue,
) -> None:
    """Run YOLO training in a subprocess for clean termination."""
    try:
        from ultralytics import YOLO

        data_yaml = Path(dataset_dir) / "data.yaml"
        if not data_yaml.exists():
            result_queue.put(TrainResult(success=False, epochs_completed=0, best_model_path=None, latest_model_path=None, error=f"data.yaml not found"))
            return

        if parent_model_path and Path(parent_model_path).exists():
            model = YOLO(parent_model_path)
        else:
            model = YOLO("yolov8n.pt")

        results = model.train(
            data=str(data_yaml),
            epochs=config.get("epochs", 3),
            imgsz=config.get("imgsz", 640),
            batch=config.get("batch", 16),
            device=config.get("device", "cpu"),
            patience=config.get("patience", 10),
            project=str(checkpoint_dir),
            name="train",
            exist_ok=True,
            verbose=False,
        )

        results_dir = Path(checkpoint_dir) / "train"
        best_src = results_dir / "weights" / "best.pt"
        last_src = results_dir / "weights" / "last.pt"

        best_path = str(best_src) if best_src.exists() else None
        latest_path = str(last_src) if last_src.exists() else None

        metrics = {}
        if hasattr(results, "results_dict"):
            metrics = {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in results.results_dict.items()}

        result_queue.put(TrainResult(
            success=True,
            epochs_completed=config.get("epochs", 3),
            best_model_path=best_path,
            latest_model_path=latest_path,
            metrics=metrics,
        ))

    except Exception as exc:
        logger.exception("Training failed")
        result_queue.put(TrainResult(
            success=False, epochs_completed=0, best_model_path=None, latest_model_path=None, error=str(exc),
        ))


class YoloTrainer:
    def __init__(self, config: dict[str, Any], check_cancel=None) -> None:
        self.config = config
        self._process: multiprocessing.Process | None = None
        self._check_cancel = check_cancel
        self._stop_event = threading.Event()

    def train(
        self,
        dataset_dir: Path,
        checkpoint_dir: Path,
        parent_model_path: str | None = None,
    ) -> TrainResult:
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_train_worker,
            args=(self.config, str(dataset_dir), str(checkpoint_dir), parent_model_path, result_queue),
            daemon=True,
        )
        self._process = proc
        self._stop_event.clear()
        proc.start()

        # Check cancellation in a separate thread
        def _cancel_checker():
            while proc.is_alive() and not self._stop_event.is_set():
                if self._check_cancel and self._check_cancel():
                    logger.info("Cancellation requested, terminating training subprocess")
                    self._stop_event.set()
                    proc.terminate()
                    return
                self._stop_event.wait(timeout=1.0)

        checker = threading.Thread(target=_cancel_checker, daemon=True)
        checker.start()
        proc.join()
        self._stop_event.set()
        checker.join(timeout=2)

        if result_queue.empty():
            return TrainResult(success=False, epochs_completed=0, best_model_path=None, latest_model_path=None, error="Training process returned no result")
        return result_queue.get(timeout=5)

    def stop(self) -> None:
        self._stop_event.set()
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
