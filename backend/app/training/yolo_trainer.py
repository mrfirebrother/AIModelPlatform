from __future__ import annotations

import logging
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


class _CancelCallback:
    """YOLO callback that checks for cancellation after each epoch."""
    def __init__(self, check_cancel) -> None:
        self._check_cancel = check_cancel
        self.cancelled = False

    def __call__(self, trainer) -> None:
        if self._check_cancel and self._check_cancel():
            self.cancelled = True
            trainer.stop = True  # YOLO supports trainer.stop to abort


class YoloTrainer:
    def __init__(self, config: dict[str, Any], check_cancel=None) -> None:
        self.config = config
        self._check_cancel = check_cancel

    def train(
        self,
        dataset_dir: Path,
        checkpoint_dir: Path,
        parent_model_path: str | None = None,
    ) -> TrainResult:
        from ultralytics import YOLO

        data_yaml = dataset_dir / "data.yaml"
        if not data_yaml.exists():
            return TrainResult(success=False, epochs_completed=0, best_model_path=None, latest_model_path=None, error=f"data.yaml not found")

        if parent_model_path and Path(parent_model_path).exists():
            model = YOLO(parent_model_path)
        else:
            model = YOLO("yolov8n.pt")

        train_args: dict[str, Any] = {
            "data": str(data_yaml),
            "epochs": self.config.get("epochs", 3),
            "imgsz": self.config.get("imgsz", 640),
            "batch": self.config.get("batch", 16),
            "device": self.config.get("device", "cpu"),
            "patience": self.config.get("patience", 10),
            "workers": 0,  # Avoid daemon processes
            "amp": False,  # Disable mixed precision
            "project": str(checkpoint_dir),
            "name": "train",
            "exist_ok": True,
            "verbose": False,
        }

        cancel_cb = _CancelCallback(self._check_cancel) if self._check_cancel else None
        if cancel_cb:
            for evt in ("on_train_batch_end", "on_train_epoch_end"):
                try:
                    model.add_callback(evt, cancel_cb)
                except Exception:
                    logger.warning("add_callback %s not supported", evt)

        try:
            results = model.train(**train_args)
        except Exception as exc:
            logger.exception("Training failed")
            return TrainResult(success=False, epochs_completed=0, best_model_path=None, latest_model_path=None, error=str(exc))

        if cancel_cb and cancel_cb.cancelled:
            return TrainResult(success=False, epochs_completed=0, best_model_path=None, latest_model_path=None, error="Training cancelled")

        results_dir = Path(checkpoint_dir) / "train"
        best_src = results_dir / "weights" / "best.pt"
        last_src = results_dir / "weights" / "last.pt"

        best_path = str(best_src) if best_src.exists() else None
        latest_path = str(last_src) if last_src.exists() else None

        metrics = {}
        if hasattr(results, "results_dict"):
            metrics = {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in results.results_dict.items()}

        return TrainResult(
            success=True,
            epochs_completed=self.config.get("epochs", 3),
            best_model_path=best_path,
            latest_model_path=latest_path,
            metrics=metrics,
        )

    def stop(self) -> None:
        pass  # No subprocess to stop
