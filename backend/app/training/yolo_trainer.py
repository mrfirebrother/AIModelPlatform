from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .yolo_dataset import YoloDataset
from .checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    success: bool
    epochs_completed: int
    best_model_path: str | None
    latest_model_path: str | None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class YoloTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.epochs = config.get("epochs", 3)
        self.imgsz = config.get("imgsz", 640)
        self.batch = config.get("batch", 16)
        self.model_name = config.get("model_name", "yolov8n")
        self.device = YoloDataset.resolve_device(config.get("device", "cpu"))
        self.patience = config.get("patience", 10)
        self.project = config.get("project", None)
        self.name = config.get("name", None)

    def train(
        self,
        dataset_dir: Path,
        checkpoint_dir: Path,
        parent_model_path: str | None = None,
    ) -> TrainResult:
        from ultralytics import YOLO

        data_yaml = dataset_dir / "data.yaml"
        if not data_yaml.exists():
            return TrainResult(
                success=False,
                epochs_completed=0,
                best_model_path=None,
                latest_model_path=None,
                error=f"data.yaml not found at {data_yaml}",
            )

        if parent_model_path:
            if not Path(parent_model_path).exists():
                return TrainResult(
                    success=False,
                    epochs_completed=0,
                    best_model_path=None,
                    latest_model_path=None,
                    error=f"Parent model not found: {parent_model_path}",
                )
            model = YOLO(parent_model_path)
        else:
            model = YOLO(f"{self.model_name}.pt")

        ckpt_mgr = CheckpointManager(checkpoint_dir)

        train_args: dict[str, Any] = {
            "data": str(data_yaml),
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device,
            "patience": self.patience,
            "project": str(checkpoint_dir),
            "name": "train",
            "exist_ok": True,
            "verbose": False,
        }

        try:
            results = model.train(**train_args)
        except Exception as exc:
            logger.exception("Training failed")
            return TrainResult(
                success=False,
                epochs_completed=0,
                best_model_path=None,
                latest_model_path=None,
                error=str(exc),
            )

        results_dir = Path(checkpoint_dir) / "train"
        best_src = results_dir / "weights" / "best.pt"
        last_src = results_dir / "weights" / "last.pt"

        best_path: str | None = None
        latest_path: str | None = None
        metrics: dict[str, Any] = {}

        if best_src.exists():
            cp = ckpt_mgr.save_checkpoint(best_src, epoch=self.epochs, is_best=True)
            best_path = str(cp.checkpoint_path)

        if last_src.exists():
            cp = ckpt_mgr.save_checkpoint(last_src, epoch=self.epochs, is_best=False)
            latest_path = str(cp.checkpoint_path)

        if hasattr(results, "results_dict"):
            results_dict = results.results_dict if hasattr(results, "results_dict") else {}
            metrics = {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in results_dict.items()
            }
        elif isinstance(results, dict):
            metrics = {k: v for k, v in results.items()}

        return TrainResult(
            success=True,
            epochs_completed=self.epochs,
            best_model_path=best_path,
            latest_model_path=latest_path,
            metrics=metrics,
        )
