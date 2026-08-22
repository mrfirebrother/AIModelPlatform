from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from backend.app.training.yolo_dataset import YoloDataset
from backend.app.training.yolo_trainer import YoloTrainer
from backend.app.training.checkpoint_manager import CheckpointManager


def _make_real_dataset(dest: Path) -> Path:
    """Create a tiny YOLO dataset with synthetic images for real training."""
    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "images" / "train").mkdir(parents=True)
    (dest / "images" / "val").mkdir(parents=True)
    (dest / "labels" / "train").mkdir(parents=True)
    (dest / "labels" / "val").mkdir(parents=True)

    for name in ["a", "b", "c"]:
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(dest / "images" / "train" / f"{name}.jpg")
        (dest / "labels" / "train" / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    for name in ["d", "e"]:
        img = Image.new("RGB", (64, 64), color=(64, 64, 64))
        img.save(dest / "images" / "val" / f"{name}.jpg")
        (dest / "labels" / "val" / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    import yaml

    data_yaml = {
        "path": str(dest),
        "train": str(dest / "images" / "train"),
        "val": str(dest / "images" / "val"),
        "nc": 1,
        "names": ["object"],
    }
    (dest / "data.yaml").write_text(yaml.dump(data_yaml), encoding="utf-8")
    return dest


@pytest.fixture()
def real_dataset(tmp_path: Path) -> Path:
    return _make_real_dataset(tmp_path / "dataset")


@pytest.fixture()
def checkpoint_dir(tmp_path: Path) -> Path:
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


@pytest.mark.slow
class TestRealYoloTraining:
    def test_training_completes_on_cpu(self, real_dataset: Path, checkpoint_dir: Path) -> None:
        config = {
            "epochs": 2,
            "imgsz": 64,
            "batch": 1,
            "device": "cpu",
            "model_name": "yolov8n",
            "patience": 1,
        }
        trainer = YoloTrainer(config)
        result = trainer.train(real_dataset, checkpoint_dir)

        assert result.success is True
        assert result.epochs_completed >= 1
        assert result.best_model_path is not None
        assert Path(result.best_model_path).exists()

    def test_training_with_different_epochs(self, real_dataset: Path, checkpoint_dir: Path) -> None:
        config = {
            "epochs": 3,
            "imgsz": 64,
            "batch": 1,
            "device": "cpu",
            "model_name": "yolov8n",
        }
        trainer = YoloTrainer(config)
        result = trainer.train(real_dataset, checkpoint_dir)

        assert result.success is True
        assert result.epochs_completed == 3

    def test_training_returns_metrics(self, real_dataset: Path, checkpoint_dir: Path) -> None:
        config = {
            "epochs": 2,
            "imgsz": 64,
            "batch": 1,
            "device": "cpu",
            "model_name": "yolov8n",
        }
        trainer = YoloTrainer(config)
        result = trainer.train(real_dataset, checkpoint_dir)

        assert result.metrics is not None
        assert isinstance(result.metrics, dict)

    def test_training_with_parent_model(self, real_dataset: Path, checkpoint_dir: Path, tmp_path: Path) -> None:
        from ultralytics import YOLO

        base_model = YOLO("yolov8n.pt")
        parent_path = tmp_path / "parent.pt"
        base_model.save(str(parent_path))

        config = {
            "epochs": 2,
            "imgsz": 64,
            "batch": 1,
            "device": "cpu",
            "model_name": "yolov8n",
        }
        trainer = YoloTrainer(config)
        result = trainer.train(real_dataset, checkpoint_dir, parent_model_path=str(parent_path))

        assert result.success is True


class TestCheckpointManager:
    def test_save_and_load_best(self, checkpoint_dir: Path) -> None:
        mgr = CheckpointManager(checkpoint_dir)
        src = checkpoint_dir / "source.pt"
        src.write_bytes(b"fake model data")

        cp = mgr.save_checkpoint(src, epoch=5, is_best=True)
        assert cp.checkpoint_path.exists()
        assert cp.is_best is True

        best = mgr.get_best_checkpoint()
        assert best is not None
        assert best.epoch == 5

    def test_save_and_load_latest(self, checkpoint_dir: Path) -> None:
        mgr = CheckpointManager(checkpoint_dir)

        src1 = checkpoint_dir / "src1.pt"
        src1.write_bytes(b"epoch 1 data")
        mgr.save_checkpoint(src1, epoch=1, is_best=False)

        src2 = checkpoint_dir / "src2.pt"
        src2.write_bytes(b"epoch 2 data")
        mgr.save_checkpoint(src2, epoch=2, is_best=False)

        latest = mgr.get_latest_checkpoint()
        assert latest is not None
        assert latest.epoch == 2

    def test_best_checkpoint_overwrites(self, checkpoint_dir: Path) -> None:
        mgr = CheckpointManager(checkpoint_dir)

        src1 = checkpoint_dir / "s1.pt"
        src1.write_bytes(b"bad model")
        mgr.save_checkpoint(src1, epoch=1, is_best=True)

        src2 = checkpoint_dir / "s2.pt"
        src2.write_bytes(b"better model")
        mgr.save_checkpoint(src2, epoch=3, is_best=True)

        best = mgr.get_best_checkpoint()
        assert best is not None
        assert best.epoch == 3

    def test_empty_checkpoint_dir_returns_none(self, checkpoint_dir: Path) -> None:
        mgr = CheckpointManager(checkpoint_dir)
        assert mgr.get_best_checkpoint() is None
        assert mgr.get_latest_checkpoint() is None
