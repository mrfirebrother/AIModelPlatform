from __future__ import annotations

import base64
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from .postprocessor import Detection, Postprocessor
from .preprocessor import Preprocessor

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata about a loaded model."""

    model_key: str
    model_path: str
    num_classes: int
    class_names: dict[int, str]
    loaded_at: float = 0.0


class YoloEngine:
    """YOLO inference engine with model caching and CPU support."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir
        self._loaded_models: dict[str, Any] = {}
        self._model_info: dict[str, ModelInfo] = {}
        self._preprocessor = Preprocessor()
        self._lock = threading.Lock()

    def load_model(self, model_path: str, model_key: str | None = None) -> ModelInfo:
        """Load a YOLO model. Returns cached instance if already loaded."""
        key = model_key or model_path

        with self._lock:
            if key in self._loaded_models:
                return self._model_info[key]

        model = YOLO(model_path)
        class_names: dict[int, str] = {}
        if hasattr(model, "names") and model.names:
            class_names = dict(model.names)

        info = ModelInfo(
            model_key=key,
            model_path=model_path,
            num_classes=len(class_names),
            class_names=class_names,
            loaded_at=time.time(),
        )

        with self._lock:
            self._loaded_models[key] = model
            self._model_info[key] = info

        logger.info("Loaded model %s from %s (%d classes)", key, model_path, len(class_names))
        return info

    def get_model(self, model_key: str) -> Any | None:
        """Get a loaded model by key, or None."""
        return self._loaded_models.get(model_key)

    def unload_model(self, model_key: str) -> bool:
        """Unload a model from memory. Returns True if found and removed."""
        with self._lock:
            if model_key in self._loaded_models:
                del self._loaded_models[model_key]
                self._model_info.pop(model_key, None)
                logger.info("Unloaded model %s", model_key)
                return True
            return False

    def list_models(self) -> list[str]:
        """Return list of loaded model keys."""
        return list(self._loaded_models.keys())

    def predict(
        self,
        model_key: str,
        image_bytes: bytes,
        image_format: str,
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
    ) -> list[Detection]:
        """Run YOLO inference on image bytes and return detections."""
        if model_key not in self._loaded_models:
            raise KeyError(f"Model '{model_key}' not loaded")

        model = self._loaded_models[model_key]
        info = self._model_info[model_key]

        pil_image = self._preprocessor.decode_image(
            base64.b64encode(image_bytes).decode(), image_format
        )

        results = model(pil_image, verbose=False)

        postprocessor = Postprocessor(
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            class_names=info.class_names,
        )

        detections: list[Detection] = []
        if results and len(results) > 0:
            result = results[0]
            # Segmentation models carry instance masks aligned with the boxes;
            # expose each mask as a polygon so consumers can draw/measure the
            # actual defect region, not just its box.
            mask_polys = None
            if hasattr(result, "masks") and result.masks is not None and hasattr(result.masks, "xy"):
                mask_polys = [poly.tolist() for poly in result.masks.xy]
            if hasattr(result, "boxes") and result.boxes is not None:
                xyxy = result.boxes.xyxy
                confs = result.boxes.conf
                clss = result.boxes.cls
                for i in range(len(xyxy)):
                    box = xyxy[i].tolist() if hasattr(xyxy[i], "tolist") else list(xyxy[i])
                    conf = float(confs[i]) if hasattr(confs, "__getitem__") else 0.0
                    cls_id = int(clss[i]) if hasattr(clss, "__getitem__") else 0
                    mask = None
                    if mask_polys is not None and i < len(mask_polys):
                        mask = [
                            [round(px, 2), round(py, 2)] for px, py in mask_polys[i]
                        ]
                    detections.append(
                        Detection(
                            class_name="",
                            confidence=conf,
                            bbox=box,
                            class_id=cls_id,
                            mask=mask,
                        )
                    )

        return postprocessor.process(detections)
