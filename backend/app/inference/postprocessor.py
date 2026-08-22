from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Detection:
    """Single detection result."""

    class_name: str
    confidence: float
    bbox: list[float] = field(default_factory=list)
    class_id: int = 0

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "class_id": self.class_id,
        }


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Postprocessor:
    """Post-process YOLO raw detections: NMS, confidence filtering, class mapping."""

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        class_names: dict[int, str] | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.class_names = class_names or {}

    def filter_by_confidence(self, detections: list[Detection]) -> list[Detection]:
        """Remove detections below confidence threshold."""
        return [d for d in detections if d.confidence >= self.confidence_threshold]

    def apply_class_mapping(self, detection: Detection) -> Detection:
        """Map class_id to class_name using the class_names map."""
        if detection.class_id in self.class_names:
            detection.class_name = self.class_names[detection.class_id]
        return detection

    def apply_nms(self, detections: list[Detection]) -> list[Detection]:
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        if not detections:
            return []

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep: list[Detection] = []

        for det in sorted_dets:
            is_duplicate = False
            for kept in keep:
                if (
                    det.class_id == kept.class_id
                    and _compute_iou(det.bbox, kept.bbox) > self.nms_iou_threshold
                ):
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(det)

        return keep

    def process(self, detections: list[Detection]) -> list[Detection]:
        """Full post-processing pipeline."""
        filtered = self.filter_by_confidence(detections)
        mapped = [self.apply_class_mapping(d) for d in filtered]
        return self.apply_nms(mapped)
