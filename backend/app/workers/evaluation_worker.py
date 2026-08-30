from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.evaluation.metrics import compute_mAP, compute_per_class_metrics
from backend.app.evaluation.policy import (
    EvaluationPolicy,
    PolicyValidationResult,
    validate_policy_for_run,
)
from backend.app.evaluation.report import write_report
from backend.app.evaluation.yolo_evaluator import EvalResult, YoloEvaluator
from backend.app.models import DatasetSnapshot, Evaluation, ModelNode
from backend.app.services.evaluation_service import (
    mark_evaluation_failed,
    mark_evaluation_running,
    record_metrics,
    write_report_path,
)
from backend.app.training.resource_scheduler import (
    InsufficientGPUError,
    ResourceScheduler,
)

from .celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="platform.evaluation.run_evaluation")
def run_evaluation(evaluation_id: str) -> str:
    from backend.app.workers.db_session import worker_session
    from uuid import UUID
    logger.info("Starting evaluation for %s", evaluation_id)
    try:
        ev_id = UUID(evaluation_id)
    except ValueError:
        return f"invalid evaluation id {evaluation_id}"
    with worker_session() as session:
        ev = session.get(Evaluation, ev_id)
        if ev is None:
            return f"evaluation {evaluation_id} not found"
        if ev.auto_status != "pending":
            return f"evaluation {evaluation_id} not pending ({ev.auto_status})"
        model = session.get(ModelNode, ev.model_node_id)
        if model is None or not Path(model.artifact_path).exists():
            mark_evaluation_failed(session, ev.id, "Model artifact not found")
            session.commit()
            return f"evaluation {evaluation_id} failed: model not found"
        snap = session.get(DatasetSnapshot, ev.dataset_snapshot_id)
        if snap is None:
            mark_evaluation_failed(session, ev.id, "Dataset snapshot not found")
            session.commit()
            return f"evaluation {evaluation_id} failed: snapshot not found"
        manifest = {
            "train_files": snap.train_manifest_json or [],
            "val_files": snap.val_manifest_json or [],
            "test_files": snap.test_manifest_json or [],
            "nc": len(snap.label_schema.classes) if snap.label_schema else 0,
            "names": [c.semantic_key for c in snap.label_schema.classes] if snap.label_schema else [],
        }
        from backend.app.config import get_settings
        settings = get_settings()
        output_dir = Path(settings.log_dir) / "evaluations" / str(ev.id)
        result = execute_evaluation(session=session, evaluation=ev, model_path=model.artifact_path, dataset_manifest=manifest, output_dir=output_dir)
        session.commit()
        return f"evaluation {evaluation_id} {'passed' if result.success and result.mAP50>0 else 'completed'} mAP50={result.mAP50:.4f}"


def execute_evaluation(
    *,
    session: Session,
    evaluation: Evaluation,
    model_path: str,
    dataset_manifest: dict[str, Any],
    output_dir: Path,
    required_memory_mb: int = 2048,
) -> EvalResult:
    lease = None
    scheduler = ResourceScheduler(session=session, cpu_mode=True)

    try:
        evaluation = mark_evaluation_running(session, evaluation.id)
        session.commit()

        evaluator = YoloEvaluator(
            confidence_threshold=0.25,
            iou_threshold=0.45,
            device="cpu",
        )
        evaluator.load_model(model_path)

        images = _load_dataset_images(dataset_manifest)
        ground_truths = _load_ground_truths(
            dataset_manifest,
            image_sizes=[image.size for image in images],
        )
        class_ids = _extract_class_ids(dataset_manifest)

        eval_result = evaluator.evaluate(
            dataset_images=images,
            dataset_ground_truths=ground_truths,
            class_ids=class_ids,
        )

        if not eval_result.success:
            mark_evaluation_failed(session, evaluation.id, eval_result.error)
            session.commit()
            return eval_result

        policy_dict = evaluation.evaluation_policy_json
        policy = EvaluationPolicy.from_dict(policy_dict)

        metrics_dict = {
            "mAP50": eval_result.mAP50,
            "mAP50_95": eval_result.mAP50_95,
            "precision": eval_result.precision,
            "recall": eval_result.recall,
            "per_class": eval_result.per_class,
            "num_images": eval_result.num_images,
            "num_predictions": eval_result.num_predictions,
            "num_ground_truths": eval_result.num_ground_truths,
        }

        test_classes = set(eval_result.per_class.keys())
        schema_classes = set(class_ids)

        policy_validation = validate_policy_for_run(
            policy,
            metrics_dict,
            test_classes=test_classes,
            schema_classes=schema_classes,
            parent_metrics=None,
        )

        passed = policy_validation.passed
        record_metrics(session, evaluation.id, metrics_dict, passed=passed)
        session.commit()

        report, report_hash = write_report(
            evaluation_id=str(evaluation.id),
            metrics=metrics_dict,
            policy=policy_dict,
            warnings=policy_validation.warnings,
            regression_failures=policy_validation.regression_failures,
        )

        report_path = output_dir / f"evaluation_{evaluation.id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        write_report_path(session, evaluation.id, str(report_path), report_hash)
        session.commit()

        logger.info(
            "Evaluation %s completed: passed=%s, mAP50=%.4f",
            evaluation.id,
            passed,
            eval_result.mAP50,
        )

        return eval_result

    except InsufficientGPUError as exc:
        mark_evaluation_failed(session, evaluation.id, f"GPU resource error: {exc}")
        session.commit()
        return EvalResult(
            success=False,
            mAP50=0.0,
            mAP50_95=0.0,
            precision=0.0,
            recall=0.0,
            per_class={},
            num_images=0,
            num_predictions=0,
            num_ground_truths=0,
            error=str(exc),
        )

    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        mark_evaluation_failed(session, evaluation.id, str(exc))
        session.commit()
        return EvalResult(
            success=False,
            mAP50=0.0,
            mAP50_95=0.0,
            precision=0.0,
            recall=0.0,
            per_class={},
            num_images=0,
            num_predictions=0,
            num_ground_truths=0,
            error=str(exc),
        )

    finally:
        if lease is not None:
            scheduler.release_gpu(evaluation.id)


def _load_dataset_images(
    dataset_manifest: dict[str, Any],
) -> list[Any]:
    from PIL import Image
    images: list[Any] = []
    source_dir = Path(dataset_manifest.get("source_dir", ""))
    for entry in dataset_manifest.get("test_files", []):
        p = entry.get("image_stored_path")
        if not p or not Path(p).exists():
            # Fall back to source_dir + relative path
            img_rel = entry.get("image", "")
            if img_rel and source_dir:
                p = str(source_dir / img_rel)
        if not p or not Path(p).exists():
            continue
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception:
            continue
    return images


def _load_ground_truths(
    dataset_manifest: dict[str, Any],
    image_sizes: list[tuple[int, int]] | None = None,
) -> list[list[dict[str, Any]]]:
    gts: list[list[dict[str, Any]]] = []
    source_dir = Path(dataset_manifest.get("source_dir", ""))
    for index, entry in enumerate(dataset_manifest.get("test_files", [])):
        p = entry.get("label_stored_path")
        if not p or not Path(p).exists():
            lbl_rel = entry.get("label", "")
            if lbl_rel and source_dir:
                p = str(source_dir / lbl_rel)
        boxes: list[dict[str, Any]] = []
        width, height = image_sizes[index] if image_sizes and index < len(image_sizes) else (1, 1)
        if p and Path(p).exists():
            try:
                for line in Path(p).read_text(encoding="utf-8").strip().splitlines():
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])
                    # convert yolo normalized xywh to xyxy normalized
                    x1 = (xc - w / 2) * width
                    y1 = (yc - h / 2) * height
                    x2 = (xc + w / 2) * width
                    y2 = (yc + h / 2) * height
                    boxes.append({"class_id": cid, "bbox": [x1, y1, x2, y2]})
            except Exception:
                pass
        gts.append(boxes)
    return gts


def _extract_class_ids(dataset_manifest: dict[str, Any]) -> list[int]:
    names = dataset_manifest.get("names", {})
    if isinstance(names, dict):
        return sorted(int(k) for k in names.keys())
    elif isinstance(names, list):
        return list(range(len(names)))
    return []


@celery_app.task(name="platform.evaluation.recover_evaluation")
def recover_evaluation(evaluation_id: str) -> str:
    logger.warning("Evaluation recovery not implemented for %s", evaluation_id)
    return f"evaluation recovery for {evaluation_id} not implemented"
