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
    logger.info("Starting evaluation for %s", evaluation_id)
    return f"evaluation {evaluation_id} dispatched"


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

        lease = scheduler.reserve_gpu(
            attempt_id=evaluation.id,
            required_memory_mb=required_memory_mb,
        )

        evaluator = YoloEvaluator(
            confidence_threshold=0.25,
            iou_threshold=0.45,
            device="cpu",
        )
        evaluator.load_model(model_path)

        images = _load_dataset_images(dataset_manifest)
        ground_truths = _load_ground_truths(dataset_manifest)
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
    return []


def _load_ground_truths(
    dataset_manifest: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    return []


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
