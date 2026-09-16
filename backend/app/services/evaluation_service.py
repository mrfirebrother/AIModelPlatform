from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.evaluation.policy import EvaluationPolicy
from backend.app.models import Evaluation, ModelNode
from backend.app.models.base import _store_immutable_snapshot


def create_evaluation(
    session: Session,
    *,
    model_node_id: UUID,
    dataset_snapshot_id: UUID,
    policy: EvaluationPolicy,
) -> Evaluation:
    model = session.get(ModelNode, model_node_id)
    if model is None:
        raise ValueError(f"Model node {model_node_id} not found")
    if model.status not in ("candidate",):
        raise ValueError(f"Cannot evaluate model in '{model.status}' status")

    ev = Evaluation(
        model_node_id=model_node_id,
        dataset_snapshot_id=dataset_snapshot_id,
        auto_status="pending",
        attempt_status="pending",
        evaluation_policy_json=policy.to_dict(),
        auto_metrics_json={},
    )
    session.add(ev)
    session.flush()
    return ev


def get_evaluation(session: Session, evaluation_id: UUID) -> Evaluation | None:
    return session.get(Evaluation, evaluation_id)


def list_pending_evaluations(session: Session) -> list[Evaluation]:
    return list(
        session.execute(
            select(Evaluation).where(Evaluation.auto_status == "pending")
        ).scalars().all()
    )


def mark_evaluation_running(session: Session, evaluation_id: UUID) -> Evaluation:
    ev = session.get(Evaluation, evaluation_id, with_for_update=True)
    if ev is None:
        raise ValueError(f"Evaluation {evaluation_id} not found")
    if ev.auto_status not in ("pending",):
        raise ValueError(f"Cannot start evaluation in '{ev.auto_status}' status")

    now = datetime.now(timezone.utc)
    ev.auto_status = "running"
    ev.attempt_status = "running"
    ev.attempt_no += 1
    ev.lease_token = uuid4()
    ev.fencing_token = ev.attempt_no
    ev.heartbeat_at = now
    ev.lease_expires_at = now
    _store_immutable_snapshot(ev)
    session.flush()
    return ev


def record_metrics(
    session: Session,
    evaluation_id: UUID,
    metrics: dict,
    passed: bool,
) -> Evaluation:
    ev = session.get(Evaluation, evaluation_id, with_for_update=True)
    if ev is None:
        raise ValueError(f"Evaluation {evaluation_id} not found")
    if ev.auto_status not in ("running",):
        raise ValueError(f"Cannot record metrics for evaluation in '{ev.auto_status}' status")

    ev.auto_metrics_json = metrics
    _store_immutable_snapshot(ev)
    ev.auto_status = "passed" if passed else "failed"
    ev.attempt_status = "passed" if passed else "failed"
    session.flush()
    return ev


def mark_evaluation_failed(
    session: Session, evaluation_id: UUID, error: str
) -> Evaluation:
    ev = session.get(Evaluation, evaluation_id, with_for_update=True)
    if ev is None:
        raise ValueError(f"Evaluation {evaluation_id} not found")
    ev.auto_status = "failed"
    ev.attempt_status = "failed"
    ev.last_error = error
    _store_immutable_snapshot(ev)
    session.flush()
    return ev


def write_report_path(
    session: Session,
    evaluation_id: UUID,
    report_path: str,
    report_hash: str,
) -> Evaluation:
    ev = session.get(Evaluation, evaluation_id, with_for_update=True)
    if ev is None:
        raise ValueError(f"Evaluation {evaluation_id} not found")
    ev.report_path = report_path
    ev.report_hash = report_hash
    session.flush()
    return ev


def record_human_review(
    session: Session,
    evaluation_id: UUID,
    *,
    reviewer: str,
    conclusion: str,
    comments: str | None = None,
) -> Evaluation:
    """REMOVED FEATURE - kept only as a hard failure so callers notice.

    There is no review/approval step in this platform by design: the quality gate produced a
    pass/fail label nobody acted on, and no one owns the reviewer role. The
    ``POST /api/evaluations/{id}/review`` endpoint is gone and migration ``0005`` dropped the
    ``human_status`` / ``reviewer`` / ``human_conclusion`` / ``human_comments`` / ``reviewed_at``
    columns, so there is nowhere left to record a review. Evaluations only carry metrics.
    """
    raise ValueError(
        "人工复核已移除：本平台没有审核机制，评测只提供指标（Precision/Recall/mAP50/mAP50-95）"
    )
