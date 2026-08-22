from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import TrainingAttempt


LEASE_DURATION_MINUTES = 30


def heartbeat(
    session: Session,
    attempt_id: UUID,
    lease_token: UUID,
) -> TrainingAttempt:
    attempt = session.get(TrainingAttempt, attempt_id, with_for_update=True)
    if attempt is None:
        raise ValueError(f"Training attempt {attempt_id} not found")
    if attempt.lease_token != lease_token:
        raise ValueError("Invalid lease token")
    if attempt.status not in ("running", "recovering"):
        return attempt
    now = datetime.now(timezone.utc)
    attempt.heartbeat_at = now
    attempt.lease_expires_at = now + timedelta(minutes=LEASE_DURATION_MINUTES)
    session.flush()
    return attempt


def find_expired_attempts(session: Session) -> list[TrainingAttempt]:
    now = datetime.now(timezone.utc)
    return list(
        session.execute(
            select(TrainingAttempt).where(
                TrainingAttempt.status.in_(["running", "recovering"]),
                TrainingAttempt.lease_expires_at.isnot(None),
                TrainingAttempt.lease_expires_at < now,
            )
        ).scalars().all()
    )


def expire_attempt(session: Session, attempt_id: UUID) -> TrainingAttempt:
    attempt = session.get(TrainingAttempt, attempt_id, with_for_update=True)
    if attempt is None:
        raise ValueError(f"Training attempt {attempt_id} not found")
    if attempt.status not in ("running", "recovering"):
        return attempt
    attempt.status = "failed"
    attempt.last_error = "Lease expired"
    attempt.finished_at = datetime.now(timezone.utc)
    session.flush()
    return attempt
