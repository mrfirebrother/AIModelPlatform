from __future__ import annotations

from sqlalchemy.orm import Session

from backend.app.models import RuntimeInstance


class Reconciler:
    """Reconciles runtime state after service restart."""

    def recover_instances(self, session: Session) -> list[RuntimeInstance]:
        """Recover all non-terminal runtime instances after restart."""
        raise NotImplementedError

    def mark_failed_instances(self, session: Session) -> list[RuntimeInstance]:
        """Mark instances that haven't sent heartbeat as failed."""
        raise NotImplementedError
