from __future__ import annotations

from uuid import UUID


class InferenceWorker:
    """Placeholder inference worker."""

    def start(self, instance_id: UUID) -> None:
        """Start inference serving for an instance."""
        raise NotImplementedError

    def stop(self, instance_id: UUID) -> None:
        """Stop inference serving for an instance."""
        raise NotImplementedError

    def health_check(self, instance_id: UUID) -> bool:
        """Check if the worker is healthy."""
        raise NotImplementedError
