from .celery_app import celery_app


@celery_app.task(name="platform.scheduler.sweep_pending_evaluations")
def sweep_pending_evaluations() -> str:
    """Placeholder for a future idempotent evaluation enqueue sweep."""
    return "pending evaluation sweep not implemented"


@celery_app.task(name="platform.scheduler.reconcile_leases")
def reconcile_leases() -> str:
    """Placeholder for future lease and worker reconciliation."""
    return "lease reconciliation not implemented"
