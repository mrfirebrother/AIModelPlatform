from celery import Celery

from ..config import get_settings

settings = get_settings()
celery_app = Celery(
    "ai_model_platform",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_default_queue="default",
    task_routes={
        "platform.training.*": {"queue": "training"},
        "platform.evaluation.*": {"queue": "evaluation"},
        "platform.scheduler.*": {"queue": "default"},
    },
    beat_schedule={
        "sweep-pending-evaluations": {
            "task": "platform.scheduler.sweep_pending_evaluations",
            "schedule": 60.0,
        },
        "reconcile-leases": {
            "task": "platform.scheduler.reconcile_leases",
            "schedule": 60.0,
        },
    },
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
)

# Import placeholder tasks after the Celery application exists so Beat can discover them.
from . import scheduler as _scheduler  # noqa: E402,F401
from . import train_worker as _train_worker  # noqa: E402,F401
