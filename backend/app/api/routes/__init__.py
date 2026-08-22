from __future__ import annotations

from .health import router as health_router
from .models import router as models_router
from .bindings import router as bindings_router
from .datasets import router as datasets_router
from .training import router as training_router
from .evaluations import router as evaluations_router
from .releases import router as releases_router
from .resources import router as resources_router
from .infer import router as infer_router
from .gpu import router as gpu_router

__all__ = [
    "health_router",
    "models_router",
    "bindings_router",
    "datasets_router",
    "training_router",
    "evaluations_router",
    "releases_router",
    "resources_router",
    "infer_router",
    "gpu_router",
]
