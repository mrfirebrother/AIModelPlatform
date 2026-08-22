from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

logger = logging.getLogger("platform.inference")


def log_inference_summary(
    *,
    binding_id: UUID,
    model_node_id: UUID,
    release_id: UUID,
    generation: int,
    config_hash: str,
    image_format: str,
    image_size_bytes: int,
    latency_ms: float,
    status: str = "success",
    error: str | None = None,
) -> None:
    log_data: dict[str, Any] = {
        "binding_id": str(binding_id),
        "model_node_id": str(model_node_id),
        "release_id": str(release_id),
        "generation": generation,
        "config_hash": config_hash,
        "image_format": image_format,
        "image_size_bytes": image_size_bytes,
        "latency_ms": round(latency_ms, 2),
        "status": status,
        "timestamp": time.time(),
    }
    if error:
        log_data["error"] = error
        logger.warning("Inference failed: %s", log_data)
    else:
        logger.info("Inference completed: %s", log_data)
