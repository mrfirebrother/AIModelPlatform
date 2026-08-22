from __future__ import annotations

import hashlib
import json
from typing import Any


def generate_evaluation_report(
    evaluation_id: str,
    metrics: dict[str, Any],
    policy: dict[str, Any],
    warnings: list[str] | None = None,
    regression_failures: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "evaluation_id": evaluation_id,
        "metrics": metrics,
        "policy": policy,
        "warnings": warnings or [],
        "regression_failures": regression_failures or [],
        "auto_passed": len(regression_failures or []) == 0
        and metrics.get("mAP50", 0) >= policy.get("min_mAP50", 0),
    }


def write_report(
    evaluation_id: str,
    metrics: dict[str, Any],
    policy: dict[str, Any],
    warnings: list[str] | None = None,
    regression_failures: list[str] | None = None,
) -> tuple[dict[str, Any], str]:
    report = generate_evaluation_report(
        evaluation_id, metrics, policy, warnings, regression_failures
    )
    report_bytes = json.dumps(report, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    report_hash = "sha256:" + hashlib.sha256(report_bytes).hexdigest()
    return report, report_hash
