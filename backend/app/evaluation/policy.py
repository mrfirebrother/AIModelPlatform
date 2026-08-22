from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationPolicy:
    min_mAP50: float
    min_precision: float
    min_recall: float
    max_regression_ratio: float
    require_per_class_coverage: bool

    def __post_init__(self) -> None:
        for name in ("min_mAP50", "min_precision", "min_recall"):
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1, got {val}")
        if self.max_regression_ratio < 0:
            raise ValueError(
                f"max_regression_ratio must be >= 0, got {self.max_regression_ratio}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_mAP50": self.min_mAP50,
            "min_precision": self.min_precision,
            "min_recall": self.min_recall,
            "max_regression_ratio": self.max_regression_ratio,
            "require_per_class_coverage": self.require_per_class_coverage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationPolicy:
        return cls(
            min_mAP50=data["min_mAP50"],
            min_precision=data["min_precision"],
            min_recall=data["min_recall"],
            max_regression_ratio=data["max_regression_ratio"],
            require_per_class_coverage=data.get("require_per_class_coverage", True),
        )


@dataclass
class PolicyValidationResult:
    passed: bool
    warnings: list[str] = field(default_factory=list)
    regression_failures: list[str] = field(default_factory=list)
    coverage_failures: list[str] = field(default_factory=list)


@dataclass
class CoverageResult:
    passed: bool
    missing_classes: set[int] = field(default_factory=set)


def validate_policy_for_run(
    policy: EvaluationPolicy,
    metrics: dict[str, Any],
    test_classes: set[int],
    schema_classes: set[int],
    parent_metrics: dict[str, Any] | None,
) -> PolicyValidationResult:
    warnings: list[str] = []
    regression_failures: list[str] = []
    coverage_failures: list[str] = []

    missing = schema_classes - test_classes
    if missing:
        for cls_id in sorted(missing):
            warnings.append(
                f"Test set missing class {cls_id}; metrics for this class will be unreliable"
            )

    passed = True

    mAP50 = metrics.get("mAP50", 0.0)
    precision = metrics.get("precision", 0.0)
    recall = metrics.get("recall", 0.0)

    if mAP50 < policy.min_mAP50:
        passed = False
        regression_failures.append(
            f"mAP50 {mAP50:.4f} < threshold {policy.min_mAP50}"
        )
    if precision < policy.min_precision:
        passed = False
        regression_failures.append(
            f"precision {precision:.4f} < threshold {policy.min_precision}"
        )
    if recall < policy.min_recall:
        passed = False
        regression_failures.append(
            f"recall {recall:.4f} < threshold {policy.min_recall}"
        )

    if parent_metrics is not None:
        reg = validate_regression(
            metrics, parent_metrics, policy.max_regression_ratio
        )
        regression_failures.extend(reg)

    if regression_failures:
        passed = False

    return PolicyValidationResult(
        passed=passed,
        warnings=warnings,
        regression_failures=regression_failures,
        coverage_failures=coverage_failures,
    )


def validate_regression(
    current: dict[str, Any],
    parent: dict[str, Any],
    max_regression_ratio: float,
) -> list[str]:
    failures: list[str] = []
    for key in ("mAP50", "precision", "recall"):
        cur = float(current.get(key, 0.0))
        par = float(parent.get(key, 0.0))
        if par > 0 and cur < par:
            drop = (par - cur) / par
            if drop > max_regression_ratio:
                failures.append(
                    f"{key} regressed {drop:.2%} (from {par:.4f} to {cur:.4f}), "
                    f"max allowed {max_regression_ratio:.2%}"
                )
    return failures


def validate_publish_coverage(
    schema_classes: set[int],
    test_classes: set[int],
) -> CoverageResult:
    missing = schema_classes - test_classes
    return CoverageResult(passed=len(missing) == 0, missing_classes=missing)
