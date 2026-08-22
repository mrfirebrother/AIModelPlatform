from __future__ import annotations

import pytest

from backend.app.evaluation.policy import (
    EvaluationPolicy,
    validate_policy_for_run,
    validate_publish_coverage,
    validate_regression,
)


class TestEvaluationPolicyValidation:
    def test_valid_policy(self) -> None:
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        assert policy.min_mAP50 == 0.5

    def test_policy_rejects_negative_thresholds(self) -> None:
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            EvaluationPolicy(
                min_mAP50=-0.1,
                min_precision=0.4,
                min_recall=0.4,
                max_regression_ratio=0.1,
                require_per_class_coverage=True,
            )

    def test_policy_rejects_threshold_over_one(self) -> None:
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            EvaluationPolicy(
                min_mAP50=1.5,
                min_precision=0.4,
                min_recall=0.4,
                max_regression_ratio=0.1,
                require_per_class_coverage=True,
            )

    def test_policy_rejects_negative_regression_ratio(self) -> None:
        with pytest.raises(ValueError, match="max_regression_ratio"):
            EvaluationPolicy(
                min_mAP50=0.5,
                min_precision=0.4,
                min_recall=0.4,
                max_regression_ratio=-0.1,
                require_per_class_coverage=True,
            )


class TestValidatePolicyForRun:
    def test_first_generation_no_parent_skips_regression(self) -> None:
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        metrics = {
            "mAP50": 0.6,
            "precision": 0.7,
            "recall": 0.6,
            "per_class": {0: {"precision": 0.7, "recall": 0.6}},
        }
        result = validate_policy_for_run(
            policy,
            metrics,
            test_classes={0},
            schema_classes={0},
            parent_metrics=None,
        )
        assert result.passed is True
        assert len(result.warnings) == 0

    def test_first_generation_missing_classes_warning(self) -> None:
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        metrics = {
            "mAP50": 0.6,
            "precision": 0.7,
            "recall": 0.6,
            "per_class": {0: {"precision": 0.7, "recall": 0.6}},
        }
        result = validate_policy_for_run(
            policy,
            metrics,
            test_classes={0},
            schema_classes={0, 1},
            parent_metrics=None,
        )
        assert result.passed is True
        assert any("class 1" in w.lower() for w in result.warnings)

    def test_generation_1_with_parent_checks_regression(self) -> None:
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        metrics = {
            "mAP50": 0.55,
            "precision": 0.6,
            "recall": 0.5,
            "per_class": {0: {"precision": 0.6, "recall": 0.5}},
        }
        parent_metrics = {
            "mAP50": 0.7,
            "precision": 0.7,
            "recall": 0.6,
            "per_class": {0: {"precision": 0.7, "recall": 0.6}},
        }
        result = validate_policy_for_run(
            policy,
            metrics,
            test_classes={0},
            schema_classes={0},
            parent_metrics=parent_metrics,
        )
        assert result.passed is False
        assert len(result.regression_failures) > 0

    def test_regression_within_tolerance_passes(self) -> None:
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        metrics = {
            "mAP50": 0.68,
            "precision": 0.65,
            "recall": 0.55,
            "per_class": {0: {"precision": 0.65, "recall": 0.55}},
        }
        parent_metrics = {
            "mAP50": 0.7,
            "precision": 0.7,
            "recall": 0.6,
            "per_class": {0: {"precision": 0.7, "recall": 0.6}},
        }
        result = validate_policy_for_run(
            policy,
            metrics,
            test_classes={0},
            schema_classes={0},
            parent_metrics=parent_metrics,
        )
        assert result.passed is True
        assert len(result.regression_failures) == 0

    def test_below_absolute_thresholds_fails(self) -> None:
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        metrics = {
            "mAP50": 0.3,
            "precision": 0.2,
            "recall": 0.2,
            "per_class": {0: {"precision": 0.2, "recall": 0.2}},
        }
        result = validate_policy_for_run(
            policy,
            metrics,
            test_classes={0},
            schema_classes={0},
            parent_metrics=None,
        )
        assert result.passed is False


class TestValidateRegression:
    def test_no_regression(self) -> None:
        result = validate_regression(
            current={"mAP50": 0.7, "precision": 0.8, "recall": 0.7},
            parent={"mAP50": 0.6, "precision": 0.7, "recall": 0.6},
            max_regression_ratio=0.1,
        )
        assert len(result) == 0

    def test_regression_detected(self) -> None:
        result = validate_regression(
            current={"mAP50": 0.5, "precision": 0.6, "recall": 0.5},
            parent={"mAP50": 0.8, "precision": 0.8, "recall": 0.8},
            max_regression_ratio=0.1,
        )
        assert len(result) > 0

    def test_improvement_no_regression(self) -> None:
        result = validate_regression(
            current={"mAP50": 0.9, "precision": 0.9, "recall": 0.9},
            parent={"mAP50": 0.5, "precision": 0.5, "recall": 0.5},
            max_regression_ratio=0.1,
        )
        assert len(result) == 0


class TestValidatePublishCoverage:
    def test_all_classes_covered(self) -> None:
        result = validate_publish_coverage(
            schema_classes={0, 1, 2},
            test_classes={0, 1, 2},
        )
        assert result.passed is True
        assert len(result.missing_classes) == 0

    def test_missing_classes(self) -> None:
        result = validate_publish_coverage(
            schema_classes={0, 1, 2},
            test_classes={0, 2},
        )
        assert result.passed is False
        assert 1 in result.missing_classes

    def test_empty_schema_always_passes(self) -> None:
        result = validate_publish_coverage(
            schema_classes=set(),
            test_classes=set(),
        )
        assert result.passed is True
