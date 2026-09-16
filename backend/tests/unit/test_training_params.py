"""Unit tests for the platform's training-parameter surface (training_params.py)."""

import pytest

from backend.app.training.training_params import (
    AUGMENTATION_PRESETS,
    defaults_from_settings,
    estimate_train_vram_mb,
    expand_augmentation,
    normalize_training_config,
    resolve_config,
)


class TestAugmentationPresets:
    def test_every_level_defines_the_same_keys(self) -> None:
        key_sets = {frozenset(preset) for preset in AUGMENTATION_PRESETS.values()}
        assert len(key_sets) == 1, "levels must stay comparable: identical key sets"

    def test_off_disables_every_coefficient(self) -> None:
        off = expand_augmentation("off")
        assert off["mosaic"] == 0.0
        assert off["fliplr"] == 0.0
        assert off["close_mosaic"] == 0

    def test_unknown_level_falls_back_to_default(self) -> None:
        assert expand_augmentation("nope") == expand_augmentation("default")
        assert expand_augmentation(None) == expand_augmentation("default")

    def test_expand_returns_a_copy(self) -> None:
        mutated = expand_augmentation("default")
        mutated["mosaic"] = 99.0
        assert expand_augmentation("default")["mosaic"] == 1.0


class TestNormalizeTrainingConfig:
    def test_empty_config_is_accepted(self) -> None:
        normalized, warnings = normalize_training_config(None)
        assert normalized == {}
        assert warnings == []

    def test_unknown_augmentation_level_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="增广档位"):
            normalize_training_config({"augmentation": "extreme"})

    def test_known_augmentation_level_is_kept(self) -> None:
        normalized, _ = normalize_training_config({"augmentation": "strong"})
        assert normalized["augmentation"] == "strong"

    def test_imgsz_must_be_a_multiple_of_32(self) -> None:
        with pytest.raises(ValueError, match="32 的倍数"):
            normalize_training_config({"imgsz": 500})

    def test_out_of_range_values_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="超出允许范围"):
            normalize_training_config({"batch": 999})

    def test_locked_arguments_cannot_be_changed(self) -> None:
        with pytest.raises(ValueError, match="不可修改"):
            normalize_training_config({"workers": 4})
        with pytest.raises(ValueError, match="不可修改"):
            normalize_training_config({"amp": True})

    def test_echoing_a_locked_value_is_allowed(self) -> None:
        normalized, _ = normalize_training_config({"workers": 0, "amp": False})
        assert normalized["workers"] == 0

    def test_raw_coefficients_pass_through_and_are_bounded(self) -> None:
        normalized, _ = normalize_training_config({"mosaic": 0.25})
        assert normalized["mosaic"] == 0.25
        with pytest.raises(ValueError, match="mosaic"):
            normalize_training_config({"mosaic": 3})

    def test_vram_guard_rejects_an_impossible_combination(self) -> None:
        with pytest.raises(ValueError, match="batch 降到"):
            normalize_training_config({"imgsz": 1280, "batch": 16}, vram_limit_mb=3300)

    def test_vram_guard_allows_the_measured_combinations(self) -> None:
        for imgsz, batch in ((416, 8), (640, 4)):
            normalized, warnings = normalize_training_config(
                {"imgsz": imgsz, "batch": batch}, vram_limit_mb=3300
            )
            assert normalized["imgsz"] == imgsz
            assert warnings == []

    def test_vram_warning_when_close_to_the_limit(self) -> None:
        _, warnings = normalize_training_config({"imgsz": 640, "batch": 8}, vram_limit_mb=3300)
        assert warnings and "接近上限" in warnings[0]

    def test_estimate_scales_with_pixels_and_batch(self) -> None:
        assert estimate_train_vram_mb(416, 8) == pytest.approx(1280)
        assert estimate_train_vram_mb(416, 4) == pytest.approx(640)
        # Doubling the side quadruples the pixels, and pixels x batch is the whole model.
        assert estimate_train_vram_mb(832, 8) == pytest.approx(4 * 1280)
        assert estimate_train_vram_mb(832, 4) == pytest.approx(2 * 1280)


class TestResolveConfig:
    """The route and the worker share this, because the task row is immutable after insert."""

    DEFAULTS = {
        "epochs": 100,
        "batch": 8,
        "imgsz": 416,
        "cache": False,
        "patience": 30,
        "augmentation": "default",
    }

    def test_absent_parameters_are_filled_from_defaults(self) -> None:
        resolved, warnings = resolve_config({"epochs": 7}, defaults=dict(self.DEFAULTS))
        assert resolved["epochs"] == 7  # explicit value wins
        assert resolved["imgsz"] == 416
        assert resolved["batch"] == 8
        assert resolved["cache"] is False
        assert resolved["augmentation"] == "default"
        assert warnings == []

    def test_false_and_zero_are_not_treated_as_absent(self) -> None:
        resolved, _ = resolve_config(
            {"cache": False, "patience": 0},
            defaults={**self.DEFAULTS, "cache": True, "patience": 30},
        )
        assert resolved["cache"] is False
        assert resolved["patience"] == 0

    def test_defaults_are_validated_too(self) -> None:
        with pytest.raises(ValueError, match="超出允许范围"):
            resolve_config({}, defaults={"batch": 0})

    def test_vram_guard_applies_after_the_merge(self) -> None:
        with pytest.raises(ValueError, match="batch 降到"):
            resolve_config({}, defaults={"imgsz": 1280, "batch": 16}, vram_limit_mb=3300)


class TestDefaultsFromSettings:
    def test_maps_every_env_backed_default(self) -> None:
        from backend.app.config import Settings

        settings = Settings(
            TRAINING_DEFAULT_EPOCHS=42,
            TRAINING_DEFAULT_BATCH_SIZE=4,
            TRAINING_DEFAULT_IMGSZ=640,
            TRAINING_DEFAULT_CACHE=True,
            TRAINING_DEFAULT_PATIENCE=7,
            TRAINING_DEFAULT_AUGMENTATION="strong",
        )
        assert defaults_from_settings(settings) == {
            "epochs": 42,
            "batch": 4,
            "imgsz": 640,
            "cache": True,
            "patience": 7,
            "augmentation": "strong",
        }
