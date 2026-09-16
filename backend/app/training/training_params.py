"""The training parameters this platform exposes, and the rules around them.

Three deliberate design rules (see AGENTS.md for the measurements behind them):

* **Small surface.**  Only parameters that measurably matter on this box are accepted:
  ``epochs`` / ``imgsz`` / ``batch`` / ``patience`` / ``cache`` / the augmentation level /
  the pre-trained parent.  Everything an implementer cannot reason about stays out of the
  UI *and* out of the API, because a mis-set knob produces a worse model and a support
  ticket - not a better one.
* **Named levels, not coefficient soup.**  Augmentation is exposed as
  ``off`` / ``default`` / ``strong``.  The ~15 raw ultralytics coefficients behind each
  level live here, so the UI and the trainer can never drift apart.
* **Fail loudly and early.**  A combination that cannot fit the 4 GB card is rejected when
  the task is created, with a message saying what to change - instead of a CUDA OOM forty
  minutes into training.

Does a task omit a parameter?  Then it is *absent* here and filled from ``TRAINING_DEFAULT_*``
by ``train_worker._build_trainer_config`` - never guessed silently by ultralytics.
"""

from __future__ import annotations

from typing import Any

#: Augmentation level -> the exact ultralytics arguments it expands to.
#: "default" reproduces ultralytics' own defaults (written out on purpose so that a
#: comparison run has a documented baseline instead of "whatever the library changed").
AUGMENTATION_PRESETS: dict[str, dict[str, Any]] = {
    "off": {
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
        "scale": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    },
    "default": {
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 10,
        "scale": 0.5,
        "degrees": 0.0,
        "translate": 0.1,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
    },
    "strong": {
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.1,
        "close_mosaic": 20,
        "scale": 0.9,
        "degrees": 10.0,
        "translate": 0.2,
        "shear": 2.0,
        "perspective": 0.0005,
        "fliplr": 0.5,
        "flipud": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.9,
        "hsv_v": 0.5,
    },
}

AUGMENTATION_LABELS: dict[str, str] = {
    "off": "关闭增广",
    "default": "默认（库的基线）",
    "strong": "强增广",
}

DEFAULT_AUGMENTATION = "default"

#: Parameters accepted from a task's ``training_config_json``, with hard inclusive bounds.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "epochs": (1, 1000),
    "imgsz": (256, 1280),
    "batch": (1, 64),
    "patience": (0, 1000),
}

#: Raw ultralytics keys an API caller may still set explicitly (each one bounded).
#: The UI does not send these; they exist so scripts can run a one-variable experiment
#: without waiting for a UI change.
RAW_ARG_BOUNDS: dict[str, tuple[float, float]] = {
    "mosaic": (0.0, 1.0),
    "mixup": (0.0, 1.0),
    "copy_paste": (0.0, 1.0),
    "close_mosaic": (0, 1000),
    "scale": (0.0, 1.0),
    "degrees": (0.0, 180.0),
    "translate": (0.0, 1.0),
    "shear": (0.0, 45.0),
    "perspective": (0.0, 0.001),
    "fliplr": (0.0, 1.0),
    "flipud": (0.0, 1.0),
    "hsv_h": (0.0, 1.0),
    "hsv_s": (0.0, 1.0),
    "hsv_v": (0.0, 1.0),
    "lr0": (1e-5, 0.1),
    "lrf": (0.0, 1.0),
    "momentum": (0.0, 1.0),
    "weight_decay": (0.0, 0.01),
    "warmup_epochs": (0.0, 10.0),
    "dropout": (0.0, 0.9),
    "label_smoothing": (0.0, 0.5),
}

#: Arguments the platform owns.  A task may echo the locked value back, but not change it.
LOCKED_ARGS: dict[str, Any] = {
    "workers": 0,
    "amp": False,
}

LOCKED_REASONS: dict[str, str] = {
    "workers": "Windows + Celery(solo) 下 DataLoader 起子进程会让 worker 崩溃，固定为 0",
    "amp": "GTX 745 是 Maxwell(sm_50)，不支持混合精度，固定为 False",
}

#: Measured on this box: imgsz 416 + batch 8 -> 1.28 GB peak (ultralytics' own GPU_mem column).
#: Peak memory tracks pixels-per-image x batch, so this single point extrapolates the rest.
_VRAM_MB_PER_IMAGE_AT_416 = 160.0


def estimate_train_vram_mb(imgsz: float, batch: float) -> float:
    """Rough peak training VRAM in MiB for one (imgsz, batch) pair."""
    return _VRAM_MB_PER_IMAGE_AT_416 * (float(imgsz) / 416.0) ** 2 * float(batch)


def expand_augmentation(level: str | None) -> dict[str, Any]:
    """Return the ultralytics arguments for an augmentation level.

    An unknown level falls back to the default preset (the worker normalizes task
    payloads, so this only guards against hand-written API calls).
    """
    return dict(AUGMENTATION_PRESETS.get(str(level or DEFAULT_AUGMENTATION), AUGMENTATION_PRESETS[DEFAULT_AUGMENTATION]))


def defaults_from_settings(settings: Any) -> dict[str, Any]:
    """The ``TRAINING_DEFAULT_*`` values as a dict keyed like a task's config."""
    return {
        "epochs": settings.training_default_epochs,
        "batch": settings.training_default_batch_size,
        "imgsz": settings.training_default_imgsz,
        "cache": settings.training_default_cache,
        "patience": settings.training_default_patience,
        "augmentation": settings.training_default_augmentation,
    }


def resolve_config(
    config: dict[str, Any] | None,
    *,
    defaults: dict[str, Any] | None = None,
    vram_limit_mb: float | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Fill every absent parameter from ``defaults``, then normalize.

    Called by **both** the create route and the worker:

    * the route stores the result as the task's ``training_config_json``, which is immutable
      from the moment the row exists (`install_immutable_guard(TrainingTask, …)`) — the
      complete parameter snapshot therefore has to be written at creation, not back-filled by
      the worker (attempting that raises `ImmutableFieldError: … immutable after sealing`);
    * the worker re-resolves defensively, for tasks created before a rule existed.
    """
    merged: dict[str, Any] = dict(config or {})
    for key, value in (defaults or {}).items():
        if merged.get(key) is None:
            merged[key] = value
    return normalize_training_config(merged, vram_limit_mb=vram_limit_mb)


def _check_bounds(name: str, value: Any, bounds: dict[str, tuple[float, float]]) -> float:
    low, high = bounds[name]
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"参数 {name} 必须是数字，收到 {value!r}") from None
    if not (low <= number <= high):
        raise ValueError(f"参数 {name}={value!r} 超出允许范围 [{_fmt(low)}, {_fmt(high)}]")
    return number


def _fmt(number: float) -> str:
    return str(int(number)) if float(number).is_integer() else str(number)


def normalize_training_config(
    config: dict[str, Any] | None,
    *,
    vram_limit_mb: float | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate and normalize a task's training config.

    Returns ``(normalized_config, warnings)`` and raises :class:`ValueError` with a
    user-facing Chinese message when a value cannot be trained at all.
    """
    normalized: dict[str, Any] = dict(config or {})
    warnings: list[str] = []

    if "augmentation" in normalized and normalized["augmentation"] is not None:
        level = str(normalized["augmentation"])
        if level not in AUGMENTATION_PRESETS:
            options = " / ".join(sorted(AUGMENTATION_PRESETS))
            raise ValueError(f"增广档位 {level!r} 不存在，可选：{options}")
        normalized["augmentation"] = level

    for name in PARAM_BOUNDS:
        if name in normalized and normalized[name] is not None:
            number = _check_bounds(name, normalized[name], PARAM_BOUNDS)
            normalized[name] = int(number) if number == int(number) else number

    if "imgsz" in normalized and normalized["imgsz"] is not None:
        imgsz = int(normalized["imgsz"])
        if imgsz % 32 != 0:
            raise ValueError(f"imgsz={imgsz} 必须是 32 的倍数（推荐 320/416/512/640/768）")

    for name in RAW_ARG_BOUNDS:
        if name in normalized and normalized[name] is not None:
            number = _check_bounds(name, normalized[name], RAW_ARG_BOUNDS)
            normalized[name] = int(number) if name == "close_mosaic" else number

    for name, locked in LOCKED_ARGS.items():
        if name in normalized and normalized[name] is not None and normalized[name] != locked:
            raise ValueError(f"参数 {name} 不可修改（{LOCKED_REASONS.get(name, '')}），固定为 {locked!r}")

    if vram_limit_mb and normalized.get("imgsz") and normalized.get("batch"):
        needed = estimate_train_vram_mb(normalized["imgsz"], normalized["batch"])
        if needed > vram_limit_mb:
            suggested = max(1, int(normalized["batch"] * vram_limit_mb / needed))
            raise ValueError(
                f"imgsz={int(normalized['imgsz'])} + batch={int(normalized['batch'])} 预计占用 "
                f"{needed:.0f} MB 显存，超过本机可用的 {vram_limit_mb:.0f} MB。"
                f"请把 batch 降到 {suggested} 或把分辨率调小"
            )
        if needed > vram_limit_mb * 0.75:
            warnings.append(f"预计显存占用 {needed:.0f} MB / {vram_limit_mb:.0f} MB，已接近上限，留意 OOM")

    return normalized, warnings
