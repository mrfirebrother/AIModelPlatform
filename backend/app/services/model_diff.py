from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import Evaluation, LabelSchemaClass, ModelNode


@dataclass
class DiffItem:
    field: str
    old_value: Any
    new_value: Any
    change: str = "changed"


@dataclass
class DiffReport:
    base_model_id: UUID
    compare_model_id: UUID
    is_identical: bool
    config_diffs: list[DiffItem] = field(default_factory=list)
    metrics_diffs: list[DiffItem] = field(default_factory=list)
    label_diffs: list[DiffItem] = field(default_factory=list)
    summary: str = ""


_CONFIG_FIELDS = [
    "task_type",
    "model_family",
    "framework",
    "artifact_format",
    "status",
]


def _load_model(session: Session, model_id: UUID) -> ModelNode:
    model = session.get(ModelNode, model_id)
    if model is None:
        raise ValueError(f"Model node {model_id} not found")
    return model


def _load_label_classes(session: Session, schema_id: UUID | None) -> dict[int, str]:
    if schema_id is None:
        return {}
    rows = session.execute(
        select(LabelSchemaClass).where(LabelSchemaClass.schema_id == schema_id)
    ).scalars().all()
    return {c.class_id: c.semantic_key for c in rows}


def _load_latest_metrics(session: Session, model_node_id: UUID) -> dict[str, Any]:
    ev = session.execute(
        select(Evaluation)
        .where(Evaluation.model_node_id == model_node_id)
        .where(Evaluation.auto_status == "passed")
        .order_by(Evaluation.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if ev is None:
        return {}
    return dict(ev.auto_metrics_json) if ev.auto_metrics_json else {}


def diff_models(session: Session, base_id: UUID, compare_id: UUID) -> DiffReport:
    base = _load_model(session, base_id)
    compare = _load_model(session, compare_id)

    config_diffs: list[DiffItem] = []
    for fname in _CONFIG_FIELDS:
        old_val = getattr(base, fname)
        new_val = getattr(compare, fname)
        if old_val != new_val:
            config_diffs.append(DiffItem(field=fname, old_value=old_val, new_value=new_val))

    base_labels = _load_label_classes(session, base.label_schema_id)
    compare_labels = _load_label_classes(session, compare.label_schema_id)
    label_diffs: list[DiffItem] = []
    all_keys = set(base_labels.keys()) | set(compare_labels.keys())
    for cid in sorted(all_keys):
        old_key = base_labels.get(cid)
        new_key = compare_labels.get(cid)
        if old_key is None:
            label_diffs.append(DiffItem(field=f"class_{cid}", old_value=None, new_value=new_key, change="added"))
        elif new_key is None:
            label_diffs.append(DiffItem(field=f"class_{cid}", old_value=old_key, new_value=None, change="removed"))
        elif old_key != new_key:
            label_diffs.append(DiffItem(field=f"class_{cid}", old_value=old_key, new_value=new_key, change="renamed"))

    base_metrics = _load_latest_metrics(session, base_id)
    compare_metrics = _load_latest_metrics(session, compare_id)
    metrics_diffs: list[DiffItem] = []
    all_metric_keys = set(base_metrics.keys()) | set(compare_metrics.keys())
    for mk in sorted(all_metric_keys):
        old_val = base_metrics.get(mk)
        new_val = compare_metrics.get(mk)
        if old_val is None:
            metrics_diffs.append(DiffItem(field=mk, old_value=None, new_value=new_val, change="added"))
        elif new_val is None:
            metrics_diffs.append(DiffItem(field=mk, old_value=old_val, new_value=None, change="removed"))
        elif old_val != new_val:
            metrics_diffs.append(DiffItem(field=mk, old_value=old_val, new_value=new_val, change="changed"))

    is_identical = not config_diffs and not label_diffs and not metrics_diffs
    summary_parts: list[str] = []
    if config_diffs:
        summary_parts.append(f"{len(config_diffs)} config field(s) differ")
    if label_diffs:
        summary_parts.append(f"{len(label_diffs)} label class(es) differ")
    if metrics_diffs:
        summary_parts.append(f"{len(metrics_diffs)} metric(s) differ")
    summary = "; ".join(summary_parts) if summary_parts else "Models are identical"

    return DiffReport(
        base_model_id=base_id,
        compare_model_id=compare_id,
        is_identical=is_identical,
        config_diffs=config_diffs,
        metrics_diffs=metrics_diffs,
        label_diffs=label_diffs,
        summary=summary,
    )


def get_model_history(session: Session, model_id: UUID) -> list[ModelNode]:
    _load_model(session, model_id)
    chain: list[ModelNode] = []
    current_id: UUID | None = model_id
    visited: set[UUID] = set()
    while current_id is not None and current_id not in visited:
        visited.add(current_id)
        model = session.get(ModelNode, current_id)
        if model is None:
            break
        chain.append(model)
        current_id = model.parent_id
    return chain
