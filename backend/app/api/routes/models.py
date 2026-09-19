from __future__ import annotations

import hashlib
import secrets
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_effective_settings, get_db, verify_api_key
from backend.app.observability.operation_log import log_operation
from backend.app.repositories.model_repository import (
    create_model_node,
    delete_model_node,
    get_model_node,
    list_child_models,
)
from backend.app.schemas.model import (
    ModelNodeCreate,
    ModelNodeListResponse,
    ModelNodeResponse,
    ModelNodeStatusUpdate,
)

_MAX_MODEL_BYTES = 500 * 1024 * 1024  # 500 MB

router = APIRouter(prefix="/api/models", tags=["models"])


@router.post("", response_model=ModelNodeResponse, status_code=status.HTTP_201_CREATED)
def create_model(
    payload: ModelNodeCreate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    # Multi-tree design: any number of root models is allowed. Each application
    # (rust segmentation, crack detection, bridge anomaly, ...) imports its own
    # pre-trained base model as a root and keeps its own lineage tree.
    try:
        model = create_model_node(
            db,
            task_type=payload.task_type,
            model_family=payload.model_family,
            artifact_path=payload.artifact_path,
            artifact_hash=payload.artifact_hash,
            name=payload.name,
            parent_id=payload.parent_id,
            label_schema_id=payload.label_schema_id,
            dataset_snapshot_id=payload.dataset_snapshot_id,
            training_attempt_id=payload.training_attempt_id,
            framework=payload.framework,
            artifact_format=payload.artifact_format,
            status=payload.status,
            metadata_json=payload.metadata_json,
        )
        log_operation(
            db,
            operation_type="model.create",
            resource_type="model_node",
            resource_id=model.id,
            status="success",
            summary_json={"model_id": str(model.id)},
        )
        return model
    except Exception as exc:
        log_operation(
            db,
            operation_type="model.create",
            status="error",
            error_summary=str(exc),
        )
        raise


@router.get("", response_model=ModelNodeListResponse)
def list_models(
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    from sqlalchemy import select as _select
    from backend.app.models import ModelNode as _ModelNode
    models = list(db.execute(_select(_ModelNode)).scalars().all())
    return ModelNodeListResponse(models=models, total=len(models))


@router.get("/{model_id}", response_model=ModelNodeResponse)
def get_model(
    model_id: str,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    # Try UUID first, then code lookup
    try:
        uuid_id = UUID(model_id)
        model = get_model_node(db, uuid_id)
    except ValueError:
        # Not a UUID, try code lookup
        from sqlalchemy import select as _select
        from backend.app.models import ModelNode as _ModelNode
        model = db.execute(
            _select(_ModelNode).where(_ModelNode.code == model_id)
        ).scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    return model


@router.patch("/{model_id}/status", response_model=ModelNodeResponse)
def update_model_status(
    model_id: str,
    payload: ModelNodeStatusUpdate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    # Try UUID first, then code lookup
    try:
        uuid_id = UUID(model_id)
        model = get_model_node(db, uuid_id)
    except ValueError:
        from sqlalchemy import select as _select
        from backend.app.models import ModelNode as _ModelNode
        model = db.execute(
            _select(_ModelNode).where(_ModelNode.code == model_id)
        ).scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    try:
        model.status = payload.status
        db.flush()
        log_operation(
            db,
            operation_type="model.update_status",
            resource_type="model_node",
            resource_id=model.id,
            status="success",
            summary_json={"model_id": str(model.id), "new_status": payload.status},
        )
        return model
    except Exception as exc:
        log_operation(
            db,
            operation_type="model.update_status",
            status="error",
            error_summary=str(exc),
        )
        raise


@router.post("/upload")
async def upload_model(
    file: UploadFile,
    _key: str = Depends(verify_api_key),
) -> Any:
    if not file.filename or not file.filename.endswith(".pt"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .pt files are allowed",
        )

    settings = get_effective_settings()
    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{secrets.token_hex(8)}_{file.filename}"
    dest = model_dir / safe_name

    sha256 = hashlib.sha256()
    total = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_MODEL_BYTES:
                    break
                f.write(chunk)
                sha256.update(chunk)
        if total > _MAX_MODEL_BYTES:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds {_MAX_MODEL_BYTES // (1024 * 1024)} MB limit",
            )
    except HTTPException:
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {exc}",
        )

    return {
        "filename": file.filename,
        "file_path": str(dest),
        "size": total,
        "sha256": f"sha256:{sha256.hexdigest()}",
    }


@router.delete("/{model_id}")
def delete_model(
    model_id: str,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    # Try UUID first, then code lookup
    try:
        uuid_id = UUID(model_id)
        model = get_model_node(db, uuid_id)
    except ValueError:
        from sqlalchemy import select as _select
        from backend.app.models import ModelNode as _ModelNode
        model = db.execute(
            _select(_ModelNode).where(_ModelNode.code == model_id)
        ).scalar_one_or_none()
        uuid_id = model.id if model else None
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    children = list_child_models(db, uuid_id)
    if children:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete model with children. Delete child models first.",
        )

    # Cascade delete dependent evaluations (FK RESTRICT would otherwise cause 500)
    try:
        from sqlalchemy import func as _func
        from sqlalchemy import select as _select  # local import to avoid cycle
        from sqlalchemy import text as _text
        from backend.app.models import BindingRelease as _BindingRelease
        from backend.app.models import Evaluation as _Evaluation
        from backend.app.models import RuntimeInstance as _RuntimeInstance

        # Task history stays, but its parent pointer is cleared so terminal task
        # rows don't block deleting a model that trained children (e.g. M002).
        # Core-level statement on purpose: `parent_model_node_id` is sealed by
        # TrainingTask's immutability guard, which rejects an ORM-level update.
        db.execute(
            _text(
                "UPDATE training_tasks SET parent_model_node_id = NULL "
                "WHERE parent_model_node_id = :dashed OR parent_model_node_id = :hex"
            ),
            # both forms: Postgres compares against the dashed text, SQLite stores
            # UUIDs as 32-char hex without dashes
            {"dashed": str(uuid_id), "hex": uuid_id.hex},
        )

        # A released model is torn down as part of the deletion: any binding whose
        # active release points at this model is 解绑下线'd (instances stopped,
        # release history dropped, binding row removed). Historical references in
        # other rows (superseded releases, stopped instances) are cleared; the
        # rows stay as history.
        from backend.app.models import ModelBinding as _ModelBinding
        from backend.app.runtime.runtime_instance import RuntimeInstanceManager

        manager = RuntimeInstanceManager()
        bindings_to_teardown = list(
            db.execute(
                _select(_ModelBinding)
                .join(_BindingRelease, _ModelBinding.current_release_id == _BindingRelease.id)
                .where(_BindingRelease.model_node_id == uuid_id)
            ).scalars().all()
        )
        for binding in bindings_to_teardown:
            # Clear the live pointers FIRST and flush: the DB enforces
            # fk_binding_current_release (declared in the DB, not in the ORM),
            # so the release rows cannot be deleted while the binding row still
            # points at them.
            binding.current_release_id = None
            binding.current_runtime_instance_id = None
            db.flush()
            binding_instances = list(
                db.execute(
                    _select(_RuntimeInstance).where(_RuntimeInstance.binding_id == binding.id)
                ).scalars().all()
            )
            for inst in binding_instances:
                if inst.status == "serving":
                    manager.transition_status(db, inst.id, "draining")
                    manager.transition_status(db, inst.id, "stopped")
            binding_releases = list(
                db.execute(
                    _select(_BindingRelease).where(_BindingRelease.binding_id == binding.id)
                ).scalars().all()
            )
            for release in binding_releases:
                db.delete(release)
            for inst in binding_instances:
                db.delete(inst)
            binding.current_release_id = None
            binding.current_runtime_instance_id = None
            db.delete(binding)

        # Flush the teardown deletes BEFORE the text updates below: leaving them
        # pending makes the autoflush order run the UPDATE first, which trips
        # the FK on the not-yet-deleted rows (observed, see test_delete_model_*).
        db.flush()

        db.execute(
            _text(
                "UPDATE model_binding_releases SET model_node_id = NULL "
                "WHERE model_node_id = :dashed OR model_node_id = :hex"
            ),
            {"dashed": str(uuid_id), "hex": uuid_id.hex},
        )
        db.execute(
            _text(
                "UPDATE runtime_instances SET model_node_id = NULL "
                "WHERE model_node_id = :dashed OR model_node_id = :hex"
            ),
            {"dashed": str(uuid_id), "hex": uuid_id.hex},
        )

        # GPU residency plans are meaningless without their model - cascade them.
        from backend.app.models import ModelResidencyPlan as _ResidencyPlan

        plans = list(
            db.execute(
                _select(_ResidencyPlan).where(_ResidencyPlan.model_node_id == uuid_id)
            ).scalars().all()
        )
        for plan in plans:
            db.delete(plan)

        evals = list(db.execute(_select(_Evaluation).where(_Evaluation.model_node_id == uuid_id)).scalars().all())
        for ev in evals:
            db.delete(ev)
        if evals:
            db.flush()

        delete_model_node(db, uuid_id)
    except HTTPException:
        raise
    except Exception as exc:
        # Translate FK violations into a 400 that names the offending table
        err_msg = str(exc)
        if "ForeignKeyViolation" in err_msg or "violates foreign key" in err_msg:
            import re as _re

            table = "unknown"
            match = _re.search(r'referenced from table "(\w+)"', err_msg)
            if match:
                table = match.group(1)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete model: still referenced by table '{table}'.",
            ) from exc
        raise
    log_operation(
        db,
        operation_type="model.delete",
        resource_type="model_node",
        resource_id=uuid_id,
        status="success",
        summary_json={"model_id": str(uuid_id)},
    )
    return {"deleted": True}


@router.post("/{model_id}/infer")
async def infer_with_model(
    model_id: str,
    file: UploadFile,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    # Try UUID first, then code lookup
    try:
        uuid_id = UUID(model_id)
        model = get_model_node(db, uuid_id)
    except ValueError:
        from sqlalchemy import select as _select
        from backend.app.models import ModelNode as _ModelNode
        model = db.execute(
            _select(_ModelNode).where(_ModelNode.code == model_id)
        ).scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    import time
    start = time.time()

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image")

    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_width, image_height = img.size
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image: {exc}")

    try:
        from ultralytics import YOLO
        import base64

        yolo = YOLO(model.artifact_path)
        results = yolo(img, verbose=False)

        detections = []
        if results and len(results) > 0:
            r = results[0]
            class_names = dict(r.names) if hasattr(r, "names") and r.names else {}
            # Segmentation models carry instance masks alongside the boxes; expose each mask
            # as a polygon (list of [x, y] points in original-image coordinates) so API
            # consumers can draw or measure the actual defect region, not just its box.
            mask_polys = None
            if hasattr(r, "masks") and r.masks is not None and hasattr(r.masks, "xy"):
                mask_polys = [poly.tolist() for poly in r.masks.xy]
            if hasattr(r, "boxes") and r.boxes is not None:
                for i in range(len(r.boxes.xyxy)):
                    box = r.boxes.xyxy[i].tolist()
                    conf = float(r.boxes.conf[i])
                    cls_id = int(r.boxes.cls[i])
                    det = {
                        "class_name": class_names.get(cls_id, str(cls_id)),
                        "confidence": round(conf, 4),
                        "bbox": [round(v, 2) for v in box],
                        "class_id": cls_id,
                    }
                    if mask_polys is not None and i < len(mask_polys):
                        det["mask"] = [
                            [round(px, 2), round(py, 2)] for px, py in mask_polys[i]
                        ]
                    detections.append(det)

        # Generate overlay image: semi-transparent mask fill + contour for segmentation
        # models, plain boxes for detection models.
        overlay_base64 = None
        try:
            import cv2
            import numpy as np

            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            for det in detections:
                label = f"{det['class_name']} {det['confidence']*100:.1f}%"
                if det.get("mask"):
                    pts = np.array([[int(px), int(py)] for px, py in det["mask"]], np.int32)
                    if len(pts) >= 3:
                        fill = img_cv.copy()
                        cv2.fillPoly(fill, [pts], (0, 224, 255))
                        img_cv = cv2.addWeighted(fill, 0.45, img_cv, 0.55, 0)
                        cv2.polylines(img_cv, [pts], True, (0, 224, 255), 2)
                    top_y = int(min(py for _, py in det["mask"]))
                    left_x = int(min(px for px, _ in det["mask"]))
                    cv2.putText(img_cv, label, (left_x, max(10, top_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 224, 255), 1)
                else:
                    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 102, 173), 2)
                    cv2.putText(img_cv, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 102, 173), 1)
            _, buffer = cv2.imencode(".jpg", img_cv, [cv2.IMWRITE_JPEG_QUALITY, 85])
            overlay_base64 = base64.b64encode(buffer).decode("utf-8")
        except Exception:
            pass

        latency = round((time.time() - start) * 1000, 1)
        return {
            "model_code": model.code,
            "model_name": model.name,
            "detections": detections,
            "latency_ms": latency,
            "image_width": image_width,
            "image_height": image_height,
            "overlay_image": overlay_base64,
        }
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {exc}")
