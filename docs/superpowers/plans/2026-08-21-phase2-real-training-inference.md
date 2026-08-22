# AI Model Platform Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace placeholder training/inference with real Ultralytics YOLO execution, enable GPU model loading/unloading, and validate the complete training-evaluation-publish-inference loop with real models.

**Architecture:** Phase 1 infrastructure and database are complete. Phase 2 replaces placeholder workers with real Ultralytics YOLO training and inference, adds GPU model lifecycle management, and validates the end-to-end flow with actual `.pt` models.

**Tech Stack:** PyTorch, Ultralytics YOLO, ONNX Runtime (optional), CUDA, NVIDIA Container Toolkit.

**Scope:** Real YOLO detection training and inference on single GPU. No video, no multi-GPU, no annotation tools, no ONNX export (optional).

---

## Current State

Phase 1 completed all 10 tasks:

```text
Task 1:  FastAPI skeleton, config, health checks, Docker Compose
Task 2:  PostgreSQL ORM, migrations, constraints
Task 3:  Dataset import, YOLO validation, immutable snapshots
Task 4:  Model nodes, label schema evolution, binding releases
Task 5:  Training tasks, checkpoints, GPU resource leases
Task 6:  Automatic evaluation, human review records
Task 7:  Runtime instances, publish/rollback, reconciliation
Task 8:  FastAPI management and inference API
Task 9:  React admin console
Task 10: E2E tests, Docker Compose, documentation
```

Placeholder workers exist at:
- `backend/app/workers/train_worker.py` (Celery placeholder)
- `backend/app/workers/inference_worker.py` (Celery placeholder)
- `backend/app/runtime/model_manager.py` (GPU management placeholder)

---

## Task 1: Real YOLO Training Engine

**Files:**
- Modify: `backend/app/workers/train_worker.py`
- Create: `backend/app/training/yolo_trainer.py`
- Create: `backend/app/training/yolo_dataset.py`
- Create: `backend/app/training/checkpoint_manager.py`
- Create: `backend/tests/unit/test_yolo_trainer.py`
- Create: `backend/tests/integration/test_real_training.py`

- [ ] **Step 1: Write failing tests for YOLO training**

```python
def test_yolo_trainer_creates_model_node():
    trainer = YoloTrainer(config)
    result = trainer.train(dataset_snapshot, parent_model)
    assert result.model_node_id is not None
    assert result.status == "completed"
```

- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement YoloTrainer**
  - Accept training config: epochs, batch_size, learning_rate, image_size
  - Load parent model weights (or YOLO pretrained if no parent)
  - Create YOLO dataset format from snapshot manifests
  - Run `model.train()` with Ultralytics API
  - Save checkpoint at configured intervals
  - Save best model weights
  - Generate training metrics (loss curves, mAP per epoch)
  - Create candidate ModelNode on success
- [ ] **Step 4: Implement YoloDataset adapter**
  - Convert snapshot manifest to YOLO dataset format
  - Create temporary train/val/test directory structure
  - Generate `data.yaml` with correct paths and class names
  - Clean up temporary files after training
- [ ] **Step 5: Implement CheckpointManager**
  - Save checkpoint every N epochs
  - Track latest and best checkpoint
  - Record parent artifact hash, config hash, epoch
  - Support resume from checkpoint
- [ ] **Step 6: Run tests and verify**
- [ ] **Step 7: Commit**

---

## Task 2: Real YOLO Inference Engine

**Files:**
- Modify: `backend/app/workers/inference_worker.py`
- Create: `backend/app/inference/yolo_engine.py`
- Create: `backend/app/inference/preprocessor.py`
- Create: `backend/app/inference/postprocessor.py`
- Create: `backend/tests/unit/test_yolo_engine.py`
- Create: `backend/tests/integration/test_inference_flow.py`

- [ ] **Step 1: Write failing tests for inference**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement YoloEngine**
  - Load `.pt` model from artifact path
  - Run inference on input image
  - Return detection results: class, confidence, bbox
  - Support model caching (loaded models stay in memory)
  - Support model eviction on demand
- [ ] **Step 4: Implement Preprocessor**
  - Resize/normalize input image
  - Handle different input formats (PIL, numpy, bytes)
- [ ] **Step 5: Implement Postprocessor**
  - NMS (Non-Maximum Suppression)
  - Confidence threshold filtering
  - Class name mapping from label schema
- [ ] **Step 6: Run tests and verify**
- [ ] **Step 7: Commit**

---

## Task 3: GPU Model Lifecycle Management

**Files:**
- Modify: `backend/app/runtime/model_manager.py`
- Create: `backend/app/runtime/gpu_monitor.py`
- Create: `backend/tests/unit/test_gpu_monitor.py`
- Create: `backend/tests/integration/test_model_lifecycle.py`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement GPU Monitor**
  - Query GPU memory usage via `torch.cuda`
  - Report total, used, free memory
  - Track which models are loaded and their memory footprint
- [ ] **Step 4: Implement ModelManager (real)**
  - Load model to GPU: allocate memory, load weights
  - Unload model from GPU: free memory
  - Evict least-recently-used model when memory full
  - Track loaded models and their GPU allocation
  - Integrate with residency plan
- [ ] **Step 5: Run tests and verify**
- [ ] **Step 6: Commit**

---

## Task 4: Training Task Queue Integration

**Files:**
- Modify: `backend/app/workers/train_worker.py`
- Modify: `backend/app/workers/celery_app.py`
- Create: `backend/app/training/resource_scheduler.py`
- Create: `backend/tests/unit/test_resource_scheduler.py`
- Create: `backend/tests/integration/test_training_queue.py`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement ResourceScheduler**
  - Check GPU memory before starting training
  - Reserve GPU memory for training
  - Release memory on completion/failure
  - Prevent concurrent training on same GPU
- [ ] **Step 4: Update train_worker**
  - Replace placeholder with real YoloTrainer call
  - Acquire GPU lease before training
  - Release lease after training
  - Save checkpoint on failure
  - Create candidate model on success
- [ ] **Step 5: Run tests and verify**
- [ ] **Step 6: Commit**

---

## Task 5: Evaluation Worker Integration

**Files:**
- Modify: `backend/app/workers/evaluation_worker.py`
- Create: `backend/app/evaluation/yolo_evaluator.py`
- Create: `backend/tests/unit/test_yolo_evaluator.py`
- Create: `backend/tests/integration/test_evaluation_integration.py`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement YoloEvaluator**
  - Load model and run inference on test set
  - Compute per-image detection results
  - Match predictions to ground truth
  - Compute precision, recall, mAP50, mAP50-95
  - Generate evaluation report with image references
- [ ] **Step 4: Update evaluation_worker**
  - Replace placeholder with real YoloEvaluator call
  - Acquire GPU lease for evaluation
  - Save evaluation metrics to database
  - Save report file
  - Release lease after evaluation
- [ ] **Step 5: Run tests and verify**
- [ ] **Step 6: Commit**

---

## Task 6: Docker GPU Configuration

**Files:**
- Modify: `infra/docker-compose.yml`
- Modify: `backend/Dockerfile`
- Modify: `infra/.env.example`
- Create: `infra/Dockerfile.gpu`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Create GPU-enabled Dockerfile**
  - Use `nvidia/cuda:12.x-runtime` as base
  - Install Python, PyTorch with CUDA support
  - Install Ultralytics
  - Copy application code
- [ ] **Step 4: Update docker-compose.yml**
  - Add GPU resource reservations for worker and inference
  - Add NVIDIA runtime configuration
  - Add GPU environment variables
  - Configure CUDA_VISIBLE_DEVICES
- [ ] **Step 5: Update .env.example**
  - Add GPU_DEVICE, GPU_MEMORY_LIMIT, CUDA_VISIBLE_DEVICES
- [ ] **Step 6: Run tests and verify**
- [ ] **Step 7: Commit**

---

## Task 7: Real Model Import and Validation

**Files:**
- Modify: `backend/app/services/model_import.py`
- Create: `backend/app/training/yolo_validator.py`
- Create: `backend/tests/unit/test_yolo_validator.py`
- Create: `backend/tests/integration/test_model_import_real.py`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement YoloValidator**
  - Load `.pt` file with Ultralytics
  - Verify model can run inference on dummy image
  - Extract model metadata: input size, class count, class names
  - Validate label schema consistency
- [ ] **Step 4: Update model_import**
  - Replace placeholder validation with real YoloValidator
  - Save model metadata to database
  - Create model artifact with checksum
- [ ] **Step 5: Run tests and verify**
- [ ] **Step 6: Commit**

---

## Task 8: End-to-End Real Model Loop

**Files:**
- Modify: `backend/tests/e2e/test_closed_loop.py`
- Create: `backend/tests/fixtures/build_real_fixtures.py`
- Create: `backend/tests/fixtures/real_yolo_dataset/`

- [ ] **Step 1: Build real test fixtures**
  - Small YOLO dataset (10-20 images, 2 classes)
  - Pre-trained YOLOv8n model
  - Expected metrics thresholds
- [ ] **Step 2: Write E2E test**
  - Import root YOLO model
  - Import real dataset
  - Create dataset snapshot
  - Create training task
  - Run real training (small epochs)
  - Verify candidate model created
  - Run real evaluation
  - Verify metrics computed
  - Publish model
  - Run real inference
  - Verify detection results
  - Rollback to previous model
  - Verify rollback works
- [ ] **Step 3: Run test and verify**
- [ ] **Step 4: Commit**

---

## Task 9: GPU Monitoring Dashboard

**Files:**
- Modify: `frontend/src/features/resources/ResourcesPage.tsx`
- Modify: `frontend/src/features/dashboard/DashboardPage.tsx`
- Create: `backend/app/api/routes/gpu.py`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement GPU monitoring API**
  - GET /api/gpu/status: memory usage, temperature, utilization
  - GET /api/gpu/models: loaded models and their memory
  - POST /api/gpu/load: load model to GPU
  - POST /api/gpu/unload: unload model from GPU
- [ ] **Step 4: Update frontend**
  - Real-time GPU memory display
  - Loaded models list
  - Load/unload model buttons
  - GPU temperature and utilization
- [ ] **Step 5: Run tests and verify**
- [ ] **Step 6: Commit**

---

## Task 10: Training Monitoring and Logs

**Files:**
- Modify: `frontend/src/features/training/TrainingPage.tsx`
- Create: `backend/app/api/routes/training_logs.py`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement training log API**
  - GET /api/training/{id}/logs: real-time training logs
  - GET /api/training/{id}/metrics: loss curves, mAP per epoch
  - GET /api/training/{id}/checkpoint: checkpoint info
- [ ] **Step 4: Update frontend**
  - Training progress with loss chart
  - Real-time log viewer
  - Checkpoint status
  - Epoch progress bar
- [ ] **Step 5: Run tests and verify**
- [ ] **Step 6: Commit**

---

## Execution Notes

- Tasks 1-3 are core and must be completed first
- Tasks 4-5 depend on Tasks 1-3
- Task 6 is independent and can be done in parallel
- Tasks 7-10 depend on Tasks 1-5
- GPU tests should be skipped when CUDA is not available
- Use small YOLO models (yolov8n) for testing
- Keep training epochs low in tests (5-10 epochs)
- All files must have UTF-8 BOM
- Run `python -m pytest -c NUL` for test execution
