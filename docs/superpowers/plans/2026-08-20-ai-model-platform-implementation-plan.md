# AI Model Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first usable single-GPU AI model platform loop: import root PT models, create immutable image dataset snapshots, manually train YOLO detection models, evaluate candidates, and publish or roll back models through stable bindings.

**Architecture:** A Python/FastAPI platform service owns PostgreSQL metadata and orchestration. Celery/Redis runs one training attempt at a time; a separate inference Worker owns PT runtime instances and communicates with FastAPI over an internal HTTP interface. React/TypeScript is a separate admin console. Models, dataset snapshots, checkpoints, reports, and logs are stored in platform-managed local files.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, PostgreSQL, Redis, Celery, PyTorch, Ultralytics YOLO, React, TypeScript, Vite, Docker Compose, NVIDIA Container Toolkit, pytest, httpx, Playwright.

**Scope:** Image object detection only in the first implementation. Production model artifacts are Ultralytics `.pt` files. No video processing, annotation editor, automatic training, ONNX/TensorRT, multi-GPU training, automatic model eviction, automatic backup, automatic cleanup, or role system.

---

## File Map

The repository currently contains the design document and static UI prototype. The implementation should add focused backend, frontend, infrastructure, and test directories:

```text
backend/
  app/
    main.py
    config.py
    db.py
    models/
    schemas/
    repositories/
    services/
    workers/
      celery_app.py
      evaluation_worker.py
    runtime/
    observability/
  tests/
    unit/
    integration/
    fixtures/
frontend/
  src/
    app/
    components/
    features/
    lib/
  tests/
  playwright.config.ts
infra/
  docker-compose.yml
  .env.example
docs/
  superpowers/
    plans/
```

The existing `platform-prototype.html` is the visual reference for the admin console. Do not replace it until the React implementation has equivalent core flows.

## Task 1: Create The Runtime Skeleton

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/app/main.py`
- Create: `backend/app/config.py`
- Create: `backend/app/db.py`
- Create: `backend/app/workers/celery_app.py`
- Create: `backend/app/workers/scheduler.py`
- Create: `backend/tests/unit/test_health.py`
- Create: `infra/.env.example`
- Create: `infra/docker-compose.yml`
- Create: `backend/Dockerfile`

- [ ] Write the health test first. Verify the FastAPI app exposes a health response containing database, Redis, GPU, and Worker readiness fields.
- [ ] Run `pytest backend/tests/unit/test_health.py -q` and confirm it fails because the app does not exist.
- [ ] Add the Python project dependencies and application factory.
- [ ] Add configuration for PostgreSQL URL, Redis URL, model/data/log/checkpoint paths, GPU device, API key, and resource limits.
- [ ] Add a health endpoint with `ready`, `degraded`, and `unhealthy` states. Do not make a missing GPU prevent the admin API from starting.
- [ ] Add Docker Compose services for PostgreSQL, Redis, migration, API, inference Worker, training/evaluation Worker, Celery Beat pending sweeper, and the future frontend container.
- [ ] Run the training and evaluation Celery tasks in the single GPU Worker with separate queues; run Celery Beat as a CPU-only scheduler service.
- [ ] Configure the Worker command with `-Q training,evaluation --concurrency=1` and the scheduler command with `celery -A app.workers.celery_app beat`.
- [ ] Add health checks and dependency conditions so migrations complete before API and Workers start.
- [ ] Set Celery training Worker concurrency to `1`.
- [ ] Configure Celery Beat to run the pending-evaluation sweep and lease/reconciliation sweeps; the sweep must be idempotent.
- [ ] Run `pytest backend/tests/unit/test_health.py -q` and expect PASS.
- [ ] Run `docker compose -f infra/docker-compose.yml config` and expect a valid Compose configuration.

## Task 2: Implement Database Models And Migrations

**Files:**
- Create: `backend/app/models/base.py`
- Create: `backend/app/models/model_node.py`
- Create: `backend/app/models/label_schema.py`
- Create: `backend/app/models/dataset.py`
- Create: `backend/app/models/training.py`
- Create: `backend/app/models/evaluation.py`
- Create: `backend/app/models/binding.py`
- Create: `backend/app/models/runtime.py`
- Create: `backend/app/models/resource.py`
- Create: `backend/app/models/operation_log.py`
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`
- Create: `backend/alembic/versions/0001_initial_schema.py`
- Create: `backend/tests/integration/test_schema_constraints.py`

- [ ] Write integration tests for root model creation, parent-child model nodes, immutable dataset snapshots, one running attempt per task, one preparing and one serving runtime per binding, and binding/release composite references.
- [ ] Run the schema tests and confirm they fail before migrations exist.
- [ ] Implement SQLAlchemy models with UUID primary keys and PostgreSQL JSONB for training config, metrics, runtime metadata, and inference configuration snapshots.
- [ ] Add immutable label schema lineage with `parent_schema_id`, `(label_schema_id, class_id)` uniqueness, and immutable `semantic_key`.
- [ ] Add `model_nodes` with nullable `parent_id`, nullable dataset snapshot for imported roots, artifact hash/path, label schema, task type, model family, and lifecycle state.
- [ ] Add `datasets` and immutable `dataset_snapshots` with train/val/test manifests and manifest hash.
- [ ] Add `training_tasks`, `training_attempts`, and checkpoints, including lease token, fencing token, heartbeat, and lease expiry.
- [ ] Add `evaluations` with separate automatic and human states, evaluation policy, metrics, report path, reviewer, and conclusion.
- [ ] Add evaluation-attempt lease fields or a dedicated `evaluation_attempts` table with lease/fencing token, heartbeat, expiry, retry count, and last error.
- [ ] Add a frozen `evaluation_policy` JSONB snapshot to `training_tasks` and `evaluations`.
- [ ] Add `model_bindings` and `model_binding_releases` with monotonic revision numbers, `release_type`, rollback target, current release, and current runtime instance.
- [ ] Add `runtime_instances` with binding/release/model IDs, config hash, generation, fencing token, status, GPU details, and heartbeat. Use the canonical runtime states `loading → preparing → ready → serving → draining → stopped`, with `failed` as a terminal error state.
- [ ] Add `gpu_resource_leases` with resource owner, reserved memory, lease/fencing tokens, heartbeat, and expiry.
- [ ] Add `model_residency_plans` for the administrator's desired resident set, including binding/release/model, GPU device, reserved memory, desired state, and priority.
- [ ] Add partial unique constraints for one running/recovering training attempt per task, one serving runtime per binding, and one non-terminal candidate runtime (`loading`/`preparing`/`ready`) per binding.
- [ ] Add immutable-reference rules for label schemas, dataset snapshots, model artifacts, and report paths; referenced records may be archived but not modified or hard-deleted.
- [ ] Add composite foreign keys so a binding cannot point at another binding's release or runtime instance.
- [ ] Add database migration and rerun integration tests until PASS.

## Task 3: Build Artifact And Dataset Storage

**Files:**
- Create: `backend/app/storage/artifacts.py`
- Create: `backend/app/storage/snapshots.py`
- Create: `backend/app/services/dataset_import.py`
- Create: `backend/app/services/dataset_validation.py`
- Create: `backend/tests/unit/test_dataset_validation.py`
- Create: `backend/tests/integration/test_dataset_snapshots.py`

- [ ] Write tests for valid YOLO train/val/test directories, empty-label negative images, missing labels, invalid class IDs, corrupt images, duplicate hashes across splits, and missing test coverage warnings.
- [ ] Run the dataset tests and confirm failure before implementing validation.
- [ ] Implement server-directory import without browser upload as the first path.
- [ ] Keep generic YOLO directory validation independent of model compatibility; defer parent/child label evolution checks to Task 4.
- [ ] Copy imported data into platform-managed content-addressed storage; never train directly from the mutable source directory.
- [ ] Generate a manifest and overall hash, mark snapshot files read-only, and persist the snapshot record.
- [ ] Preserve positive and negative sample provenance; represent confirmed negative samples as images with empty YOLO label files.
- [ ] Reject structural errors and exact duplicates across train/val/test. Store quality warnings for imbalance, near duplicates, and suspected leakage.
- [ ] Validate the dataset label schema against the target model schema before training.
- [ ] Recompute manifest/file hashes immediately before training and evaluation; refuse execution if content changed.
- [ ] Prevent deletion of snapshots referenced by tasks, models, or evaluations; expose a manual archive operation instead.
- [ ] Add startup and admin-triggered consistency scans that mark missing model, snapshot, checkpoint, or report files as anomalous without deleting metadata.
- [ ] Run unit and integration dataset tests and expect PASS.

## Task 4: Implement Model Registry, Label Evolution, And Bindings

**Files:**
- Create: `backend/app/repositories/model_repository.py`
- Create: `backend/app/repositories/binding_repository.py`
- Create: `backend/app/services/model_import.py`
- Create: `backend/app/services/label_schema_service.py`
- Create: `backend/app/services/binding_service.py`
- Create: `backend/app/schemas/model.py`
- Create: `backend/app/schemas/binding.py`
- Create: `backend/tests/unit/test_label_schema_rules.py`
- Create: `backend/tests/integration/test_model_lineage.py`
- Create: `backend/tests/integration/test_binding_releases.py`

- [ ] Write tests for multiple root models, first-generation schema replacement, later schema extension, rejection of missing inherited classes, stable binding IDs, empty bindings, shared model nodes, monotonic revisions, rollback revisions, semantic-key reuse, display-name changes, and attempts to mutate referenced label classes.
- [ ] Run the tests and confirm failure before service implementation.
- [ ] Implement explicit root `.pt` import with file load and smoke inference validation; imported roots may become `approved` without a dataset snapshot.
- [ ] Implement child model creation with a single parent and immutable artifact metadata.
- [ ] Implement first-generation label schema replacement only when the parent is a root/pretrained node.
- [ ] Implement later schema extension requiring inherited semantic keys and train/val coverage.
- [ ] Implement binding creation, including empty `unbound` bindings, and never reuse IDs.
- [ ] Implement release creation only through a transaction gate that checks approved model/evaluation state, target schema compatibility, and binding ownership.
- [ ] Implement explicit `candidate → approved/rejected → archived` transitions; only the root-import validation path may approve a root without an evaluation record.
- [ ] Implement binding-row locking, revision allocation, composite binding/release constraints, and rollback records with `rollback_target_release_id`.
- [ ] Run model and binding tests and expect PASS.

## Task 5: Implement Training Attempts And Resource Leases

**Files:**
- Create: `backend/app/services/resource_lease.py`
- Create: `backend/app/services/training_service.py`
- Create: `backend/app/workers/train_worker.py`
- Create: `backend/app/workers/attempt_lease.py`
- Create: `backend/tests/unit/test_resource_lease.py`
- Create: `backend/tests/unit/test_training_attempts.py`
- Create: `backend/tests/integration/test_training_task_flow.py`

- [ ] Write tests for queued/running/completed/failed/cancelled transitions, one running attempt, lease renewal, expired lease rejection, fencing-token rejection, compatible checkpoint resume, and changed-input rejection.
- [ ] Run the tests and confirm failure before implementation.
- [ ] Implement manual task creation with fixed `parentModelNodeId`, `datasetSnapshotId`, task type, model family, target binding, and fully expanded training config.
- [ ] Implement PostgreSQL GPU resource reservation with row locking, estimated memory, lease expiry, heartbeat, and fencing token.
- [ ] Implement Celery queue execution with `--concurrency=1` and database-backed attempt ownership.
- [ ] Implement periodic latest/best checkpoint persistence with parent artifact hash, snapshot ID, config hash, epoch, and checkpoint hash.
- [ ] Reject all progress/checkpoint/model writes unless attempt ID, fencing token, and unexpired lease match.
- [ ] Mark orphaned attempts failed after lease timeout; do not auto-recover. Require an administrator confirmation that the old process stopped before retry.
- [ ] On successful training, validate the PT artifact and create a candidate model node; do not create a release automatically.
- [ ] In the same database transaction as candidate creation, create an `evaluation` row with `auto_status=pending`; publish the Celery evaluation message only after commit, relying on the Beat sweep if enqueueing fails.
- [ ] Add operation-log entries for task creation, attempt start/stop/retry, lease expiry, checkpoint recovery, and artifact creation.
- [ ] Run training unit/integration tests and expect PASS.

## Task 6: Implement Automatic Evaluation And Human Review Records

**Files:**
- Create: `backend/app/evaluation/metrics.py`
- Create: `backend/app/evaluation/policy.py`
- Create: `backend/app/evaluation/report.py`
- Create: `backend/app/services/evaluation_service.py`
- Create: `backend/app/workers/evaluation_worker.py`
- Create: `backend/tests/unit/test_detection_metrics.py`
- Create: `backend/tests/unit/test_evaluation_policy.py`
- Create: `backend/tests/integration/test_evaluation_flow.py`

- [ ] Write deterministic tests for TP/FP/FN matching, precision, recall, mAP50, per-class coverage, negative false positives, first-generation absolute evaluation, inherited-class regression evaluation, and threshold failures.
- [ ] Run the tests and confirm failure before implementation.
- [ ] Implement evaluation against the immutable labeled test split from the exact dataset snapshot.
- [ ] Store the full evaluation policy at task/evaluation creation time, including overall thresholds, per-class thresholds, minimum test image counts, negative coverage, and allowed regression drop.
- [ ] Make each automatic evaluation attempt acquire a PostgreSQL GPU lease with the current resident-set reservation; release the lease on success, failure, timeout, or cancellation.
- [ ] Commit the candidate and an `evaluation.pending` record after training, then enqueue a Celery evaluation task after commit. A periodic pending sweep retries enqueue failures; the task claims one evaluation row with `FOR UPDATE SKIP LOCKED` and is idempotent.
- [ ] Skip parent regression comparison when the first-generation target schema is incompatible with the root schema.
- [ ] Require inherited-class coverage and regression thresholds for later schema extensions.
- [ ] Generate a report file with aggregate metrics, per-class metrics, confusion data, false positives, false negatives, and image-level references.
- [ ] Report both `mAP50` and `mAP50-95`.
- [ ] Reset stale `running` evaluation attempts to `pending` through the Celery Beat sweep only after lease expiry and fencing validation.
- [ ] Implement separate automatic and human evaluation statuses; a candidate is publishable only after both pass.
- [ ] On automatic pass and human approval, run the model-node approval transition. Release creation remains in Task 7; on either failure, keep the model `rejected` or `candidate` with a report and no active release.
- [ ] Store reviewer, review time, conclusion, and comments without storing temporary human-uploaded media in the normal inference log.
- [ ] Run evaluation tests and expect PASS.

## Task 7: Implement Runtime Instances And Safe Publish/Rollback

**Files:**
- Create: `backend/app/runtime/model_manager.py`
- Create: `backend/app/runtime/runtime_instance.py`
- Create: `backend/app/runtime/reconciler.py`
- Create: `backend/app/services/release_service.py`
- Create: `backend/app/workers/inference_worker.py`
- Create: `backend/tests/unit/test_runtime_generation.py`
- Create: `backend/tests/integration/test_publish_rollback.py`

- [ ] Write tests for loading, prewarming, health failure, concurrent load rejection, one serving plus one non-terminal candidate, generation fencing, in-flight old-generation requests, failed switch preservation, rollback revision creation, and Worker restart reconciliation.
- [ ] Run the tests and confirm failure before implementation.
- [ ] Implement the inference Worker internal HTTP endpoint. It resolves only `modelBindingId`; callers cannot choose a model node or release directly.
- [ ] Implement runtime instance preparation, PT loading, warmup, health check, heartbeat, and GPU lease acquisition.
- [ ] Reconcile `model_residency_plans` on Worker startup; rehydrate desired pending/ready resident releases as well as the current serving release, subject to GPU capacity.
- [ ] Use the Task 5 GPU reservation service for serving/preparing runtime loads; reject a new load or training attempt when the resident set plus requested reservation exceeds capacity.
- [ ] Keep runtime residency manual: a publish operation may only switch to a ready, administrator-loaded instance and must never load, evict, or reallocate models by itself.
- [ ] Implement manual residency commands separately from publish: load, unload, and inspect are administrator operations; publish refuses a model that is not already ready and resident.
- [ ] Implement the two-phase switch: prepare new instance while old serves, acknowledge ready with fencing token, then atomically update binding release/runtime/generation in PostgreSQL.
- [ ] Define release lifecycle ownership: this task creates a `pending` release first, then manual residency loads a runtime instance linked to that pending release; only the final activation transaction changes the binding current pointers.
- [ ] Implement release transitions explicitly: `pending → preparing` when a residency load starts, `preparing → active` after ready instance activation, `preparing → failed` on load/health failure, and prior active release `active → superseded` after successful activation.
- [ ] Allow a root model release only through an explicit `approval_source=root_import` exception; all non-root releases require an approved model and passed evaluation.
- [ ] Make each request capture the serving runtime snapshot at start so a concurrent switch cannot alter returned model/release/config metadata.
- [ ] Implement draining and stopped states for old instances.
- [ ] Implement startup reconciliation and heartbeat timeout handling: API-only restarts verify existing Worker heartbeats; inference Worker restarts mark in-memory instances stale and rehydrate the current release for the administrator's resident set before serving again.
- [ ] Add runtime operation logs for load, warmup, switch, drain, stop, failure, and reconciliation.
- [ ] Bound the inference Worker request queue and concurrency, release request slots on timeout/error, and test queue-full and timeout behavior.
- [ ] Implement rollback as a new monotonic release pointing to the historical release target, after target runtime preparation succeeds.
- [ ] Refuse serving when no healthy current runtime exists; do not auto-load or auto-evict another model.
- [ ] Run runtime and publish tests and expect PASS.

## Task 8: Implement FastAPI Admin And Internal Endpoints

**Files:**
- Create: `backend/app/api/dependencies.py`
- Create: `backend/app/api/routes/health.py`
- Create: `backend/app/api/routes/models.py`
- Create: `backend/app/api/routes/bindings.py`
- Create: `backend/app/api/routes/datasets.py`
- Create: `backend/app/api/routes/training.py`
- Create: `backend/app/api/routes/evaluations.py`
- Create: `backend/app/api/routes/releases.py`
- Create: `backend/app/api/routes/resources.py`
- Create: `backend/app/api/routes/infer.py`
- Create: `backend/app/schemas/infer.py`
- Create: `backend/app/schemas/evaluation_session.py`
- Create: `backend/app/observability/operation_log.py`
- Create: `backend/app/observability/inference_log.py`
- Create: `backend/tests/integration/api/`

- [ ] Write API tests for API-key authentication, health/degraded states, root model import, dataset import/snapshot, task creation, evaluation submission, binding release, rollback, and `POST /api/infer`.
- [ ] Run API tests and confirm failure before implementation.
- [ ] Add a single platform key check using `X-API-Key`; do not accept keys in URL parameters.
- [ ] Expose admin operations with transaction-backed services only; routes must not update lifecycle fields directly.
- [ ] Add task status, logs/metrics summaries, model lineage, binding history, resource status, evaluation report, and runtime status responses.
- [ ] Add internal inference gateway routing from FastAPI to the inference Worker with connection timeout, inference timeout, maximum input size, and `model_unavailable` mapping.
- [ ] Add the image-only `POST /api/infer` request/response schemas and return binding ID, release ID, model node ID, runtime generation, and inference config hash from the Worker result.
- [ ] Write tests for serving metadata consistency during a concurrent runtime switch, unavailable model responses, input-size rejection, and timeout mapping. Batch inference is out of first-phase scope.
- [ ] Add temporary evaluation-image endpoints and a session service; store files only in the evaluation-session temp directory and exclude them from datasets and normal inference logs.
- [ ] Delete temporary evaluation files when the review session is closed or expires; expose only session metadata after deletion.
- [ ] Implement operation logs and inference summary logs with request ID, model identifiers, latency, result summary, and error; never persist raw media or API keys.
- [ ] Run all FastAPI integration tests and expect PASS.

## Task 9: Build The React Admin Console

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/Dockerfile`
- Create: `frontend/nginx.conf`
- Create: `frontend/vite.config.ts`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/app/App.tsx`
- Create: `frontend/src/app/routes.tsx`
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/src/lib/mockApi.ts`
- Create: `frontend/src/features/dashboard/`
- Create: `frontend/src/features/models/`
- Create: `frontend/src/features/bindings/`
- Create: `frontend/src/features/datasets/`
- Create: `frontend/src/features/training/`
- Create: `frontend/src/features/evaluations/`
- Create: `frontend/src/features/releases/`
- Create: `frontend/src/features/resources/`
- Create: `frontend/playwright.config.ts`
- Create: `frontend/tests/admin-flows.spec.ts`

- [ ] Write Playwright tests for the core admin flows before page implementation: lineage selection, dataset validation result, training task modal, evaluation review, publish confirmation, and rollback.
- [ ] Run the Playwright tests and confirm failure before implementing pages.
- [ ] Use `platform-prototype.html` as the visual baseline: Chinese labels, blue-white palette, desktop operations console, model tree as the primary visual.
- [ ] Implement the navigation and dashboard cards for GPU, resident models, training queue, pending evaluations, and exceptions.
- [ ] Implement the model lineage tree and node detail panel.
- [ ] Implement flat binding list, binding history, empty binding state, current release, and rollback action.
- [ ] Implement server-directory dataset import flow, validation report, snapshot list, and dataset statistics.
- [ ] Implement training creation, preflight warnings/errors, task progress, logs, metrics, checkpoint, and retry state.
- [ ] Implement test-image evaluation view, automatic metrics, temporary image verification, human verdict, and final approval.
- [ ] Implement publish preparation/status, runtime health, release switch, failure state, and rollback.
- [ ] Implement resource control view for manual model residency and GPU reservations; do not add automatic eviction controls.
- [ ] Run Playwright tests and expect PASS.

## Task 10: Integrate, Deploy, And Validate The Closed Loop

**Files:**
- Modify: `infra/docker-compose.yml`
- Create: `infra/README.md`
- Create: `backend/tests/e2e/test_closed_loop.py`
- Create: `backend/tests/fixtures/build_fixtures.py`
- Create: `backend/tests/fixtures/README.md`
- Create: `frontend/tests/e2e/closed-loop.spec.ts`
- Create: `docs/operations/first-run.md`

- [ ] Build all images and start PostgreSQL, Redis, migration, API, inference Worker, training/evaluation Worker, Celery Beat, and frontend with Docker Compose.
- [ ] Set explicit Compose build contexts (`backend` for Python services and `frontend` for the React image), environment injection, and host volume paths for models, datasets, snapshots, logs, checkpoints, and PostgreSQL.
- [ ] Verify health checks and readiness states after a clean restart.
- [ ] Run the fixed acceptance fixtures: root model, first-generation new schema, later schema extension, invalid inherited-class dataset, model load failure, resource shortage, expired training lease, concurrent switch, and service restart.
- [ ] Build deterministic fixtures in `backend/tests/fixtures`: a tiny valid PT smoke model or mocked model adapter for fast tests, fixed YOLO image datasets with known labels and metrics, and explicit failure fixtures. Reserve real GPU PT training for a marked integration job.
- [ ] Verify the complete path: import root → import dataset → create snapshot → create training task → run attempt → create candidate → automatic evaluation → human approval → manual residency → publish → inference → rollback.
- [ ] Verify no inference log contains raw image/video bytes or the platform key.
- [ ] Verify operation and inference logs are non-empty and contain the expected identifiers, state transitions, latency, and error summaries.
- [ ] Verify snapshot hash mutation, stale Worker writes, stale runtime generations, and invalid release references are rejected.
- [ ] Run `pytest backend/tests/unit backend/tests/integration -q` and expect PASS.
- [ ] Run the marked GPU integration suite separately with `pytest backend/tests/e2e/test_closed_loop.py -m gpu -q`; skip it explicitly when NVIDIA/CUDA fixtures are unavailable.
- [ ] Configure the frontend test script as `npx playwright test` and run it with a controlled API fixture or test backend.
- [ ] Configure `frontend/src/lib/api.ts` to use `mockApi.ts` when `VITE_USE_MOCKS=true`; configure `frontend/playwright.config.ts` `webServer` to start Vite with that variable so browser tests are self-contained.
- [ ] Run `npm --prefix frontend exec playwright test` and expect PASS.
- [ ] Run `docker compose -f infra/docker-compose.yml config` and expect PASS.
- [ ] Record first-run configuration, manual backup procedure, manual model residency procedure, and recovery from missing artifacts.

## Execution Notes

- Implement tasks in order; each task should leave a testable boundary.
- Use test-first development for state transitions, validation, leases, and release gates.
- Keep model files and dataset snapshots outside PostgreSQL; only metadata and hashes belong in the database.
- Do not add a role system, automatic model selection, automatic GPU eviction, or unrequested model formats.
- Keep external NVR API details minimal until the internal platform loop is stable.
- The existing prototype is a UI reference, not a production frontend implementation.
