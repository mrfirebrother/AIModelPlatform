# Training Worker Closed-Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a queued training task into a verified YOLO training attempt and a persisted candidate model, with correct state, resource, checkpoint, and failure handling.

**Architecture:** Keep the HTTP API responsible for validation, persistence, and dispatch only. Keep the Celery Worker responsible for loading the task and immutable dataset snapshot, creating and leasing a training attempt, invoking the existing synchronous training function, and persisting the terminal result. Use one owner for each state transition so a task cannot remain `queued` after a worker failure or be completed twice.

**Tech Stack:** FastAPI, SQLAlchemy, PostgreSQL/SQLite tests, Celery, Redis, Ultralytics YOLO, pytest.

**Scope:** YOLO object-detection training from an existing `DatasetSnapshot` to a candidate `ModelNode`. Segmentation support, evaluation, human review, publishing, and rollback are separate follow-up work.

---

## Current Gaps

The implementation worker must account for these existing conditions:

- `backend/app/api/routes/training.py` creates the task and dispatches `run_training`, but the dispatch currently occurs before the dependency-managed transaction is committed.
- `backend/app/workers/train_worker.py:40-55` is a placeholder and only returns a string.
- `execute_training()` already contains GPU lease, manifest preparation, and trainer calls, but no task-level orchestration calls it.
- `training_service.start_attempt()` changes the task to `running` and creates an attempt; `complete_attempt()` creates a candidate model; `fail_attempt()` marks failed state.
- `TrainingTask` stores `epochs` inside `training_config_json`, not as a column.
- `TrainingLogsResponse` reads attempts from the database, so no attempt means the UI correctly reports zero attempts.
- A snapshot manifest contains immutable stored-file paths; training must not read the mutable upload directory.
- First-version retry policy is explicit: `queued` may start, `running` is an idempotent no-op on redelivery, `failed` is terminal with no automatic retry, and `completed`/`cancelled` are terminal no-ops. A user-facing retry operation is a separate follow-up and is not part of this plan.

## File Map

Files expected to change:

- `backend/app/api/routes/training.py`: commit task creation before broker publication and expose dispatch failures as task failures.
- `backend/app/workers/train_worker.py`: implement the Celery entry point and orchestration around `execute_training()`.
- `backend/app/workers/db_session.py`: create a worker-owned SQLAlchemy session factory from platform settings, instead of using request dependency state.
- `backend/app/services/training_service.py`: add narrowly scoped helpers only where the worker needs an atomic state transition or terminal failure update.
- `backend/app/workers/attempt_lease.py`: correct attempt lease initialization and provide the worker heartbeat operation.
- `backend/pyproject.toml`: add the runtime Ultralytics dependency required by the Worker.
- `backend/Dockerfile`: ensure the training runtime installs the dependency and has a documented model-artifact source.
- `backend/Dockerfile.gpu`: provide a CUDA-capable Worker image for the explicit GPU path.
- `infra/docker-compose.gpu.yml`: add an optional GPU Worker override with explicit GPU device reservation; the default compose path remains CPU-safe.
- `backend/app/training/yolo_trainer.py`: add a progress/checkpoint callback boundary if the worker needs per-epoch progress; do not put database access in the trainer.
- `backend/app/services/resource_lease.py`: reuse or extend token-checked GPU lease heartbeat/expiry operations.
- `backend/app/training/resource_scheduler.py`: expose the selected lease identity and synchronize GPU lease heartbeat with the attempt heartbeat.
- `backend/app/workers/scheduler.py`: replace the placeholder lease reconciliation with fenced expiration of both the attempt and its GPU lease, plus task-level failure recording.
- `backend/app/models/training.py`: only if a missing persisted field is proven necessary; prefer existing attempt fields first.
- `backend/alembic/versions/0002_training_cancellation_request.py`: add the task cancellation-request field required for safe cooperative cancellation.
- `backend/tests/integration/api/test_training.py`: verify commit-before-dispatch and response behavior.
- `backend/tests/unit/test_train_worker.py`: add isolated orchestration tests with a fake trainer and temporary snapshot files.
- `backend/tests/integration/test_training_task_flow.py`: verify queued, running, completed, and failed state transitions.
- `backend/tests/integration/test_training_queue.py`: extend resource release and failure-path coverage.

Files to inspect but not change unless required:

- `backend/app/storage/snapshots.py`
- `backend/app/training/yolo_dataset.py`
- `backend/app/training/resource_scheduler.py`
- `backend/app/services/resource_lease.py`
- `backend/app/workers/scheduler.py`
- `backend/app/training/checkpoint_manager.py`
- `backend/app/api/routes/training_logs.py`
- `infra/docker-compose.yml`

## Task 1: Lock the Worker Contract

**Files:**

- Create: `backend/tests/unit/test_train_worker.py`
- Modify: `backend/tests/integration/api/test_training.py`
- Modify: `backend/tests/integration/test_training_task_flow.py`

- [ ] **Step 1: Write the failing worker tests**

  Cover one behavior per test:

  - A queued task creates exactly one `TrainingAttempt` and transitions to `running` before training starts.
  - A successful fake training result transitions the attempt to `completed`, the task to `completed`, and creates one candidate `ModelNode`.
  - A failed fake training result transitions the attempt and task to `failed`, stores the error, and releases the resource lease.
  - A missing task ID fails without creating an attempt.
  - A task already in a terminal state is idempotent and does not create a second attempt.

- [ ] **Step 2: Run the focused tests and confirm the expected failures**

  Run:

  ```text
  python -m pytest backend/tests/unit/test_train_worker.py backend/tests/integration/test_training_task_flow.py -q
  ```

  Expected: the new worker tests fail because `run_training()` is still a placeholder or the orchestration entry point is absent. Existing state-machine tests must remain passing.

- [ ] **Step 3: Define the test seams**

  Use dependency injection or monkeypatching for the database session factory, trainer, clock where needed, and Celery publication. Do not mock the state-machine assertions themselves. The fake trainer must return real `TrainResult` values and create temporary `best.pt`/`last.pt` files when testing success.

- [ ] **Step 4: Commit the contract tests**

  ```text
  git add backend/tests/unit/test_train_worker.py backend/tests/integration/api/test_training.py backend/tests/integration/test_training_task_flow.py
  git commit -m "test: define training worker lifecycle"
  ```

## Task 2: Make Task Publication Transactionally Safe

**Files:**

- Modify: `backend/app/api/routes/training.py:77-123`
- Modify: `backend/tests/integration/api/test_training.py`

- [ ] **Step 1: Add a failing API dispatch test**

  Patch `run_training.delay`, create a task through the API, and assert it receives the committed task ID. Add a failure case where publication raises and the task is not left falsely queued.

- [ ] **Step 2: Run the API tests and verify they fail for the current ordering**

  Run:

  ```text
  python -m pytest backend/tests/integration/api/test_training.py -q
  ```

  Expected: the new transaction/publication assertion fails until the route explicitly handles the commit boundary.

- [ ] **Step 3: Implement the minimal commit boundary**

  Keep the existing task validation and operation logging. Flush the task, commit the creation transaction before calling Celery, then publish the task ID. If publication fails, open a controlled transaction, mark the task failed with a dispatch error, and return an actionable server error. Do not publish a task ID before its row is visible to the Worker.

- [ ] **Step 4: Verify the API behavior**

  Run:

  ```text
  python -m pytest backend/tests/integration/api/test_training.py -q
  ```

  Expected: all API tests pass, including the publication failure case.

- [ ] **Step 5: Commit**

  ```text
  git add backend/app/api/routes/training.py backend/tests/integration/api/test_training.py
  git commit -m "fix: publish committed training tasks"
  ```

## Task 3: Add a Worker-Owned Database Session

**Files:**

- Create: `backend/app/workers/db_session.py`
- Modify: `backend/tests/unit/test_train_worker.py`

- [ ] **Step 1: Write a failing session-factory test**

  Verify the worker can construct and close its own SQLAlchemy session without FastAPI request dependencies. Use a test settings object and a SQLite test engine or an injected factory.

- [ ] **Step 2: Run the test and confirm it fails**

  ```text
  python -m pytest backend/tests/unit/test_train_worker.py -q
  ```

- [ ] **Step 3: Implement the worker session boundary**

  Build the engine/session factory from `get_settings()` and expose a context manager that commits on success, rolls back on exception, and always closes. Keep request-scoped `get_db()` untouched.

- [ ] **Step 4: Verify and commit**

  ```text
  python -m pytest backend/tests/unit/test_train_worker.py -q
  git add backend/app/workers/db_session.py backend/tests/unit/test_train_worker.py
  git commit -m "feat: add worker database session"
  ```

## Task 4: Implement Training Task Orchestration

**Files:**

- Modify: `backend/app/workers/train_worker.py:40-55`
- Modify: `backend/app/services/training_service.py` only for missing atomic helpers
- Modify: `backend/tests/unit/test_train_worker.py`
- Modify: `backend/tests/integration/test_training_task_flow.py`

- [ ] **Step 1: Add failing orchestration tests**

  Test the exact order and state effects:

  - Load the task by UUID and reject missing IDs.
  - Skip terminal tasks without creating duplicate attempts.
  - Validate the referenced snapshot and parent model before starting an attempt.
  - Call `start_attempt()` and commit the `running` state before expensive work.
  - Call the execution function once with the task configuration and snapshot manifest.
  - Call `complete_attempt()` only after a successful result and valid best artifact.
  - Call `fail_attempt()` for expected training/resource errors.

- [ ] **Step 2: Run and confirm failure**

  ```text
  python -m pytest backend/tests/unit/test_train_worker.py backend/tests/integration/test_training_task_flow.py -q
  ```

- [ ] **Step 3: Implement `run_training()`**

  The Celery task should:

  1. Parse the task UUID.
  2. Open a worker session.
  3. Load and lock the task where the database supports row locking.
  4. Return idempotently for missing/terminal tasks with an explicit result. Return `already_running` for a redelivered `running` task; do not create a second attempt. Do not auto-retry `failed` tasks in this first version.
  5. Validate snapshot, parent model, task type, model family, and label-schema compatibility.
  6. Create an attempt through `training_service.start_attempt()` and commit.
  7. Build the immutable dataset manifest from the snapshot.
  8. Call `execute_training()` with a unique output directory.
  9. On success, verify artifacts, then call `complete_attempt()` exactly once.
  10. On failure, call `fail_attempt()` exactly once and also set `TrainingTask.failure_reason`.
  11. Commit terminal state and close the session.

  The Worker must never leave a task in `queued` after it has started processing. Any uncaught exception must be converted into an attempt/task failure before being re-raised for Celery observability.

  The Worker owns normal success/failure transitions. `execute_training()` must not set an attempt to `completed` or `failed`; it may release resources and return a `TrainResult`, while the Worker alone calls `complete_attempt()` or `fail_attempt()`. The reconciliation task is the only owner of the stale-worker path: after locking task -> attempt -> lease, it may conditionally fence an expired running attempt, expire its GPU lease, mark the attempt failed, and set the task failure reason. The Worker must use the attempt lease/fencing token in its final update and become a no-op if reconciliation has already fenced it.

  Define token ownership explicitly: `TrainingAttempt.fencing_token` is the monotonic integer used for stale-worker protection and is never overwritten with the GPU lease UUID. `TrainingAttempt.lease_token` belongs only to the attempt lease; the GPU lease keeps its own token. Terminal updates must include the expected attempt fencing token and status in the conditional predicate, and must return a no-op result when the predicate matches zero rows.

  If validation or resource setup fails before an attempt exists, call a task-level `mark_task_failed()` helper that sets `TrainingTask.status=failed` and `failure_reason` atomically. If the worker crashes after an attempt starts, reconciliation will mark it failed after lease expiry and expire/release its GPU lease; this plan does not automatically requeue it.

- [ ] **Step 4: Verify success and failure paths**

  ```text
  python -m pytest backend/tests/unit/test_train_worker.py backend/tests/integration/test_training_task_flow.py -q
  ```

  Expected: queued-to-running, success, failure, terminal no-op, and idempotency tests pass. Automatic retry is explicitly not part of this first version.

- [ ] **Step 5: Commit**

  ```text
  git add backend/app/workers/train_worker.py backend/app/services/training_service.py backend/tests/unit/test_train_worker.py backend/tests/integration/test_training_task_flow.py
  git commit -m "feat: orchestrate queued training tasks"
  ```

## Task 5: Connect Snapshot, Resource, Trainer, and Artifacts

**Files:**

- Modify: `backend/app/workers/train_worker.py:58-137`
- Modify: `backend/app/training/yolo_trainer.py` if progress callbacks are required
- Modify: `backend/app/training/yolo_dataset.py` only for manifest normalization gaps
- Modify: `backend/app/training/checkpoint_manager.py` if persisted metadata is missing
- Modify: `backend/pyproject.toml`
- Modify: `backend/Dockerfile` if runtime installation or model preflight needs it
- Create: `backend/Dockerfile.gpu` if the explicit GPU path requires a CUDA base image
- Modify: `backend/app/services/resource_lease.py`
- Modify: `backend/app/training/resource_scheduler.py`
- Modify: `backend/app/workers/scheduler.py`
- Modify: `backend/tests/integration/test_training_queue.py`
- Modify: `backend/tests/unit/test_train_worker.py`

- [ ] **Step 1: Add failing integration tests with a tiny real snapshot**

  Build a temporary snapshot with one or two YOLO detection images, labels, `nc`, and `names`. Assert that training preparation reads only `image_stored_path` and `label_stored_path`, creates the normalized YOLO directory, and writes `data.yaml`.

- [ ] **Step 2: Add resource and cleanup assertions**

  Assert that the required memory comes from `resource_config_json` with a documented default, a missing GPU resource produces a failed attempt, and every success/failure path releases an active lease.

- [ ] **Step 3: Implement manifest normalization**

  Convert the snapshot manifest into the shape required by `YoloDataset.from_manifest`:

  - `train_files`, `val_files`, `test_files`
  - numeric `nc`
  - normalized `names` list

  Derive `nc` and class names from `snapshot.label_schema.classes`, using the snapshot manifest's `data_yaml` only as a consistency check. Read the manifest JSON from `manifest_path`, recompute and compare its SHA-256 to `snapshot.manifest_hash` before consuming any file paths, and reject legacy entries that do not contain immutable stored paths rather than copying from the mutable upload source. Reject an empty/invalid manifest with a clear task failure instead of starting a zero-image training run.

- [ ] **Step 4: Implement unique output and artifact persistence**

  Use a path such as `/data/checkpoints/{task_id}/attempt_{attempt_no}`. After training, verify `best.pt` or the configured result artifact exists, calculate its SHA-256 hash, and pass the immutable artifact path/hash to `complete_attempt()`. Persist checkpoint rows only for artifacts that actually exist.

- [ ] **Step 5: Add progress updates without coupling the trainer to SQLAlchemy**

  Use a background heartbeat loop owned by the Worker. Every 60 seconds it opens a separate session, locks task -> attempt -> lease, updates both lease expiries, and commits independently of the training transaction. Stop the loop in a `finally` block after the trainer returns. If it sees a cancellation request, it signals the Worker but does not release the GPU while the synchronous trainer may still be using it. If per-epoch progress is available, it may update `current_epoch` through the same worker boundary; the YOLO trainer must remain database-agnostic.

  Implement the `platform.scheduler.reconcile_leases` task for expired attempts and GPU leases. It must lock task -> attempt -> lease, condition on the expected fencing token, mark the attempt and task failed with an actionable reason, and expire the GPU lease without double-updating an already terminal record. A stale Worker that later returns must fail its conditional terminal update and must not create a candidate.

- [ ] **Step 6: Define resource and cancellation semantics**

  Normalize resource configuration with this precedence: `resource_config_json.memory_mb`, legacy `gpu_memory_mb`, legacy `gpu.memory_mb`, then the settings default. Set `attempt.gpu_device` before reserving a GPU. The exact device contract is: `device=cpu` uses CPU and creates no GPU lease; `device=gpu` requires CUDA and a matching `GPUResource`, otherwise the task fails; an omitted device uses GPU only when `GPU_READY=true`, otherwise it uses CPU. Do not silently downgrade an explicit GPU request to CPU. GPU training must reserve and release exactly one lease.

  Synchronize both the attempt lease and GPU lease expiry. `start_attempt()` must initialize `lease_expires_at` to now plus 30 minutes. Commit the attempt state before expensive work, commit the GPU reservation before entering the trainer, commit each heartbeat independently at least every 60 seconds, and commit release/terminalization in a final transaction. Heartbeats use a separate worker-owned database session so they are visible during synchronous training. Let the reconciliation task fail only expired attempts and update the owning task's `failure_reason`. Use one lock order everywhere: task row, then attempt row, then GPU lease row. Before creating a candidate model or writing a success terminal state, reload the task/attempt with a conditional lock and verify it was not cancelled or fenced. `cancel_task()` and worker terminalization must use the same lock order and conditional status update; a cancellation observed after the trainer returns must win over success. A redelivered `running` task returns `already_running`; it is not requeued in this first version.

  Add `TrainingTask.cancellation_requested` with a migration. For a queued task, cancellation can immediately mark the task cancelled. For a running task, the API sets only this request flag; the Worker cooperatively observes it, waits for the trainer call to return, then marks the attempt/task cancelled and releases the GPU lease. Do not release a lease while the trainer may still execute. The Worker must not overwrite `cancelled` with `completed` or create a candidate after cancellation. Keep the lock order `task -> attempt -> lease` in cancellation and terminalization paths.

- [ ] **Step 7: Verify and commit**

  ```text
  python -m pytest backend/tests/integration/test_training_queue.py backend/tests/unit/test_train_worker.py -q
  git add backend/app/workers/train_worker.py backend/app/training/yolo_trainer.py backend/app/training/yolo_dataset.py backend/app/training/checkpoint_manager.py backend/pyproject.toml backend/Dockerfile backend/tests/integration/test_training_queue.py backend/tests/unit/test_train_worker.py
  git commit -m "feat: execute training from dataset snapshots"
  ```

  Before container verification, rebuild the Worker image and verify the parent `.pt` artifact is available in the mounted `/data/models` directory. The CPU image must install `ultralytics` and use CPU-compatible PyTorch; the optional GPU image must provide CUDA-compatible PyTorch and expose the GPU device. The Worker must fail clearly during preflight if the dependency or model artifact is missing.

## Task 6: Persist Checkpoints And Candidate Lineage

**Files:**

- Modify: `backend/app/services/training_service.py:115-152`
- Modify: `backend/app/workers/train_worker.py`
- Modify: `backend/app/models/training.py`
- Create: `backend/alembic/versions/0002_training_cancellation_request.py`
- Modify: `backend/tests/unit/test_train_worker.py`
- Modify: `backend/tests/integration/test_training_task_flow.py`

- [ ] **Step 1: Add failing persistence tests**

  Assert that a successful run stores the candidate model with the task snapshot's `label_schema_id`, and that each persisted checkpoint contains `parent_artifact_hash`, `dataset_snapshot_id`, `training_config_hash`, epoch, artifact path, and artifact hash. Test the no-parent case with an explicit sentinel parent hash such as `sha256:none`.

- [ ] **Step 2: Define the terminal transition sequence**

  Keep the attempt in `running` while `execute_training()` returns. On success, persist checkpoint records first, set `latest_checkpoint_id`/`best_checkpoint_id`, then call `complete_attempt()` while the attempt is still running. On failure, call `fail_attempt()` while the attempt is still running and set `TrainingTask.failure_reason` in the same transaction. Never call `complete_attempt()` or `fail_attempt()` after `execute_training()` has changed the attempt status.

- [ ] **Step 3: Implement candidate lineage**

  Update `complete_attempt()` or its call contract so the generated candidate copies `label_schema_id` from the dataset snapshot. Before starting training, allow a parent with no schema for the bootstrap model; when both parent and snapshot have schemas, require matching schema IDs in this first version. Defer append-only class evolution to a separate plan. Preserve the parent model hash and training configuration hash in checkpoint metadata.

  Update cancellation handling so cancelling a running task marks the attempt cancelled and releases/invalidates its active GPU lease. The worker must not overwrite `cancelled` with `completed` or create a candidate after cancellation. Keep the lock order `task -> attempt -> lease` in cancellation and terminalization paths.

- [ ] **Step 4: Verify and commit**

  ```text
  python -m pytest backend/tests/unit/test_train_worker.py backend/tests/integration/test_training_task_flow.py -q
  git add backend/app/services/training_service.py backend/app/workers/train_worker.py backend/tests/unit/test_train_worker.py backend/tests/integration/test_training_task_flow.py
  git commit -m "feat: persist training checkpoints and lineage"
  ```

## Task 7: End-to-End Container Smoke Test

**Files:**

- Modify: `frontend/src/features/training/TrainingPage.tsx` only if displayed state fields are missing
- Modify: `backend/tests/e2e/test_closed_loop.py` or add a focused smoke test
- Inspect: `infra/docker-compose.yml`
- Create: `infra/docker-compose.gpu.yml` if GPU reservation is not already provided by the base Compose file

- [ ] **Step 1: Add a smoke-test checklist**

  Use a one-epoch CPU-safe test configuration and verify:

  - `POST /api/training/tasks` returns `201`.
  - The task first appears as `queued`.
  - Worker logs show `platform.training.run_training` received.
  - The task changes to `running`.
  - At least one attempt exists.
  - The task reaches `completed` or a meaningful `failed` state.
  - No active GPU lease remains after terminal state.
  - A candidate model or explicit failure reason is persisted.
  - `/training` shows no `undefined` and displays task name, status, epoch, and attempt count.

- [ ] **Step 2: Build and start the services**

  ```text
  docker compose -f infra/docker-compose.yml build api frontend worker
  docker compose -f infra/docker-compose.yml up -d api frontend worker
  ```

- [ ] **Step 3: Execute the smoke test**

  Verify `dataset/example1.zip` exists; if it does not, create it from `dataset/example1` before the smoke run. Upload it through `/api/datasets/upload`, create the dataset from the returned path, then call `GET /api/datasets` and read the created item's `latestSnapshotId`. This avoids assuming a persisted snapshot exists before the test. Do not use `example2/crack-seg` because segmentation is outside this plan.

  For GPU verification, apply `infra/docker-compose.gpu.yml`, build the CUDA Worker image, set `device=gpu`, and require CUDA availability. The default CPU smoke test must not create a GPU lease or require a `GPUResource` row.

  Use an explicit parent model from `GET /api/models`. If none exists, upload `yolov10n.pt` through `/api/models/upload` and create a model row from the returned path. Verify its artifact path exists inside `/data/models`; do not rely on the no-parent fallback to `yolov8n.pt` unless that exact file is mounted into the Worker container.

  Use an explicit parent model from `GET /api/models` for the smoke task and verify its artifact path exists inside `/data/models`. Do not rely on the no-parent fallback to `yolov8n.pt` unless that exact file has been mounted; the repository-root weight file is not automatically available inside the Worker container.

- [ ] **Step 4: Inspect logs and database state**

  ```text
  docker logs --since 10m infra-worker-1
  docker compose -f infra/docker-compose.yml ps
  ```

  Confirm the Worker received the task and the database state agrees with the logs.

- [ ] **Step 5: Verify the real page**

  Open `http://localhost:8080/training`, inspect the rendered DOM, click the task, switch Logs/Loss/Checkpoint tabs, check browser console errors, and confirm the key state transitions are reflected in the DOM.

- [ ] **Step 6: Commit the smoke coverage**

  ```text
  git add backend/tests/e2e/test_closed_loop.py frontend/src/features/training/TrainingPage.tsx
  git commit -m "test: verify training worker closed loop"
  ```

## Final Verification Checklist

- [ ] Run the focused backend suite:

  ```text
  python -m pytest backend/tests/unit/test_train_worker.py backend/tests/unit/test_training_attempts.py backend/tests/integration/api/test_training.py backend/tests/integration/test_training_task_flow.py backend/tests/integration/test_training_queue.py -q
  ```

  Expected: all tests pass.

- [ ] Run the broader backend tests without modifying fixtures:

  ```text
  python -m pytest backend/tests -q
  ```

  Record and investigate every failure. Do not treat unrelated fixture deletions or infrastructure failures as successful verification.

- [ ] Build the frontend:

  ```text
  npm run build
  ```

- [ ] Verify the Worker and API containers are healthy:

  ```text
  docker compose -f infra/docker-compose.yml ps
  ```

- [ ] Verify the task state in the API and UI after a real smoke run.

- [ ] Verify no task remains `queued` after Worker receipt.

- [ ] Verify no active lease remains after success or failure.

- [ ] Verify cancelling a running task releases/invalidates its GPU lease and prevents candidate creation.

- [ ] Verify candidate model artifact and hash exist after success.

- [ ] Verify task/attempt error text is actionable after forced failure.

- [ ] Verify the final diff contains only intended code/tests/docs and no generated Redis/database artifacts.

## Out of Scope Follow-Ups

- Implement real per-epoch checkpoint persistence and resume semantics if the current trainer callback is insufficient.
- Add evaluation task dispatch after candidate-model creation.
- Add human review and publish/rollback state transitions.
- Add YOLO segmentation support for `example2/crack-seg`.
- Add a transactional outbox if broker publication reliability must survive API process failure between database commit and Celery publish.
