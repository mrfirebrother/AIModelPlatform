# AGENTS.md

> **Baseline**: `4bcebb77` (`master`). Read the "Local Environment", "Verified Runtime & Build" and
> "Known Test Failures" sections before trusting any command below.
>
> **What was actually executed when this file was written** (as opposed to read out of the source):
> `git` 2.55.0 (repo/ignore checks), `python backend/scripts/verify_bom.py`, a full
> `pytest -c NUL -p no:cacheprovider backend/tests -q -m "not slow" --continue-on-collection-errors` run in a
> Python 3.11.9 venv with a **minimal** dependency set (no torch, no ultralytics) →
> **544 passed, 154 failed, 3 skipped, 14 errors, 30 deselected**; a live `uvicorn` smoke test with real HTTP
> probes; `npm install` + `npm run build` for the frontend (exit 0); and a full **non-Docker deployment** —
> PostgreSQL 16.12 + Redis 5.0.14.1 as Windows services, `alembic upgrade head` (including the new `0004` repair
> revision), verified end to end through the Vite proxy. 132 of the 154 failures are pre-existing fixture drift
> (`model_nodes.code`), **not** an artifact of the minimal env. `docker compose` was **not** run (no Docker on
> this machine).

## Stack & Layout

- **Backend**: Python 3.11 (`backend/pyproject.toml`: `requires-python = ">=3.11,<3.12"`), FastAPI (`backend/app/main.py:create_app`), SQLAlchemy 2 + Alembic, Celery + Redis, Ultralytics YOLO. Entrypoint `app.main:app` / `uvicorn`.
- **Frontend**: React 18 + TypeScript + Vite (`frontend/src/main.tsx`), React Router. `VITE_USE_MOCKS=true` switches `src/lib/createApi.ts` to `mockApi`.
- **Infra**: `infra/docker-compose.yml` — postgres:16, redis:7-alpine, migrate (one-shot `alembic upgrade head`), api, inference (placeholder: prints and `time.sleep`s), worker (`celery -A app.workers.celery_app worker -Q training,evaluation,default --concurrency=1 --loglevel=INFO`), scheduler (beat), frontend (node build + nginx). Frontend nginx at `8080` proxies `/api` to `api:8000`; direct API at `8000`.
- **Dirs**: `backend/app/{api/routes,models,schemas,services,repositories,workers,runtime,evaluation,training,inference,storage,observability}` | `backend/alembic/versions` | `frontend/src/{app,features,lib}` | `frontend/tests/{e2e}` | `infra/` | `docs/`
- **Routers**: 17 `include_router` calls in `create_app` — alerts, health, `observability.health_probes`, models, model_diff, bindings, datasets, training, training_logs, evaluations, releases, resources, infer, inference, gpu, operations, backup. The authoritative endpoint count is `GET /openapi.json` → **64 paths**.
  - Do **not** count endpoints via `len(app.routes)`: this FastAPI version keeps included routers nested as `_IncludedRouter` objects (17) instead of flattening them, so `app.routes` is only 23 (17 nested + 4 FastAPI defaults + the 2 `/health*` routes defined inline).
- **Health routes — the effective set comes from the probes router, not `main.py`**:
  - `/health/live` → `{"status": "ok"}` (from `observability/health_probes.py:57`). `main.py` defines a *second* `/health/live` returning `{"status": "alive"}`, but `probes_router` is included first, so **the `main.py` handler is shadowed dead code**. Confirmed by a live request (`{"status":"ok"}`), and `backend/tests/unit/test_health.py` / `integration/api/test_health.py` assert `"alive"` and therefore fail.
  - `/health/ready` → 200 `{"status": "ok", "checks": {...}}`, or 503 `{"status": "fail", ...}` when postgres/redis fail.
  - `/health/startup` exists as well.
  - `/health` (defined inline in `create_app`) aggregates the four checks: missing postgres **or** redis ⇒ `unhealthy`, gpu **or** worker ⇒ `degraded`, else `ready`.

## Repo Size & Tracked Artifacts (read before cloning or `git add`)

Measured with `git ls-files` at `4bcebb77`:

- 22,728 tracked files; a fresh clone is ≈ **2.7 GB / ~22,700 files**. `dataset/` alone is ≈ 1.7 GB: `dataset/example1.zip` (80 MB) *plus* the extracted `dataset/example1/` and `dataset/example2/`. `.git` itself holds a 913 MiB pack (18,897 objects).
- `.gitignore` lists `dataset/`, `infra/{datasets,models,checkpoints,redis,logs}/`, `*.pt`, `*.zip`, `*.tar.gz` — **but 22,482 tracked files match those ignore rules** (`git ls-files -i -c --exclude-standard`). They were committed before the ignore rules landed, and an ignore rule never untracks existing files. Breakdown: `dataset/` 22,466, `infra/` 7, `*.zip` 5, `*.pt` 4 (patterns overlap).
- `git check-ignore` prints **nothing** for these paths because it skips tracked files — that is expected, not a sign the rules are missing. Use `git ls-files -i -c --exclude-standard` instead.
- Present in the clone despite the rules: `infra/datasets/*.zip`, `infra/models/d5db428c0e05afc0_yolov10n.pt`, `infra/checkpoints/<uuid>/attempt_1/checkpoints/train/args.yaml`, `infra/redis/dump.rdb`, root `yolov8n.pt` / `yolov10n.pt` (~12 MB).
- Consequences: never `git add -A` blindly; edits or deletions under those paths are real diffs; `infra/{models,datasets,logs,checkpoints,redis}` are **bind-mount targets, not build inputs** (`infra/logs` and `infra/postgres` do not exist in the clone).

## Local Environment (this machine, verified)

- **Python**: 3.11.9 installed at `C:\Users\admin\AppData\Local\Programs\Python\Python311\python.exe`; the `py` launcher is at `C:\Windows\py.exe` (`py -3.11 -V` works).
- **Trap**: `python` and `python3` on PATH resolve to the Microsoft Store alias stub
  (`C:\Users\admin\AppData\Local\Microsoft\WindowsApps\python.exe`), which fails with
  *"Python was not found but can be installed from the Microsoft Store"* even though 3.11.9 is installed.
  Either turn off the App Execution Aliases (Settings → Apps → Advanced app settings → App execution aliases →
  `python.exe` / `python3.exe`) or call `py -3.11` / the full path. Every bare `python ...` command below is
  affected.
- **Git**: 2.55.0 installed at `C:\Program Files\Git\cmd\git.exe` but **not on PATH** — a bare `git` raises
  `CommandNotFoundException`. Use the full path or add that directory to PATH.
- **Docker is not installed** (no `docker` on PATH, nothing under `Program Files\Docker`), so the compose
  workflow is unverified here. `node` v22.23.2 and `npm` 10.9.8 are available.
- A venv exists at `.venv` (3.11.9) with only the light deps installed: fastapi, starlette, sqlalchemy, alembic,
  pydantic-settings, celery, redis, httpx, python-multipart, pyyaml, pillow, anyio, pytest, pytest-asyncio, uvicorn.
  **No torch and no ultralytics.**
- `.gitignore` has **no `.venv/` entry** (nor `venv/`), so `.venv` is instead hidden via `.git/info/exclude`
  (local-only). Adding `.venv/` to `.gitignore` is the durable fix if you want it shared.
- Useful interpreter variable for the commands below:
  `$vpy = ".venv\Scripts\python.exe"`.

## Verified Runtime & Build (executed)

- **API smoke test**: `.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`
  starts cleanly with `PLATFORM_API_KEY=change-me` and no Postgres/Redis. Observed responses:
  - `GET /health/live` → 200 `{"status":"ok"}`
  - `GET /health/ready` → **503** `{"status":"fail","checks":{"postgres":"fail","redis":"fail"}}`
  - `GET /health` → 200 `{"status":"unhealthy","checks":{"postgres":false,"redis":false,"gpu":false,"worker":false}}`
  - `GET /api/models` with no `X-API-Key` → 401 `{"detail":"Invalid or missing API key"}`
  - `GET /openapi.json` → 64 paths (the real endpoint count).
- **Caution**: DB-backed endpoints (e.g. `GET /api/models` *with* a valid key) **hang** when Postgres is
  unreachable — the application engine sets no connect timeout, so a local smoke test without a database stalls
  for minutes. `/health/ready` is safe (1 s timeouts) and is the right probe for local checks.
- **Frontend**: `npm install` (78 packages, 13 s, default registry) then `npm run build`
  (`tsc -b && vite build`) → exit 0, 374 modules, `dist/assets/index-*.js` 334.51 kB (gzip 106.63 kB),
  CSS 33.10 kB, built in 4.37 s.
  - The build emits a CSS-minify warning caused by a **committed bug at `frontend/src/app/App.css:1027`**:
    `background: #fafbfc;` sits between the already-closed `.grid-table-row { ... }` rule (line 1026) and
    `.grid-table-row-selected` (line 1028) with **no selector**, so esbuild drops it. There is no
    `.grid-table-row:hover` rule either — the declaration looks like the orphan of a lost selector line.
  - **`tsc -b` rewrites the tracked `frontend/tsconfig.tsbuildinfo`**, dirtying `git status` on every build.
    Run `git restore frontend/tsconfig.tsbuildinfo` afterwards. `frontend/dist/` and `frontend/node_modules/` are
    ignored by `.gitignore` and stay clean.

## Commands (exact)

### Backend

```bash
# All tests from repo root — must use -c NUL because pyproject.toml/alembic.ini may carry UTF-8 BOM.
# Add --continue-on-collection-errors, otherwise the run ABORTS with exit 2 and executes ZERO tests
# (test_yolo_engine.py and test_inference_flow.py import yolo_engine -> `from ultralytics import YOLO`
# at module level, which is missing in a light env).
python -m pytest -c NUL -p no:cacheprovider backend/tests -q --continue-on-collection-errors
# Single file / single test
python -m pytest -c NUL -p no:cacheprovider backend/tests/unit/test_health.py -q
python -m pytest -c NUL -p no:cacheprovider backend/tests/unit/test_health.py -k test_gpu_ready_override -q
# Skip GPU/slow tests (deselects exactly 30 tests at this commit)
python -m pytest -c NUL -p no:cacheprovider backend/tests -q -m "not slow" --continue-on-collection-errors
# Compile check, BOM check
python -m compileall backend
python backend/scripts/verify_bom.py
# Migrations (needs POSTGRES_URL or POSTGRES_HOST/PORT/DB/USER/PASSWORD env)
alembic upgrade head
```

Tests import exclusively as `backend.app.*` (216 such imports, zero `from app.*`), which is why `-c NUL` is safe locally even though it discards `pythonpath = ["."]`.

### Frontend

```bash
cd frontend
npm install
npm run dev      # Vite :3000, proxies /api -> 127.0.0.1:8000 (vite.config.ts)
npm run build    # tsc -b && vite build  (then: git restore frontend/tsconfig.tsbuildinfo)
npm run test     # playwright test — config runs `npx vite --mode mock` automatically
npx playwright test tests/e2e/ --reporter=list
```

Specs live at `frontend/tests/admin-flows.spec.ts` and `frontend/tests/e2e/closed-loop.spec.ts` (`playwright.config.ts` `testDir: "./tests"`, so a bare `npm run test` covers both). Playwright browsers were **not** installed during verification, so `npm run test` is unverified here.

### Docker

```bash
cd infra
cp .env.example .env   # edit PLATFORM_API_KEY, POSTGRES_PASSWORD, HOST_*_DIR
docker compose up -d
docker compose logs api
docker compose build api --no-cache
# Health: api 8000/health/ready, frontend 8080/, postgres 5432, redis 6379
```

`backend/Dockerfile` pins `pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/` before `pip install .`, so image builds pull from the Aliyun PyPI mirror. The same mirror was used to populate `.venv` and worked.

### Local run without Docker

`start.bat` starts uvicorn + Vite locally, but it **hardcodes `cd /d F:\Work\EStructure\AIModelPlatform`** (the original author's machine) and calls `python -m uvicorn backend.app.main:app`. Fix that path before using the script; otherwise start the two processes manually:

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
cd frontend && npx vite --host 127.0.0.1 --port 3000
```

## Local Non-Docker Deployment (this machine, verified running)

Docker is not required — the whole stack runs natively on Windows:

| Piece | How it was installed | Location / service |
|---|---|---|
| PostgreSQL 16.12 | EDB `postgresql-16.12-1-windows-x64-binaries.zip` + `initdb -E UTF8 --locale=C`, then `pg_ctl register` | `C:\PostgreSQL\pgsql` (data `…\pgsql\data`), service `postgresql-x64-16`, Automatic |
| Redis 5.0.14.1 | `Redis-x64-5.0.14.1.msi` (tporadowski build), silent `msiexec /qn` | service `Redis`, Automatic, binds **127.0.0.1:6379** only |
| Python env | `.venv` (3.11.9) with the deps pinned by `backend/pyproject.toml`, minus torch/ultralytics | `.venv\Scripts\python.exe` |
| Runtime data | `MODEL_DIR` / `DATASET_DIR` / `LOG_DIR` / `CHECKPOINT_DIR` point outside the repo | `C:\Users\admin\aip-data\{models,datasets,logs,checkpoints}` |
| DB | database `ai_platform`, user `postgres`, password `change-me` (localhost only) | `alembic_version = 0005`, 19 tables |

- **Respect the version ranges in `backend/pyproject.toml`.** Installing newest-by-name silently broke Redis:
  `redis` 8.x opens with `HELLO 3` (RESP3), which the Redis 5.0 server rejects (`unknown command HELLO`), so
  `/health/ready` reported `redis: fail` while every individual version looked fine. `redis>=5,<6` and
  `Pillow>=10,<11` are the pins that mattered here.
- Prerequisite that cost a detour: the EDB zip ships **no MSVC runtime**, so every PostgreSQL binary aborts with
  `0xC0000135` (`STATUS_DLL_NOT_FOUND`) until the **VC++ 2015-2022 x64 redistributable** is installed.
- `Settings` reads `env_file=".env"` **relative to the working directory**, so keep one `.env` at the repo root and
  run uvicorn from the root (`uvicorn backend.app.main:app`, as `start.bat` does). For Alembic from the root you
  also need `PYTHONPATH=backend` (see Migrations).
- The frontend reaches the API through the Vite dev proxy (`/api` → `127.0.0.1:8000`), so no CORS is involved.
  For real data **do not set `VITE_USE_MOCKS`**: `--mode mock` alone does not enable mocks (there is no
  `.env.mock`), the env var is what `createApi()` checks. Vite binds `::1` only → use `http://localhost:3000`
  (`http://127.0.0.1:3000` does not connect).
- Launcher (kept outside the repo so `git status` stays clean):
  `powershell -ExecutionPolicy Bypass -File C:\Users\admin\aip-data\start-platform.ps1` — writes the hidden
  launchers (`run-hidden.vbs` + `run-{api,worker,frontend}.cmd`) and registers/runs three **Scheduled Tasks**
  (`AIP-API`, `AIP-Worker`, `AIP-Frontend`; `onlogon`, highest privileges), so the processes are parented by the
  Task Scheduler service (`svchost`). `stop-platform.ps1` ends them and kills strays; logs land in
  `C:\Users\admin\aip-data\logs\`.
  - **`run-hidden.vbs` exists because a visible console is a kill switch.** A `cmd /c ...` task opens a console
    window, and closing that window sends Ctrl+C: the whole stack died exactly this way once
    (`LastTaskResult = 0xC000013A` on all three tasks) shortly after a 2.4 h training run had finished. Launching via
    `wscript` + `shell.Run(cmd, 0, False)` keeps the window hidden.
  - **API and worker run as `SYSTEM`** (session 0): no window at all, and logging off does not kill them mid-run.
    The frontend stays as the current user (hidden), since viewing the UI requires being logged on anyway.
    - Caveat of running as SYSTEM: Ultralytics' cache moves to
      `C:\Windows\System32\config\systemprofile\AppData\Roaming\Ultralytics`. On first use it tries to download
      `Arial.ttf` from `github.com` — unreachable on this network — and stalls for ~90 s on 30 s connect timeouts
      (an evaluation hung that way). The font was seeded into that directory; the evaluation path also passes
      `plots=False`, which avoids the dependency entirely.
  - **Never start the stack with plain `Start-Process` from a sandboxed/agent shell:** when that shell is killed its
    whole process tree goes with it — an observed 120 s timeout tore down uvicorn, the Celery worker and Vite
    mid-run.
  - **System sleep silently eats training time — and it was the default.** The balanced power plan had
    `STANDBYIDLE` AC = 1800 s, so after 30 min without user input the box slept mid-run (Kernel-Power event 42 at
    17:19:41, resume at 19:34:51; a single epoch logged **8265 s instead of ~160 s**, because ultralytics' `time`
    column is wall-clock based). The run survived and resumed, but 2 h 15 m were lost. Now fixed with
    `powercfg /change standby-timeout-ac 0`; display-off is harmless, only sleep/hibernate stops training. Verify
    with `powercfg /q SCHEME_CURRENT SUB_SLEEP STANDBYIDLE` — the AC line must read `0x00000000`.
  - Helper scripts kept beside the launcher (outside the repo, so `git status` stays clean): `_launch_train.py`
    (create a training task through the real API path), `_watch_train.py` (GPU/CPU telemetry for a run),
    `_cancel_check.py` / `_cancel_check2.py` (cancel-path verification), `_ab_cache.py` / `_ab_control2.py` (the
    `cache` A/B), plus the earlier `_schema_check.py` / `_cuda_check.py` / `_gpu_check.py` probes.
  - PostgreSQL and Redis start with Windows automatically.
- Verified end to end: `http://localhost:3000/` serves the app, `http://localhost:3000/api/models` returns real
  backend JSON, `http://127.0.0.1:8000/health/ready` → `{"status":"ok","checks":{"postgres":"ok","redis":"ok"}}`,
  `http://127.0.0.1:8000/docs` → 64 paths, and `/health` → `"ready"` once `GPU_READY`/`WORKER_READY` are true.
- **`/api/gpu/status` now reports real telemetry.** `runtime/gpu_monitor.py` reads NVML through `pynvml`
  (`nvidia-ml-py`, which ultralytics pulls in) and falls back to parsing `nvidia-smi`, then to the simulated values
  with a logged warning. `routes/gpu.py` builds it via `build_gpu_monitor()` — real whenever `GPU_DEVICE != cpu`.
  `used_memory_mb` is now **whole-card VRAM in use**, not just the memory this API handed out
  (`_manager.get_total_memory_used()`); the response also carries `utilization_percent`, `source`
  (`nvml` / `nvidia-smi` / `mock`) and a `devices[]` detail array. Verified: `{"device_count":1,
  "total_memory_mb":4096,"used_memory_mb":308,"source":"nvml"}` against `nvidia-smi` reading 258 MiB at the same
  moment.
  - `GpuMonitor(cpu_mode=True)` still returns the simulated 2 × 8192 MB figures — that is what the unit tests and
    any machine without a usable driver get. `reserve_memory`/`release_memory` stay bookkeeping-only and never
    affect a real device's numbers.
  - **This cannot destabilise training/leasing:** `ResourceScheduler._calculate_available_memory()` is purely
    `capacity − active leases` from the database and never calls the monitor.
  - Before this change the endpoint was hardcoded (`device_count() == 2`, `_MOCK_TOTAL_MB = 8192`, real branch
    `raise NotImplementedError`), so it reported "2 devices / 16384 MB" on a machine that has **1 GPU with 4095 MB**.

## GPU Training (this machine, verified running)

Hardware: **NVIDIA GeForce GTX 745** — Maxwell, compute capability **sm_50**, **4095 MB** VRAM, driver **560.94**
(CUDA 12.6). That combination constrains everything below.

- **PyTorch 2.15+ ships no Maxwell wheels.** CUDA 12.6 binaries were dropped from CD in 2.15 and CUDA 13.x
  requires sm_75+, so **2.14 is the last release with prebuilt Maxwell support**
  ([PyTorch notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)).
  This venv has `torch==2.14.0+cu126` / `torchvision==0.29.0+cu126`; `torch.cuda.get_arch_list()` contains
  `sm_50`, and both `matmul` and `conv2d` execute on the device (verified, `cuda available: True`).
- Install order matters — torch from the PyTorch index first, then ultralytics with an extra index so pip cannot
  replace it:
  `pip install -i <pypi-mirror> --extra-index-url https://download.pytorch.org/whl/cu126 "torch==2.14.0+cu126" "torchvision==0.29.0+cu126"`
  then `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "ultralytics>=8.0,<9"` (8.4.152 installed).
  The Aliyun PyPI mirror served a **corrupted `opencv-python` wheel** (sha256 mismatch) here; Tsinghua worked.
- **`gpu_resources` must contain a row for the device** or GPU training fails with
  `InsufficientGPUError: GPU device 0 not found`. `ResourceScheduler.ensure_gpu_resource()` is only ever called
  from tests, so a migrated database has zero rows — this one was seeded manually (`'0'`, capacity 4096, reserved 0).
- **Celery needs `--pool=solo` on Windows** (prefork/billiard is unsupported there):
  `python -m celery -A backend.app.workers.celery_app worker -Q training,evaluation,default --concurrency=1 --pool=solo`
  run from the repo root so `.env` resolves. Beat is **not** required for training: `routes/training.py:157`
  enqueues `run_training.delay()` directly.
- **Device selection** lives in `train_worker._resolve_training_device()`: the task's `training_config['device']`
  wins, otherwise `GPU_DEVICE` from `.env`, resolved through `YoloDataset.resolve_device()` (graceful CPU fallback
  when CUDA is missing). `TrainingCreatePage.tsx` sends **no** `device` field — so UI-created tasks depend on that
  fallback (they silently trained on CPU before).
- **`train_worker._build_trainer_config()`** fills `epochs`/`batch`/`imgsz`/`cache`/`patience` from
  `TRAINING_DEFAULT_EPOCHS` / `TRAINING_DEFAULT_BATCH_SIZE` / `TRAINING_DEFAULT_IMGSZ` / `TRAINING_DEFAULT_CACHE` /
  `TRAINING_DEFAULT_PATIENCE` when the task omits them, and injects the resolved device. Without it a UI task fell
  through to ultralytics' own `batch=16, imgsz=640`, which overflows a 4 GB card. `.env` here sets batch 8 /
  imgsz 416 / cache false / patience 30; verified by direct call — a UI-shaped config resolves to `device=0,
  epochs=100, batch=8, imgsz=416, cache=False, patience=30`, and explicit task values still win.
  - Use `_pick()` (an explicit `is None` test) rather than `… or default` for `cache`/`patience`: `False` and `0`
    are legitimate values and `or` would silently replace an explicit `cache=false` with the configured default.
  - `YoloTrainer.train_args` is built from `training_params.py`: `workers` and `amp` are taken from
    `LOCKED_ARGS` (a task cannot change them), everything else in the exposed surface comes from the task config.
- **Model-tree invariant: a training task without `parentModelNodeId` produces a NEW ROOT node.** The UI does send
  one (`TrainingCreatePage.tsx:54`, defaulting to `models[0]`), but hand-rolled API payloads that omit it litter the
  tree with extra roots — and `ModelsPage` assumes a single root. Pass the parent when scripting training tasks.
- VRAM budget on the 4 GB card: `batch=4, imgsz=416` peaked at **0.518 GB**; `batch=8, imgsz=416` reports
  **1.27 GB** in ultralytics' own `GPU_mem` column (whole-card `nvidia-smi` reads ~1.4-1.6 GB with the desktop's
  ~0.3 GB included). The repo defaults (`batch=16`, `imgsz=640`) will not fit — keep batches small or reduce `imgsz`.
- `nvidia-smi` on this box reports `temperature.gpu = 0` and `power.draw = [N/A]` — the driver exposes neither, so
  only `utilization.gpu`, `memory.used` and `clocks.sm` are usable for monitoring here.
- Per-epoch cost on this box (task `b91f635f`, example1, `batch=8`, `imgsz=416`, GPU): read it from the run's own
  `checkpoints/<task_id>/attempt_1/checkpoints/train/results.csv` — the `time` column is **cumulative seconds**.
  Measured **151-219 s/epoch, steady state ≈ 165 s/epoch (2.75 min)**, consistent across four runs (two A/B runs at
  165-170, one 30-epoch control at 164). `workers=0` and `amp=False` are hardcoded in `YoloTrainer`, which suits
  Maxwell. The training-create form derives its estimate from `MINUTES_PER_EPOCH = 2.75` /
  `MEASURED_TRAIN_IMAGES = 1083` in `TrainingCreatePage.tsx` (scaled by the selected datasets' `trainCount`) —
  **re-calibrate these (or derive them from completed-task history) whenever the GPU, batch, imgsz or dataset size
  change**, since a wrong estimate is worse than none.
  - **An earlier revision of this file claimed 87.8 s/epoch — that was wrong** and the error is instructive: the
    100-epoch task's 8779 s wall clock was divided by the *configured* 100 epochs, but `patience=10` had
    **early-stopped it at epoch 52** (best epoch 42), so `results.csv` only had 52 rows. Always divide by the row
    count, and beware that "configured epochs" ≠ "epochs actually run".
  - **`results.csv` `time` is wall-clock based, so a system sleep inflates one epoch enormously.** A run here lost
    2 h 15 m inside a single epoch (8265 s instead of ~160 s) because the machine slept (see "Sleep kills the
    clock" below); excluding that row the same run was 164 s/epoch. When comparing runs, drop such outliers
    instead of averaging over them.
- **Early stopping fires surprisingly early and silently truncates runs.** `patience` was hardcoded at 10
  (`YoloTrainer`: `self.config.get("patience", 10)`), and on example1 the validation metric oscillates for the
  first ~15 epochs before it starts climbing: measured best-epoch 42 → stop at 52 (the 100-epoch run) and
  best-epoch 1 → stop at **11** (a 30-epoch run whose mAP50-95 never beat epoch 1's 0.0983). Two runs with
  *identical* config except `cache` produced 11 vs 30 epochs purely from that noise, and the 11-epoch model ended
  at mAP50 0.155 versus 0.212 for the full one — i.e. **the configured epoch count does not determine the outcome,
  `patience` does.** It is now configurable: `TRAINING_DEFAULT_PATIENCE` (`.env`, currently **30**) fills
  `patience` for tasks that omit it, task-level values still win.
- **`cache` buys nothing here — verified, do not re-enable it by hope.** A 30-epoch A/B on example1 (identical
  config, only `cache` flipped; `args.yaml` confirmed `cache: true` reached ultralytics) gave steady-state
  167.2 s/epoch with `cache=false` vs 169.7 s/epoch with `cache=true`, and the two mAP50-95 curves through epoch
  11 were indistinguishable. The apparent 10 % win for `cache=true` on epochs 1-5 was a confound: the `false` run
  was first and paid the cold-file cost, the `true` run inherited a warm Windows page cache. The dataset is only
  ~1 GB, so the OS file cache already does this job. It stays configurable (`TRAINING_DEFAULT_CACHE`, default
  `false`) for bigger datasets.
- **`workers=0` is deliberate, not an oversight.** Ultralytics' DataLoader spawns processes on `workers > 0`, and
  training runs inside a Celery `--pool=solo` worker on Windows (see the `freeze_support()` failure that broke a
  `val()` cross-check). Do not raise it without a throwaway run proving the worker survives.
- **Bottleneck of a full run, measured over 4.1 h (1473 samples, `aip-data\_watch_train.py`)**: GPU
  `utilization.gpu` mean **78 %**, median **95 %**, p10 31 %, p90 99 %, and it sat **below 50 % for only 15.8 % of
  the time**; the worker process consumed **1.0 core of 32** (max 1.27) with the SM clock steady at 1032 MHz. So
  the GTX 745 is the limit, and the idle ~16 % is concentrated at the epoch/validation boundaries
  (single-threaded dataloading with `workers=0`, visible as 2-3 % dips in any short sample). Together with the
  `cache` result above (no gain) there is **no cheap throughput win left on this machine** — ~165 s/epoch is the
  floor for batch 8 / imgsz 416. Re-measure with `aip-data\_watch_train.py <task_id>` (samples
  util/VRAM/SM clock/CPU every 10 s; writes `logs/train_watch_<task8>.{csv,json}` at the end) before spending
  effort here.
- **A full 100-epoch run finished in 16508 s (4.59 h), 165.1 s/epoch** (median 163.7, min 151.1, max 178.3; task
  `952af3a8` → model node **M003**), which confirms the 2.75 min/epoch calibration and that `patience=30` did
  *not* truncate it (best epoch 78, so all 100 ran). Training-val metrics improved a lot over the old
  `patience=10` model: M003 mAP50 **0.2448** / mAP50-95 0.1477 vs M002 0.1569 / 0.0993 (best checkpoint at epoch
  78; final epoch 0.2341 / 0.1366). The platform's own gate metric moved far less (0.1728 vs 0.1673), because it
  is computed on example1's **18-image test split, which is not stratified**: it contains only classes 0/1/2
  (2/53/112 boxes) and **zero class-3 boxes**, even though class 3 is 7021 of train's 7813 labels. Treat the auto
  gate as a smoke test, not as the measure of model quality.

- Verified end to end twice: explicit `device=0` task `f04c7b1e` → model node **M002**, and a UI-shaped task with
  no `device` field `7926d6ae` → **M003**; both `completed`, both printing
  `CUDA:0 (NVIDIA GeForce GTX 745, 4096MiB)`, with all GPU leases released afterwards.
- **Two bugs had to be fixed before training worked on any device** (both in `train_worker.py`): the merged dataset
  manifest dropped each snapshot's `source_dir`, so `YoloDataset.from_manifest` resolved relative entries against
  the process CWD and died with `FileNotFoundError: 'train\images\...'`; and the trainer received the raw task
  config, so a resolved GPU device never reached ultralytics (it held a GPU lease while training on CPU).
- **Evaluation metrics now come from ultralytics' own validator** — `YoloEvaluator.validate_on_split()` calls
  `model.val(split=…, device=…, workers=0, plots=False)`. The previous in-memory path computed AP from boxes
  already filtered at `conf >= 0.25` and reported `mAP50_95` from a **single** IoU=0.95 pass, so it understated
  models badly: for M002 on the same test split it stored `mAP50 = 0.0 / recall = 0.0` while `ultralytics val`
  reports `mAP50 = 0.1673 / recall = 0.5717`. The old `evaluate()` is kept for the in-memory unit tests but is
  **deprecated for gating** (documented in its docstring).
  - Verified end to end: the platform now stores `mAP50 = 0.167318, mAP50_95 = 0.104353, precision = 0.0988,
    recall = 0.5717, num_images = 18, num_ground_truths = 167` — matching `ultralytics val` digit for digit.
  - Evaluation runs on **CPU on purpose**: `YoloEvaluator(device="cpu")`, and `device` is finally passed through to
    predict/val (it used to be a dead parameter, so evaluation silently used the GPU *without* taking a lease and
    would have raced training for the single 4 GB card).
  - The worker writes a per-evaluation `data.yaml` (`<LOG_DIR>/evaluations/<id>/data.yaml`) whose split entry is a
    **text file of absolute image paths** — it does not assume any images/labels layout.
- **The quality gate and the human-review workflow were REMOVED on request (2026-09-16).** The product owner's
  call: a threshold that turns a continuous metric into 通过/未通过 adds no decision value, and nobody owns the
  reviewer role ("这个平台没有审核机制") — leaving either in place invites misreading.
  - **Frontend**: the training form no longer asks for 质量门槛 (nor sends `evaluationPolicyJson`), and the
    evaluations page no longer has a status column, verdict badge, status filter or 待人工验收 count — the list
    now shows mAP50 / mAP50-95 / 测试集 / 时间, and the drawer shows metrics only.
  - **Backend**: `POST /api/evaluations/{id}/review` is gone (verified live: **404**, and `GET /openapi.json`
    dropped **64 → 63** paths), `EvaluationResponse` no longer carries `human_status`, and
    `evaluation_service.record_human_review()` is a hard failure (`raise ValueError("人工复核已移除…")`) so any
    out-of-tree caller fails loudly instead of believing it reviewed something. `TestHumanReviewIsGone` in
    `tests/integration/test_evaluation_flow.py` locks both the 404 and the raise.
  - **What deliberately stays**: `evaluations.auto_status` is *not* a verdict — it is the evaluation pipeline
    state (`pending` → `running` → `passed`/`failed`) used by `evaluation_worker`, the `sweep-pending-evaluations`
    scheduler job and `model_diff` (which picks baseline metrics from `passed` rows).
  - **The five review columns are gone from the schema too** (migration **`0005_drop_evaluation_review`**, applied
    here: `alembic_version = 0005`): `human_status`, `reviewer`, `human_conclusion`, `human_comments`,
    `reviewed_at` plus the `evaluation_human_status` CHECK constraint. The constraint had landed double-prefixed
    as `ck_evaluations_ck_evaluations_evaluation_human_status` (0001 names it `ck_evaluations_…` and the metadata
    naming convention adds its own prefix), so `0005` drops every plausible spelling with
    `DROP CONSTRAINT IF EXISTS`. Verified afterwards by diffing `inspect(engine)` against `Base.metadata`: zero
    columns differ in either direction, and `GET /api/evaluations` still serves both rows with their metrics.
  - Test-side fallout, all handled: `build_fixtures.FixtureBuilder.evaluation()` no longer takes `human_status`
    (it is the shared fixture — leaving it would have broken far more than the drift baseline) and
    `test_model_diff` / `test_closed_loop` / `api/test_evaluations` / `test_schema_constraints` lost their
    references. The affected-test run afterwards (`test_evaluation_flow`, `api/test_evaluations`,
    `test_schema_constraints`, `test_model_diff`, `test_closed_loop`) gives **82 failed / 59 passed / 2 skipped /
    3 errors**, and every single failure is the pre-existing `model_nodes.code` drift — none mentions the removed
    feature.
  - **`model_nodes.status` is consequently always `candidate` and is no longer displayed anywhere**: the models
    page lost its status filter and root badge, the model-detail page lost its 状态 row, and the dashboard's
    「候选模型」 KPI became 「模型」 (a lineage-node count) while its evaluation badge shows the mAP50 instead of the
    old verdict. The column itself stays — the worker still writes `candidate` when it creates a node.
  - Historical note, kept because it documents the old fallback: **the all-zero threshold made every model "pass"
    (`0.0 >= 0.0`)** — verified on M002: `passed` at threshold 0 before, `failed` at 0.1 while its metric was
    still 0.0, then `passed` at 0.1 once the metric was corrected to 0.167. That whole path is now unreachable
    from the UI, and the threshold no longer influences what a user sees.
- Evaluation still takes no GPU lease (`ResourceScheduler(cpu_mode=True)` and no `reserve_gpu` call), which is why
  keeping it on CPU is the safe choice; giving it a lease would be required before ever moving it to the GPU.

## Known Test Failures at `4bcebb77`

A full run with light deps gives **544 passed / 154 failed / 3 skipped / 14 errors / 30 deselected (~142 s cold)**.
Failure causes, classified from `--tb=line` output (not guessed):

- **132 of 154 failures: `sqlalchemy.exc.IntegrityError: NOT NULL constraint failed: model_nodes.code`.**
  `ModelNode.code` (`backend/app/models/model_node.py:29`) is `nullable=False` with **no ORM default and no
  `before_insert` hook** — the only generator is `repositories/model_repository.py:_generate_model_code`, called
  from `create_model_node()`. Tests that build rows directly therefore insert `code=NULL`.
  The column became NOT NULL in `50fae1c3` ("model code field", 2026-08-29, migration `0003_model_code`), while
  the fixtures were last touched in `c6791075` (2026-08-22) and never updated — i.e. **this is committed drift,
  reproducible without regard to the dependency set**.
  Worst hit: `tests/e2e/test_closed_loop.py` (28), `tests/integration/test_schema_constraints.py` (26),
  `tests/unit/test_model_diff.py` (19, via `_make_model`), `tests/integration/test_evaluation_flow.py` (12),
  `tests/integration/test_binding_releases.py` (10), plus `tests/fixtures/build_fixtures.py` (5 `ModelNode(`
  sites, none passing `code`). Most of the 14 errors are async-test setup failures with the same cause.
  Fix direction: pass `code=` in the fixtures, or route fixture creation through `create_model_node()`.
- **3 failures/errors from the light env** (expected here, will pass once torch+ultralytics are installed):
  collection error in `tests/unit/test_yolo_engine.py` and `tests/integration/test_inference_flow.py`, and
  `tests/unit/test_yolo_trainer.py:46` (`ModuleNotFoundError: ultralytics`).
- **The rest are independent code/test drift**, each reproducible:
  - `health/live` returns `{"status": "ok"}` while `tests/unit/test_health.py:88` and
    `tests/integration/api/test_health.py:36` assert `{"status": "alive"}` (route shadowing described above).
  - `tests/unit/test_runtime_generation.py` — `AttributeError: 'ModelManager' object has no attribute 'load_model'`.
  - `tests/unit/test_frontend_dockerfile.py:8` expects `COPY nginx.conf /tmp/nginx.conf`; the Dockerfile now uses
    `/etc/nginx/nginx.conf`.
  - `tests/integration/test_dataset_snapshots.py` — `assert False` and `KeyError: 'image_stored_path'`.
  - `tests/integration/test_training_task_flow.py:308` — a monotonic-time assertion comparing two datetimes that
    are 90 minutes apart.

## Env & Config Quirks

- `backend/app/config.py:Settings` uses `env_file=".env"` with `utf-8-sig` and `extra="ignore"`. If `POSTGRES_URL` is empty it is built from `POSTGRES_HOST/PORT/DB/USER/PASSWORD` via `sqlalchemy.URL.create` (password URL-encoded). `alembic/env.py` reads the same Settings.
- Default `PLATFORM_API_KEY=change-me` (header `X-API-Key` only — never in URL). Enforced: a request without it returns 401 `Invalid or missing API key`. Frontend sends `VITE_API_KEY` (defaults to `change-me`) in `createApi.ts`.
- `GPU_READY`/`WORKER_READY` are boolean readiness overrides consumed by `backend/app/db.py:default_health_checks` — `/health` returns `ready`/`degraded`/`unhealthy` based on postgres+redis (unhealthy) vs gpu+worker (degraded).
- `HOST_*_DIR` in `.env` bind-mounts `./models,./datasets,./logs,./checkpoints` into containers at `/data/*`. `infra/.env.example` additionally defines `HOST_POSTGRES_DIR`, `HOST_REDIS_DIR`, `GPU_COUNT`, `NVIDIA_VISIBLE_DEVICES`, `NVIDIA_DRIVER_CAPABILITIES` (`.env.example` carries a BOM — expected, it is in verify_bom's suffix list).

## BOM / File Encoding

- `backend/scripts/verify_bom.py` requires a UTF-8 BOM (`EF BB BF`) on `Dockerfile` plus every file under `backend/`, `frontend/`, `infra/` with suffix `.cfg .conf .css .html .ini .js .json .md .py .toml .ts .tsx .txt .yml .yaml .example` (skipping `node_modules`, `__pycache__`, `.pytest_cache`, `test-results`, `dist`).
- **Executed at `4bcebb77`: 188 files carry a BOM, 34 do not** (exit code 1). Always re-run the script for the live list.
  - After the 2026-09-16 changes the count is **33**: editing `frontend/src/lib/mockApi.ts` with `utf-8-sig` gave it the
    BOM the convention requires all along. Nothing else changed (re-add a BOM after every `edit`, which strips it).
- Files currently missing a BOM that you are most likely to touch: `backend/pyproject.toml`, `backend/alembic.ini`, `backend/alembic/versions/{0002_training_cancellation_request,0003_model_code}.py`, `backend/app/api/routes/gpu.py`, `backend/app/workers/db_session.py`, `backend/tests/{integration/api/test_gpu.py,integration/api/test_training_logs.py,integration/test_evaluation_integration.py,unit/test_dataset_background_import.py,unit/test_train_worker.py,unit/test_worker_fixes.py,unit/test_yolo_evaluator.py}`, `frontend/{Dockerfile,nginx.conf,package.json,package-lock.json,playwright.config.ts,tsconfig.json,vite.config.ts}`, `frontend/src/{main.tsx,vite-env.d.ts,app/App.tsx,app/App.css,lib/api.ts,lib/mockApi.ts,lib/toast.tsx,lib/LoadingSpinner.tsx,features/bindings/BindingsPage.tsx,features/releases/ReleasesPage.tsx,features/resources/ResourcesPage.tsx}`, `frontend/tests/admin-flows.spec.ts`, `infra/docker-compose.yml`, `infra/checkpoints/90fed2a0-*/attempt_1/checkpoints/train/args.yaml`.
- Both `backend/pyproject.toml` (`5B 62 75` = `[bu`) and `backend/alembic.ini` (`5B 61 6C` = `[al`) currently have **no** BOM. Keep passing `-c NUL`: adding the mandated BOM to either file breaks `pytest`/`alembic` config parsing (TOML/INI reject a BOM), and the Dockerfile's `COPY pyproject.toml alembic.ini` + `pip install .` does not strip one.
  - With `-c NUL`, pytest also ignores `[tool.pytest.ini_options] addopts = "-q"` and `pythonpath = ["."]`. Imports still resolve because `backend/__init__.py` exists, so pytest prepends the repo root for the `backend.tests.*` package chain.
- **Exception: `frontend/nginx.conf` must NOT have BOM** (verified: first bytes `65 76 65` = `eve`) — nginx rejects `﻿events` as unknown directive. Remove BOM after editing: `python -c "d=open('frontend/nginx.conf','rb').read(); open('frontend/nginx.conf','wb').write(d[3:] if d[:3]==b'\xef\xbb\xbf' else d)"`.
- BOM check order: edit file → add BOM → if `nginx.conf` remove BOM → rebuild.
- Tooling note: the `edit` tool strips a file-level BOM, so re-add it after an edit if the file had one (root `AGENTS.md` and `ai-model-platform-design.md` both carry a BOM).

## Testing & Fixtures

- `backend/tests/conftest.py` only registers the `slow` marker — those tests are **not** auto-skipped; use `-m "not slow"` to exclude them (30 tests at this commit).
- Fixtures: `backend/tests/fixtures/build_fixtures.py:FixtureBuilder` (deterministic UUIDs via sha256 seed) and `build_real_fixtures.py`. Unit `backend/tests/unit/` (31 files), integration `backend/tests/integration/` (+ `integration/api/`), e2e `backend/tests/e2e/test_closed_loop.py`, perf `backend/tests/performance/{test_inference_perf,test_training_perf}.py`.
- `backend/tests/integration/conftest.py` builds an in-memory SQLite engine (`StaticPool`) and `create_app` with overrides, so most integration tests need no Postgres/Redis.
- Frontend Playwright `frontend/playwright.config.ts`: `baseURL http://localhost:3000`, `webServer: npx vite --mode mock`, `reuseExistingServer: !CI`.
- UI self-test: prefer driving the real page (Chrome DevTools MCP) over Playwright — open `8080`/`3000`, check the console, exercise interactions, read the DOM, report what was verified. There is no `CLAUDE.md` in this repo; this section is the standing instruction.

## Data Model (non-obvious)

- `model_nodes` has `code` (`M001`, `M002` … via `model_repository.py:_generate_model_code` — `max(code)` + 1, zero-padded), `nullable=False` + `unique=True`, added by migration `0003_model_code`; it is used in URLs (`GET /api/models/M001`) and tree labels. `POST /api/models/{model_id}/infer` resolves the path segment as a UUID first and falls back to `code`. `model_bindings` stable IDs; `model_binding_releases` monotonic `revision_no` per binding (`UniqueConstraint(binding_id, revision_no)`, `CheckConstraint("revision_no > 0")`); `dataset_snapshots` immutable with `manifest_hash`; `training_tasks`/`training_attempts` with `attempt_lease` fencing (`workers/attempt_lease.py`); `evaluations` (pipeline status only — see the 2026-09-16 note on the removed review columns; `metrics.per_class` is shown in the eval detail drawer); `gpu_resource_leases` Postgres-backed; `model_residency_plans`.
- `backend/app/models/base.py` installs an immutability guard through four Session-level listeners
  (`loaded_as_persistent`, `before_flush`, `before_commit`, `after_flush_postexec`) — it snapshots sealed fields
  and raises `ImmutableFieldError` on mutation. There is **no** insert-time hook that fills defaults such as
  `code`.

## Migrations

- Heads in `backend/alembic/versions/`: `0000_empty` → `0001_initial_schema` → `0002_training_cancellation_request` → `0003_model_code` (adds `code` nullable, backfills `M001…`, then sets NOT NULL + unique) → **`0004_orm_schema_repair`** (adds `model_nodes.name`, `datasets.source_path`, and creates the `alerts` + `backup_records` tables no earlier revision created) → **`0005_drop_evaluation_review`** (drops the five human-review columns and their CHECK constraint, and is the current head: `alembic_version = 0005`).
- **The chain was incomplete before `0004`**: `alembic upgrade head` used to leave a schema the ORM could not query, so a freshly migrated database returned HTTP 500 for `GET /api/models`, `GET /api/datasets`, `GET /api/alerts` and `GET /api/backup/list` (`UndefinedColumn: model_nodes.name`, `UndefinedColumn: datasets.source_path`, missing `alerts` / `backup_records`). The integration tests never caught it because they build the schema with `Base.metadata.create_all()` instead of running migrations. Diff `inspect(engine)` against `Base.metadata` before trusting that migrations are complete.
- `alembic/env.py` imports `from app.config …` / `from app.models …` — the `app` package, **not** `backend.app`. From the repo root you must therefore put `backend` on `PYTHONPATH`:
  `$env:PYTHONPATH="$PWD\backend"; .venv\Scripts\alembic.exe -c backend\alembic.ini upgrade head`
  (Running with CWD=`backend/` also works, but then `Settings` looks for `.env` in `backend/` rather than the repo root.)

## Training / Worker Gotchas

- Celery task `platform.training.run_training` takes task UUID; beat schedule in `backend/app/workers/celery_app.py` (`sweep-pending-evaluations`, `reconcile-leases`, `sweep-queued-tasks` every 60s) with `task_acks_late`, `task_reject_on_worker_lost`, `worker_prefetch_multiplier=1`; queues are routed per `platform.training.*` / `platform.evaluation.*` / `platform.scheduler.*`. `celery_app` imports the three worker modules at the bottom of the file, so importing it pulls them in.
- Worker must use `backend/app/workers/db_session.py:worker_session()` (own engine/session), not request-scoped `get_db`. `failed` is terminal, no auto-retry; cancellation is cooperative (API sets flag, worker checks before/after). CPU mode skips GPU lease; GPU mode requires `torch.cuda.is_available()`.
- Heavy imports are deliberately lazy (inside functions) in `train_worker`, `db.check_gpu`, `yolo_trainer`,
  `yolo_validator`, `yolo_evaluator` and the route handlers. **The one exception is
  `backend/app/inference/yolo_engine.py:11`, which imports `ultralytics` at module level** — that is what forces
  torch/ultralytics to be installed just to collect two test modules.
- **A long-lived worker transaction used to make cancellation impossible — fixed, keep it that way.**
  `run_training` called `attempt_heartbeat(session, …)` (a `SELECT … FOR UPDATE` on the attempt row) and only
  committed after `trainer.train()` returned, so for the entire run the worker backend sat `idle in transaction`
  holding row locks on `training_tasks` / `training_attempts`. Measured consequences (`pg_stat_activity`):
  `POST /api/training/tasks/{id}/cancel` **blocked >120 s** and returned nothing, `cancellation_requested` stayed
  `f` — a UI cancel click is silently lost; the heartbeat thread blocked on its own `FOR UPDATE` (so
  `heartbeat_at` never advanced — it is `NULL` on every `gpu_resource_leases` row); and the scheduler's lease
  reconciliation was frozen too. Fix: commit (roll back on error) immediately after the first heartbeat, before
  the long call, capturing `attempt.id`/`lease_token`/lease ids into locals first because `commit()` expires the
  ORM objects. Verified after the fix: `heartbeat_at` advancing 30 s apart, and `cancel` returning in **0.1 s**
  with `status=cancelled, cancellation_requested=t`. **Any future write to the main worker session before or
  during `trainer.train()` must be committed, or this regression comes straight back.**
- **A stale `active` row in `gpu_resource_leases` blocks every later GPU run.** `_calculate_available_memory()`
  is `capacity − active leases`, so one orphan lease (4096 MB on a 4096 MB card) makes the next task fail within
  a second with `failure_reason = GPU 0 has 0MB available, need 4096MB`. Symptom + remedy:
  `SELECT status, reserved_memory_mb FROM gpu_resource_leases WHERE status='active';` then
  `UPDATE gpu_resource_leases SET status='released' WHERE status='active'; UPDATE gpu_resources SET reserved_memory_mb=0;`
- **Do not `DELETE /api/training/tasks/{id}` while its worker is still finishing.** The row disappears mid-flight
  and the worker raises `ObjectDeletedError: Instance '<TrainingTask …>' has been deleted`, which aborts the
  finish path *before* the lease release runs — that is exactly how the orphan lease above appeared. Wait for a
  terminal status (the worker logs `Task … succeeded`) before deleting, and note that a lease whose owning attempt
  no longer exists is never reaped on its own: nothing sweeps `active` leases whose attempt is gone.
- **The exposed training-parameter surface is small on purpose, and it is enforced in code.**
  `backend/app/training/training_params.py` is the single source of truth: it owns the augmentation levels
  (`off` / `default` / `strong` → the ~14 raw ultralytics coefficients), the bounds for
  `epochs`/`imgsz`/`batch`/`patience`, the raw-coefficient whitelist for scripted experiments, the locked
  arguments and the VRAM guard. `YoloTrainer` builds `train_args` from it, `routes/training.py` validates every
  create request with it (HTTP 400 + readable Chinese message), and `train_worker._build_trainer_config`
  re-normalizes defensively before training.
  - **Exposed**: `epochs`, `imgsz` (multiple of 32), `batch`, `patience`, `cache`, `augmentation`, `recipe`
    (free-text label kept for the snapshot), plus the raw coefficients in `RAW_ARG_BOUNDS` for API/script use.
    **Locked**: `workers` (stays 0) and `amp` (stays False) — a task may echo the locked value, anything else is
    a 400. **Not exposed at all** (unreachable even from the API): `optimizer`, `lr0`, `cos_lr`, `freeze`,
    loss weights, NMS settings.
  - The UI therefore shows three **recipes** (快速验证 / 标准 / 高精度 — each a complete parameter set) plus a
    collapsed 「高级参数」 panel with only 轮数 / 分辨率 / 批大小 / 早停 / 增广. Editing any field marks the run
    `recipe: "custom"`, so the snapshot never claims a recipe it did not run.
  - **VRAM guard**: `estimate_train_vram_mb(imgsz, batch) = 160 MB x (imgsz/416)^2 x batch`, anchored on the
    measured `imgsz 416 + batch 8 = 1280 MB`. The API limit is `gpu_memory_reservation_mb - 800` (3296 MB here);
    above 75 % of it the task is still accepted but flagged. Verified: `imgsz 1280 + batch 16` → 400
    `预计占用 24237 MB … 请把 batch 降到 2`, `imgsz 500` → 400 `必须是 32 的倍数`, `workers: 4` → 400
    `参数 workers 不可修改`. The worker only *warns* on the same input (it may be running a task created before
    the rule existed), which is why the API is the real gate.
  - **Parameter snapshot lives on the task row and is written at CREATION, not by the worker.**
    `install_immutable_guard(TrainingTask, {… "training_config_json" …})` seals those fields on insert, so the
    worker *cannot* back-fill what a run actually used — attempting it raises
    `ImmutableFieldError: TrainingTask fields are immutable after sealing: training_config_json` (observed; my
    first implementation lost the snapshot silently for exactly this reason). The create route therefore calls
    `resolve_config(payload.get_config(), defaults=defaults_from_settings(settings))` and stores the **complete**
    snapshot up front, which is also the right semantics: a run's definition is fixed when it is created. Verified
    live — a payload carrying only `{"epochs": 5}` came back as
    `{"epochs":5,"imgsz":416,"batch":8,"patience":30,"cache":false,"augmentation":"default","device":"0"}`.
    The worker still resolves the same way (real `.env` values for legacy tasks) and logs
    `resolved training parameters for task …` + `ultralytics train args: {…}` for every run.
  - **Don't couple an unrelated write to the heartbeat block.** The main worker session's `attempt_heartbeat()`
    can raise (`with_for_update`), and my first version put the snapshot write inside that `try`, so its
    `rollback()` discarded the write. Whatever else must be persisted there needs its own `commit()`, or it must
    happen in the route.
  - Per-epoch time is a function of `imgsz` **only** (batch changes VRAM and gradient noise, not total compute):
    the UI estimate is `2.75 min x (imgsz/416)^2 x (train_images/1083)`, so 640 costs ~2.4x a 416 run.
  - Rough edge, not fixed: cancelling a task before its worker reaches the training call still ends as
    `failed` (the early-return path calls `fail_attempt`), not `cancelled`. The DB says
    `cancellation_requested = t` while `status = failed`, which reads like a crash in the UI. Fix direction: after
    a failed result, check `cancellation_requested` and call `cancel_task` instead.
- **Route changes need an API restart — restarting the worker is not enough.** Observed the hard way: after
  adding the create-time validation I restarted only `AIP-Worker`, so the *old* uvicorn code accepted
  `imgsz 1280 + batch 16` and it started training on the 4 GB card. uvicorn serves `api/routes/*`, the Celery
  worker only reloads its own modules. Restart `AIP-API` (kill the uvicorn pids, then
  `Start-ScheduledTask -TaskName AIP-API`) whenever you touch routes, and verify with a live HTTP probe — the
  `/health` endpoint says nothing about which route code is loaded.

## Frontend API Note

- `frontend/src/lib/mockApi.ts` is the mock; `createApi()` picks it when `VITE_USE_MOCKS=true`. Real API base is `VITE_API_URL` (empty = relative). Nginx `frontend/nginx.conf` and Vite proxy both forward `/api`.
- `frontend/src/features/api/ApiTestPage.tsx` — per-model inference test (`POST /api/models/{code}/infer` returns `overlay_image` + `detections`). Each model gets a card; left-right layout (param docs vs. result JSON + image). Uses `code` (e.g. `M001`) not UUID.
- Small gotcha: `createApi.ts` camelizes response keys for most endpoints (`camelizeKeys`), but sends snake_case bodies on the GPU/infer paths (`model_name`, `memory_mb`) — match the existing shape when adding calls.

## Dataset Import Pipeline

- Two-step upload: `POST /api/datasets/upload` (multipart) only stores the file as `{DATASET_DIR}/{token_hex(8)}_{filename}` and returns `{filename, file_path, size, sha256}` — no DB row, no snapshot. It rejects non `.zip/.tar.gz/.tgz` (400) and anything over `_MAX_DATASET_BYTES` = 2 GB (413).
- `POST /api/datasets` (async): inserts the `Dataset` row from `source_path` (accepts `source_path` or camelCase `sourcePath`) → `db.commit()` → schedules `_build_snapshot_background` via `BackgroundTasks` → returns `201` with `0/0/0` counts immediately.
- Background task: extracts zip → validates via `_locate_yaml` / `_parse_data_yaml` → `create_dataset_snapshot` → inserts `DatasetSnapshot` + `LabelSchema` → frontend polling detects completion.
- `_build_snapshot_background` uses `worker_session()` (separate DB connection), not request-scoped `get_db`. Must `commit()` before scheduling to avoid invisible rows.
- Frontend polls `GET /api/datasets` every 3s after upload, displays `解析中` badge for rows with `imageCount 0 && validationStatus "-" && !latestSnapshotId`.
- `nginx.conf` `proxy_read_timeout 600s` required — default 300s causes 504 on 700MB+ archives (`client_max_body_size 2g` on `/api/`, `0` globally).
- `_locate_yaml` supports `data.yaml`, any `*.yaml` at root, and `*.yaml` in one-level subdirs (e.g., `crack-seg.yaml`).
- `_parse_data_yaml` infers `nc` from `len(names)` when only a dict `names: {0: crack}` is present (no explicit `nc` field).
- Annotation support lives in the same router: `POST /{dataset_id}/images`, `GET|PUT|DELETE /{dataset_id}/annotations/{image_id}`, `GET /{dataset_id}/image_list`, `GET /{dataset_id}/label_classes`, `POST /{dataset_id}/batch`; explicit snapshots via `POST /{dataset_id}/snapshot` and `POST /snapshots` (both 201).

## Model Tree

- Uses `dagre` (dependency in `frontend/package.json`) for DAG layout of model lineage.
- Delete button (×) only shows for leaf nodes (`byParent(n.id).length === 0` in `ModelsPage.tsx`).
- Backend `DELETE /api/models/{id}` cascades dependent evaluations before deleting the model node.

## Docs Index

`docs/operations/first-run.md` is the setup entry point (referenced by `infra/docker-compose.yml`). Also available: `docs/deployment/{installation,configuration,operations,troubleshooting}.md`, `docs/performance/benchmark.md`, `docs/production/checklist.md`, and the design history under `docs/superpowers/plans/` (7 dated plans) and `docs/superpowers/specs/` (2 dated specs). Repo root has `ai-model-platform-design.md` (the overall design doc) and `platform-prototype.html` (static UI prototype), plus `check_datasets.py` for inspecting `dataset/`.
