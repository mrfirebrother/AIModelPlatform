# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AI model platform for road inspection (道路智能巡查养护): YOLO model lifecycle management — datasets, training, evaluation, release, a model-lineage tree, and per-model inference testing — with a Chinese-language admin UI.

- **Backend**: Python 3.11 (strictly `>=3.11,<3.12`), FastAPI + SQLAlchemy 2 + Alembic + Celery/Redis + Ultralytics YOLO. Entrypoint `backend.app.main:app`.
- **Frontend**: React 18 + TypeScript + Vite (`frontend/`), React Router, Playwright tests.
- **Infra**: runs natively on Windows (PostgreSQL + Redis services, Scheduled Tasks) — Docker compose is an alternative, not the default here.
- **Deployment target**: sibling package `../rust-analysis-api-exe` — a packaged Rust API (`rust-analysis-api.exe`) that serves a trained model (`models/best.pt`, config in `config.json`) on `http://127.0.0.1:8001`. Models released from this platform are shipped there.

## Deep reference

`AGENTS.md` (553 lines) is the authoritative, machine-verified guide: exact test-failure breakdown, GPU training measurements on this machine, deployment state, data-model quirks. **Read the relevant section before touching training, migrations, or deployment.** Its stated baseline is `4bcebb77`, but later sections cover HEAD (quality-gate removal, migration `0005`, Windows deploy scripts).

## Commands (verified on this machine)

### Python trap (read first)

Bare `python`/`python3` hit the Microsoft Store stub and fail. Use `.venv\Scripts\python.exe` or `py -3.11`. `git` is also not on PATH (full path: `C:\Program Files\Git\cmd\git.exe`). `.venv` exists (3.11.9) with the full dep set **including torch/ultralytics** now installed for GPU training.

### Backend (run from repo root — `Settings` reads `.env` relative to CWD)

```bash
# Tests: -c NUL is mandatory (BOM in pyproject.toml/alembic.ini breaks config parsing);
# --continue-on-collection-errors is mandatory (yolo_engine.py imports ultralytics at module level)
.venv/Scripts/python.exe -m pytest -c NUL -p no:cacheprovider backend/tests -q --continue-on-collection-errors
# Single test / slow tests excluded
.venv/Scripts/python.exe -m pytest -c NUL -p no:cacheprovider backend/tests/unit/test_health.py -k test_name -q
.venv/Scripts/python.exe -m pytest -c NUL -p no:cacheprovider backend/tests -q -m "not slow" --continue-on-collection-errors
# Compile + BOM checks
.venv/Scripts/python.exe -m compileall backend
.venv/Scripts/python.exe backend/scripts/verify_bom.py
# Migrations (needs DB env; alembic/env.py imports `app.*`, so backend must be on PYTHONPATH)
PYTHONPATH="$PWD/backend" .venv/Scripts/alembic.exe -c backend/alembic.ini upgrade head
# API
.venv/Scripts/python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
# Celery worker — --pool=solo is mandatory on Windows
.venv/Scripts/python.exe -m celery -A backend.app.workers.celery_app worker -Q training,evaluation,default --concurrency=1 --pool=solo
```

### Frontend

```bash
cd frontend
npm run dev      # Vite :3000, proxies /api -> 127.0.0.1:8000
npm run build    # tsc -b && vite build — then: git restore frontend/tsconfig.tsbuildinfo (tsc -b dirties it)
npm run test     # Playwright (auto-starts vite --mode mock)
```

Use `http://localhost:3000` (Vite binds `::1`; `127.0.0.1:3000` does not connect). Mocks are enabled by `VITE_USE_MOCKS=true`, not by `--mode mock`.

### Deployment (canonical, replaces the stale `start.bat`)

`start.bat` hardcodes `F:\Work\EStructure\AIModelPlatform` from the original author's machine — do not use it. Use:

```powershell
powershell -ExecutionPolicy Bypass -File infra\windows\deploy.ps1 install   # idempotent, needs admin
powershell -ExecutionPolicy Bypass -File infra\windows\deploy.ps1 status|restart|logs|uninstall
```

This installs PostgreSQL/Redis as Windows services and AIP-API / AIP-Worker / AIP-Beat / AIP-Frontend as SYSTEM Scheduled Tasks with a 1-minute watchdog. Docker compose (`infra/docker-compose.yml`, `docs/operations/first-run.md`) is untested on this machine — no Docker installed.

## Architecture

### Backend (`backend/app/`)

- `create_app()` in `main.py` includes 17 routers (`api/routes/*.py`); the real endpoint count is `GET /openapi.json` → 63 paths, not `len(app.routes)` (routers stay nested as `_IncludedRouter`). All API access requires header `X-API-Key` (env `PLATFORM_API_KEY`).
- **Health quirk**: the effective routes come from `observability/health_probes.py` (included first), which **shadows** `main.py`'s inline `/health/live` (dead code). `/health/live` returns `{"status":"ok"}`, not `"alive"` — some tests assert the wrong value.
- Layering: `models` (SQLAlchemy) / `schemas` / `repositories` / `services` / `api/routes`; `workers` (Celery tasks), `training` (params + YoloTrainer), `evaluation` (ultralytics validator), `inference` (yolo_engine), `runtime` (GPU monitor, resource scheduler), `storage`, `observability`.
- **Immutability guard** (`models/base.py`): sealed fields on selected models raise `ImmutableFieldError` on mutation. There is no insert-time default hook — `model_nodes.code` (`M001`…) is generated only by `repositories/model_repository.py:_generate_model_code`.
- `training/training_params.py` is the **single source of truth** for the exposed training surface (epochs/imgsz/batch/patience/cache/augmentation + raw coefficient whitelist; `workers` and `amp` locked; VRAM guard). The **complete parameter snapshot is written on the task row at creation** (route calls `resolve_config`), because the immutability guard forbids the worker back-filling it.
- Worker rules (`workers/`): use `worker_session()` (own engine), never request-scoped `get_db`; **commit before the long `trainer.train()` call** — a held transaction made cancellation block for minutes (documented regression); heavy imports are deliberately lazy except `inference/yolo_engine.py:11` (module-level `ultralytics` import).
- Migrations: chain `0000…0005` (head). `alembic/env.py` imports `app.*` (not `backend.app.*`), hence `PYTHONPATH=backend` from root. `0004` repaired a chain that previously produced a schema the ORM couldn't query — diff `inspect(engine)` vs `Base.metadata` before trusting migrations.
- Evaluations run on **CPU on purpose** (no GPU lease). The quality gate and human-review workflow were **removed on request (2026-09-16)** — do not re-add them; `auto_status` is pipeline state only.

### Frontend (`frontend/src/`)

- `app/` (shell, routing), `features/` (one dir per domain page: dashboard, datasets, models, training, evaluations, releases, bindings, resources, operations, api), `lib/` (`createApi.ts` picks `mockApi` on `VITE_USE_MOCKS=true`, `api.ts` real calls).
- Multi-tree model lineage: any number of root models (one lineage per application base, e.g. `yolo11n-seg.pt` for segmentation, `yolov8n.pt` for detection). Import dialog selects task type (`object_detection` / `instance_segmentation`); training page has an explicit 新建根节点 option. New-root training without a parent falls back to `yolov8n.pt` (detection only).
- Lineage layout uses `dagre` for DAG layout; delete only on leaf nodes.
- `createApi.ts` camelizes response keys but **sends snake_case bodies on GPU/infer paths** — match the existing shape.
- UI copy is Chinese.

## Hard-won gotchas (details in AGENTS.md)

- **BOM convention**: every file under `backend/`, `frontend/`, `infra/` (and root `.md` docs) must carry a UTF-8 BOM — `backend/scripts/verify_bom.py` enforces it. **The `edit` tool strips BOMs — re-add after every edit.** Exceptions that must stay BOM-free: `backend/pyproject.toml` and `backend/alembic.ini` (parsers reject BOM; that's why tests use `-c NUL`) and `frontend/nginx.conf` (nginx rejects it).
- **Repo size**: ~22,700 tracked files / 2.7 GB clone; `dataset/`, `infra/{datasets,models,checkpoints,redis}` and `*.pt` are ignored by rules but ~22,482 files are tracked anyway. Never `git add -A` blindly; edits under those paths are real diffs.
- **Known test failures are pre-existing and committed**: ~132 failures from `model_nodes.code` NOT NULL fixture drift (fixtures insert without `code`), plus a few from the light env. Don't mass-"fix" them without reading the AGENTS.md breakdown.
- **GPU is a GTX 745 (Maxwell sm_50, 4095 MB)**: PyTorch 2.15+ has no Maxwell wheels — this env pins `torch==2.14.0+cu126` (don't upgrade). Repo default `batch=16/imgsz=640` overflows the card; `.env` sets batch 8 / imgsz 416 (~165 s/epoch measured). `workers=0` is deliberate (Celery solo on Windows).
- **A stale `active` row in `gpu_resource_leases` blocks all later GPU runs** (capacity − active leases). Don't `DELETE /api/training/tasks/{id}` while its worker is finishing — that's how orphan leases appear.
- **Route changes need an API restart** (Scheduled Task `AIP-API`), restarting only the worker leaves old route code loaded.
- **System sleep kills training silently** — this box is configured with `powercfg /change standby-timeout-ac 0`; don't let that regress.
- Pin discipline matters: `redis>=5,<6` (Redis 8's RESP3 breaks the local Redis 5 server), `Pillow>=10,<11`.
- Local DB: PostgreSQL 16.12 (service `postgresql-x64-16`) + Redis 5.0.14.1 (service `Redis`, binds 127.0.0.1 only), db `ai_platform` at `alembic_version = 0005`.

## Docs

`docs/operations/first-run.md` (setup entry point), `docs/deployment/{installation,configuration,operations,troubleshooting}.md`, `docs/performance/benchmark.md`, `docs/production/checklist.md`, design history in `docs/superpowers/`, plus root `ai-model-platform-design.md` (overall design, Chinese) and `platform-prototype.html`.
