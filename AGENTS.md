# AGENTS.md

## Stack & Layout

- **Backend**: Python 3.11, FastAPI (`backend/app/main.py:create_app`), SQLAlchemy 2 + Alembic, Celery + Redis, Ultralytics YOLO. Entrypoint `app.main:app` / `uvicorn`.
- **Frontend**: React 18 + TypeScript + Vite (`frontend/src/main.tsx`), React Router. `VITE_USE_MOCKS=true` switches `src/lib/createApi.ts` to `mockApi`.
- **Infra**: `infra/docker-compose.yml` — postgres:16, redis:7-alpine, migrate (one-shot `alembic upgrade head`), api, inference (placeholder), worker (`celery -A app.workers.celery_app worker -Q training,evaluation,default --concurrency=1`), scheduler (beat), frontend (node build + nginx). Frontend nginx at `8080` proxies `/api` to `api:8000`; direct API at `8000`.
- **Dirs**: `backend/app/{api/routes,models,schemas,services,workers,runtime,evaluation,training,storage,observability}` | `backend/alembic/versions` | `frontend/src/{app,features,lib}` | `infra/` | `docs/`

## Commands (exact)

### Backend

```bash
# All tests from repo root — must use -c NUL because pyproject.toml/alembic.ini may carry UTF-8 BOM
python -m pytest -c NUL -p no:cacheprovider backend/tests -q
# Single file / single test
python -m pytest -c NUL -p no:cacheprovider backend/tests/unit/test_health.py -q
python -m pytest -c NUL -p no:cacheprovider backend/tests/unit/test_health.py -k test_gpu_ready_override -q
# Skip GPU/slow tests
python -m pytest -c NUL -p no:cacheprovider backend/tests -q -m "not slow"
# Compile check, BOM check
python -m compileall backend
python backend/scripts/verify_bom.py
# Migrations (needs POSTGRES_URL or POSTGRES_HOST/PORT/DB/USER/PASSWORD env)
alembic upgrade head
```

### Frontend

```bash
cd frontend
npm install
npm run dev      # Vite :3000, proxies /api -> 127.0.0.1:8000 (vite.config.ts)
npm run build    # tsc -b && vite build
npm run test     # playwright test — config runs `npx vite --mode mock` automatically
npx playwright test tests/e2e/ --reporter=list
```

### Docker

```bash
cd infra
cp .env.example .env   # edit PLATFORM_API_KEY, POSTGRES_PASSWORD, HOST_*_DIR
docker compose up -d
docker compose logs api
docker compose build api --no-cache
# Health: api 8000/health/ready, frontend 8080/, postgres 5432, redis 6379
```

## Env & Config Quirks

- `backend/app/config.py:Settings` uses `env_file=".env"` with `utf-8-sig` and `extra="ignore"`. If `POSTGRES_URL` is empty it is built from `POSTGRES_HOST/PORT/DB/USER/PASSWORD` via `sqlalchemy.URL.create` (password URL-encoded). `alembic/env.py` reads the same Settings.
- Default `PLATFORM_API_KEY=change-me` (header `X-API-Key` only — never in URL). Frontend sends `VITE_API_KEY` (defaults to `change-me`) in `createApi.ts`.
- `GPU_READY`/`WORKER_READY` are boolean readiness overrides consumed by `backend/app/db.py:default_health_checks` — `/health` returns `ready`/`degraded`/`unhealthy` based on postgres+redis (unhealthy) vs gpu+worker (degraded).
- `HOST_*_DIR` in `.env` bind-mounts `./models,./datasets,./logs,./checkpoints` into containers at `/data/*`.

## BOM / File Encoding

- Global instruction requires UTF-8 with BOM (`EF BB BF`) for Python files. In practice `verify_bom.py` expects BOM on every text file under `backend/`, `frontend/`, `infra/` (suffixes `.py .ts .tsx .js .json .yml` etc. + `Dockerfile`) and currently reports many missing — run it to see drift.
- Dockerfile `COPY pyproject.toml alembic.ini` then `pip install .` does **not** strip BOM (check still fails locally); local `pytest`/`alembic` fail on BOM configs without `-c NUL` or a clean copy, so always pass `-c NUL -p no:cacheprovider`.
- **Exception: `frontend/nginx.conf` must NOT have BOM** — nginx rejects `﻿events` as unknown directive. Remove BOM after editing: `python -c "d=open('frontend/nginx.conf','rb').read(); open('frontend/nginx.conf','wb').write(d[3:] if d[:3]==b'\xef\xbb\xbf' else d)"`.
- BOM check order: edit file → add BOM → if `nginx.conf` remove BOM → rebuild.

## Testing & Fixtures

- `backend/tests/conftest.py` only registers `slow` marker — not auto-skipped; use `-m "not slow"` to exclude.
- Fixtures: `backend/tests/fixtures/build_fixtures.py:FixtureBuilder` (deterministic UUIDs via sha256 seed) and `build_real_fixtures.py`. Unit `backend/tests/unit/`, integration `backend/tests/integration/` (+ `conftest.py` with DB overrides), e2e `backend/tests/e2e/test_closed_loop.py`.
- Frontend Playwright `frontend/playwright.config.ts`: `baseURL http://localhost:3000`, `webServer: npx vite --mode mock`, `reuseExistingServer: !CI`.
- UI self-test: prefer Chrome DevTools MCP over Playwright per `CLAUDE.md` — open real page at `8080`/`3000`, check console, exercise interactions, read DOM, report verification.

## Data Model (non-obvious)

- `model_nodes` has auto-increment `code` (`M001`, `M002` … via `model_repository.py:_generate_model_code` + `alembic/versions/0003_model_code.py`); immutable with parent lineage; `code` is used in URLs (`GET /api/models/M001`) and tree labels. `model_bindings` stable IDs; `model_binding_releases` monotonic `revision_no` per binding; `dataset_snapshots` immutable with `manifest_hash`; `training_tasks`/`training_attempts` with `attempt_lease` fencing; `evaluations` auto+human (`metrics.per_class` shown in eval detail drawer); `gpu_resource_leases` Postgres-backed; `model_residency_plans`.

## Training / Worker Gotchas

- Celery task `platform.training.run_training` takes task UUID; beat schedule in `backend/app/workers/celery_app.py` (`sweep-pending-evaluations`, `reconcile-leases`, `sweep-queued-tasks` every 60s).
- Worker must use `backend/app/workers/db_session.py:worker_session()` (own engine/session), not request-scoped `get_db`. `failed` is terminal, no auto-retry; cancellation is cooperative (API sets flag, worker checks before/after). CPU mode skips GPU lease; GPU mode requires `torch.cuda.is_available()`.

## Frontend API Note

- `frontend/src/lib/mockApi.ts` is the mock; `createApi()` picks it when `VITE_USE_MOCKS=true`. Real API base is `VITE_API_URL` (empty = relative). Nginx `frontend/nginx.conf` and Vite proxy both forward `/api`.
- `frontend/src/features/api/ApiTestPage.tsx` — per-model inference test (`POST /api/models/{code}/infer` returns `overlay_image` + `detections`). Each model gets a card; left-right layout (param docs vs. result JSON + image). Uses `code` (e.g. `M001`) not UUID.

## Hotpatch Workflow

When `docker compose build` is too slow for iterative changes:

```bash
# Copy changed file into container
docker cp backend/app/api/routes/datasets.py infra-api-1:/app/app/api/routes/datasets.py
docker restart infra-api-1
# Wait for health
sleep 8; curl -s http://localhost:8000/health/ready
```

For frontend changes (must rebuild nginx image):
```bash
cd infra && docker compose build frontend --no-cache && docker compose up -d frontend --no-deps
```

## Container Structure

- Backend container: code at `/app` with `PYTHONPATH=/app`. `backend/` at `/app/backend` is a symlink (`Dockerfile:RUN ln -s /app /app/backend`). Both `from backend.xxx` and `from app.xxx` resolve to the same code.
- Dataset import stores extracted files at `/data/datasets/extracted/{dataset_id}/` (bind-mounted from `./datasets/`). Snapshots at `/data/datasets/snapshots/`; `store` (`/data/datasets/store/`) is no longer populated during import — `create_dataset_snapshot` only writes `manifest.json` with `source_dir` + relative paths, training/evaluation read directly from extracted files (avoids 20k+ 9p `copy2` on Windows bind mount).
- Worker and scheduler containers share the same image and env; worker runs Celery with `--concurrency=1`, scheduler runs `celery beat`.

## Dataset Import Pipeline

- `POST /api/datasets` (async): inserts `Dataset` row → `db.commit()` → schedules `_build_snapshot_background` via `BackgroundTasks` → returns `201` with `0/0/0` counts immediately.
- Background task: extracts zip → validates via `_locate_yaml` / `_parse_data_yaml` → `create_dataset_snapshot` → inserts `DatasetSnapshot` + `LabelSchema` → frontend polling detects completion.
- `_build_snapshot_background` uses `worker_session()` (separate DB connection), not request-scoped `get_db`. Must `commit()` before scheduling to avoid invisible rows.
- Frontend polls `GET /api/datasets` every 3s after upload, displays `解析中` badge for rows with `imageCount 0 && validationStatus "-" && !latestSnapshotId`.
- `nginx.conf` `proxy_read_timeout 600s` required — default 300s causes 504 on 700MB+ archives.
- `_locate_yaml` supports `data.yaml`, any `*.yaml` at root, and `*.yaml` in one-level subdirs (e.g., `crack-seg.yaml`).
- `_parse_data_yaml` infers `nc` from `len(names)` when only a dict `names: {0: crack}` is present (no explicit `nc` field).

## Model Tree

- Uses `dagre` (dependency in `frontend/package.json`) for DAG layout of model lineage.
- Delete button (×) only shows for leaf nodes (`byParent(n.id).length === 0`).
- Backend `DELETE /api/models/{id}` cascades dependent evaluations before deleting the model node.
