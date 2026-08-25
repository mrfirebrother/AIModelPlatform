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

## Testing & Fixtures

- `backend/tests/conftest.py` only registers `slow` marker — not auto-skipped; use `-m "not slow"` to exclude.
- Fixtures: `backend/tests/fixtures/build_fixtures.py:FixtureBuilder` (deterministic UUIDs via sha256 seed) and `build_real_fixtures.py`. Unit `backend/tests/unit/`, integration `backend/tests/integration/` (+ `conftest.py` with DB overrides), e2e `backend/tests/e2e/test_closed_loop.py`.
- Frontend Playwright `frontend/playwright.config.ts`: `baseURL http://localhost:3000`, `webServer: npx vite --mode mock`, `reuseExistingServer: !CI`.

## Data Model (non-obvious)

- `model_nodes` immutable with parent lineage; `model_bindings` stable IDs; `model_binding_releases` monotonic `revision_no` per binding; `dataset_snapshots` immutable with `manifest_hash`; `training_tasks`/`training_attempts` with `attempt_lease` fencing; `evaluations` auto+human; `gpu_resource_leases` Postgres-backed; `model_residency_plans`.

## Training / Worker Gotchas

- Celery task `platform.training.run_training` takes task UUID; beat schedule in `backend/app/workers/celery_app.py` (`sweep-pending-evaluations`, `reconcile-leases`, `sweep-queued-tasks` every 60s).
- Worker must use `backend/app/workers/db_session.py:worker_session()` (own engine/session), not request-scoped `get_db`. `failed` is terminal, no auto-retry; cancellation is cooperative (API sets flag, worker checks before/after). CPU mode skips GPU lease; GPU mode requires `torch.cuda.is_available()`.

## Frontend API Note

- `frontend/src/lib/mockApi.ts` is the mock; `createApi()` picks it when `VITE_USE_MOCKS=true`. Real API base is `VITE_API_URL` (empty = relative). Nginx `frontend/nginx.conf` and Vite proxy both forward `/api`.
