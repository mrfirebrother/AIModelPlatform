# AGENTS.md

## Project Overview

AI model platform for importing root YOLO models, managing dataset snapshots, training detection models, evaluating candidates, and publishing/rolling back models through stable bindings.

## Tech Stack

- **Backend**: Python 3.11, FastAPI, SQLAlchemy, Alembic, Celery, Redis, Ultralytics YOLO
- **Frontend**: React 18, TypeScript, Vite, React Router
- **Database**: PostgreSQL
- **Deployment**: Docker Compose + NVIDIA Container Toolkit
- **Testing**: pytest, Playwright

## Developer Commands

### Backend Tests

```bash
cd backend
python -m pytest -c NUL -p no:cacheprovider backend/tests -q
```

- Use `-c NUL` because `pyproject.toml` has UTF-8 BOM; standard `pytest` cannot parse it.
- GPU tests are marked `@pytest.mark.slow` and skipped by default.
- Unit tests: `backend/tests/unit/`
- Integration tests: `backend/tests/integration/`
- E2E tests: `backend/tests/e2e/`

### Frontend

```bash
cd frontend
npm install
npm run dev      # Vite dev server on :3000
npm run build    # TypeScript check + production build
npm run test     # Playwright tests
```

### Docker

```bash
cd infra
docker compose up -d          # Start all services
docker compose build api --no-cache  # Rebuild backend
docker compose logs api       # Check API logs
```

API proxy is at `http://localhost:8080` (nginx). Dev server at `http://localhost:3000`.

### BOM Verification

```bash
python backend/scripts/verify_bom.py
```

All Python files must have UTF-8 BOM. Frontend `.ts/.tsx` files must NOT have BOM.

## Architecture

```
frontend/         React admin console (mock or real API)
backend/app/      FastAPI application
  api/routes/     REST endpoints
  models/         SQLAlchemy ORM models
  schemas/        Pydantic request/response
  services/       Business logic
  workers/        Celery tasks (training, evaluation, scheduler)
  runtime/        Model loading and lifecycle
  evaluation/     Metrics and policy
  training/       YOLO dataset, trainer, checkpoint
  storage/        File snapshots
  observability/  Logs and alerts
infra/            Docker Compose
```

## Key Data Model

- `model_nodes`: Immutable model artifacts with parent-child lineage
- `model_bindings`: Stable project-level binding IDs
- `model_binding_releases`: Per-binding release history (revision_no monotonic)
- `dataset_snapshots`: Immutable dataset snapshots
- `training_tasks` / `training_attempts`: Task queue with lease/fencing
- `evaluations`: Auto + human review
- `gpu_resource_leases`: PostgreSQL-backed GPU capacity control
- `model_residency_plans`: Administrator-desired resident set

## API Key

- Header: `X-API-Key`
- Default: `change-me` (configured via `PLATFORM_API_KEY` env var)
- Do NOT pass key in URL parameters

## Python File Convention

All `.py` files under `backend/` must start with UTF-8 BOM (`EF BB BF`). Docker strips BOM from `pyproject.toml` and `alembic.ini` at build time. Frontend files must NOT have BOM.

## Training Worker Notes

- Celery task `platform.training.run_training` receives a task UUID
- Worker uses `worker_session()` for its own DB session (not request-scoped)
- CPU mode skips GPU lease; GPU mode requires CUDA availability
- Manifest hash is verified against stored snapshot hash before training
- `failed` status is terminal; no automatic retry
- Cancellation is cooperative: flag set by API, checked by worker before/after training
