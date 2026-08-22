# First Run Guide

This document describes how to start the AI Model Platform for the first time
using Docker Compose.

## Prerequisites

- Docker Engine 24+ with Compose V2
- NVIDIA Container Toolkit (for GPU passthrough)
- At least 16 GB system RAM recommended
- A `.pt` root model file for inference

## Quick Start

### 1. Clone and configure

```bash
cd ai-model-platform
cp infra/.env.example infra/.env   # create from template if available
```

Edit `infra/.env` to set:

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_PASSWORD` | `change-me` | PostgreSQL password |
| `PLATFORM_API_KEY` | `change-me` | API key for admin operations |
| `GPU_DEVICE` | `0` | GPU device index |
| `GPU_READY` | `false` | Set `true` if GPU is available |
| `HOST_MODELS_DIR` | `./models` | Host path for model storage |
| `HOST_DATASETS_DIR` | `./datasets` | Host path for dataset storage |
| `HOST_LOGS_DIR` | `./logs` | Host path for training logs |
| `HOST_CHECKPOINTS_DIR` | `./checkpoints` | Host path for checkpoints |

### 2. Start infrastructure

```bash
cd infra
docker compose up -d postgres redis
docker compose run --rm migrate
```

### 3. Start all services

```bash
docker compose up -d
```

This starts:
- **postgres** - metadata storage
- **redis** - task queue and cache
- **migrate** - one-shot Alembic migration
- **api** - FastAPI platform service (port 8000)
- **worker** - Celery training/evaluation worker
- **scheduler** - Celery beat scheduler
- **inference** - GPU inference worker
- **frontend** - React admin UI (port 8080)

### 4. Verify health

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "degraded",
  "checks": {
    "postgres": true,
    "redis": true,
    "gpu": false,
    "worker": false
  }
}
```

Status `degraded` means infrastructure is healthy but GPU/worker are not ready.
Set `GPU_READY=true` and `WORKER_READY=true` in `.env` when the GPU environment
is properly configured, then restart the affected services.

### 5. Import a root model

Place a `.pt` file in the host models directory:

```bash
cp /path/to/yolov8n.pt ./models/
```

Then use the admin UI at `http://localhost:8080` to import the root model, or
call the API directly:

```bash
curl -X POST http://localhost:8000/api/models \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-platform-api-key" \
  -d '{
    "task_type": "object_detection",
    "model_family": "yolo",
    "artifact_path": "/data/models/yolov8n.pt",
    "artifact_hash": "sha256:..."
  }'
```

### 6. Import a dataset

Place YOLO-format dataset in the host datasets directory:

```
./datasets/my-dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  data.yaml
```

Use the admin UI or API to import and snapshot the dataset.

## Service Ports

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI REST API |
| Frontend | 8080 | React admin UI |
| PostgreSQL | 5432 | Database (internal) |
| Redis | 6379 | Queue/cache (internal) |

## Volumes

| Container Path | Host Path | Description |
|----------------|-----------|-------------|
| `/data/models` | `./models` | Model `.pt` files |
| `/data/datasets` | `./datasets` | YOLO dataset files |
| `/data/logs` | `./logs` | Training logs |
| `/data/checkpoints` | `./checkpoints` | Training checkpoints |

## Troubleshooting

### PostgreSQL not starting

Check `docker compose logs postgres`.  Common causes:
- Port 5432 already in use
- Insufficient disk space for data volume

### API returns unhealthy

```bash
docker compose logs api | tail -20
```

Check that PostgreSQL and Redis are healthy first, then check API logs for
connection errors.

### GPU not detected

Ensure NVIDIA Container Toolkit is installed:

```bash
nvidia-smi                         # host GPU check
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi  # container GPU check
```

Set `GPU_READY=true` in `.env` and restart `api`, `worker`, and `inference`.

### Worker not connecting

Check Redis connectivity:

```bash
docker compose exec redis redis-cli ping
```

Expected response: `PONG`
