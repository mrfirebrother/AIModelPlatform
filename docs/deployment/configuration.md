# Configuration Guide

This document describes all configuration options for the AI Model Platform.

## Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `.env` | `infra/.env` | Environment variables |
| `docker-compose.yml` | `infra/docker-compose.yml` | Service definitions |
| `pyproject.toml` | `backend/pyproject.toml` | Python dependencies |
| `package.json` | `frontend/package.json` | Frontend dependencies |

## Environment Variables

### Database Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `POSTGRES_HOST` | `postgres` | Yes | PostgreSQL hostname |
| `POSTGRES_PORT` | `5432` | Yes | PostgreSQL port |
| `POSTGRES_DB` | `ai_platform` | Yes | Database name |
| `POSTGRES_USER` | `ai_platform` | Yes | Database user |
| `POSTGRES_PASSWORD` | `change-me` | Yes | Database password |
| `POSTGRES_URL` | (empty) | No | Full connection URL (overrides above) |

### Redis Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | Yes | Redis connection URL |

### Storage Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `MODEL_DIR` | `/data/models` | Yes | Model files directory |
| `DATASET_DIR` | `/data/datasets` | Yes | Dataset files directory |
| `LOG_DIR` | `/data/logs` | Yes | Training logs directory |
| `CHECKPOINT_DIR` | `/data/checkpoints` | Yes | Checkpoint files directory |

### Host Volume Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST_POSTGRES_DIR` | `./postgres` | Host path for PostgreSQL data |
| `HOST_MODELS_DIR` | `./models` | Host path for model files |
| `HOST_DATASETS_DIR` | `./datasets` | Host path for dataset files |
| `HOST_LOGS_DIR` | `./logs` | Host path for log files |
| `HOST_CHECKPOINTS_DIR` | `./checkpoints` | Host path for checkpoint files |

### API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_EXTERNAL_PORT` | `8000` | API server external port |
| `PLATFORM_API_KEY` | `change-me` | API key for admin operations |
| `MAX_INFERENCE_INPUT_BYTES` | `10485760` | Max inference input size (10MB) |
| `MAX_INFERENCE_CONCURRENCY` | `1` | Max concurrent inference requests |

### GPU Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_DEVICE` | `0` | GPU device index |
| `GPU_READY` | `false` | Set `true` when GPU is configured |
| `GPU_COUNT` | `1` | Number of GPUs to reserve |
| `GPU_MEMORY_RESERVATION_MB` | `4096` | GPU memory reservation (MB) |
| `NVIDIA_VISIBLE_DEVICES` | `all` | NVIDIA device visibility |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | NVIDIA driver capabilities |

### Training Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_READY` | `false` | Set `true` when worker is ready |
| `TRAINING_DEFAULT_EPOCHS` | `100` | Default training epochs |
| `TRAINING_DEFAULT_BATCH_SIZE` | `16` | Default training batch size |

### Frontend Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FRONTEND_EXTERNAL_PORT` | `8080` | Frontend external port |

## Production Configuration

### Security Settings

```bash
# Change default passwords
POSTGRES_PASSWORD=<strong-random-password>
PLATFORM_API_KEY=<strong-random-api-key>

# Disable default credentials
POSTGRES_PASSWORD=<unique-password>
```

### Performance Settings

```bash
# Increase concurrency for high-traffic
MAX_INFERENCE_CONCURRENCY=4

# Adjust GPU memory based on model size
GPU_MEMORY_RESERVATION_MB=8192

# Increase batch size for training
TRAINING_DEFAULT_BATCH_SIZE=32
```

### Storage Settings

```bash
# Use dedicated storage volumes
HOST_MODELS_DIR=/mnt/storage/models
HOST_DATASETS_DIR=/mnt/storage/datasets
HOST_LOGS_DIR=/mnt/storage/logs
HOST_CHECKPOINTS_DIR=/mnt/storage/checkpoints
```

## GPU Configuration

### Single GPU Setup

```bash
# .env
GPU_DEVICE=0
GPU_READY=true
GPU_COUNT=1
NVIDIA_VISIBLE_DEVICES=0
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Multi-GPU Setup

```bash
# .env
GPU_DEVICE=0
GPU_READY=true
GPU_COUNT=2
NVIDIA_VISIBLE_DEVICES=0,1
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

### Verify GPU Configuration

```bash
# Check GPU is visible in container
docker compose exec inference nvidia-smi

# Check GPU is ready
curl http://localhost:8000/health
# Should show: "gpu": true
```

## Database Configuration

### Default Configuration

```bash
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ai_platform
POSTGRES_USER=ai_platform
POSTGRES_PASSWORD=change-me
```

### External Database

```bash
# Use external PostgreSQL
POSTGRES_HOST=external-db-host
POSTGRES_PORT=5432
POSTGRES_DB=ai_platform
POSTGRES_USER=ai_platform
POSTGRES_PASSWORD=<password>

# Or use full URL
POSTGRES_URL=postgresql://user:password@host:5432/dbname
```

### Connection Pool

The application uses SQLAlchemy connection pooling with default settings:

- Pool size: 5
- Max overflow: 10
- Pool timeout: 30s
- Pool recycle: 1800s

## Docker Compose Configuration

### Service Ports

```yaml
services:
  api:
    ports:
      - "${API_EXTERNAL_PORT:-8000}:8000"
  frontend:
    ports:
      - "${FRONTEND_EXTERNAL_PORT:-8080}:80"
```

### Resource Limits

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Health Checks

All services have built-in health checks. Customize timing:

```yaml
services:
  api:
    healthcheck:
      interval: 10s
      timeout: 5s
      retries: 6
      start_period: 15s
```

## Configuration Examples

### Development Environment

```bash
# .env for development
POSTGRES_PASSWORD=dev-password
PLATFORM_API_KEY=dev-api-key
GPU_READY=false
WORKER_READY=false
MAX_INFERENCE_CONCURRENCY=1
```

### Production Environment

```bash
# .env for production
POSTGRES_PASSWORD=<strong-random-password>
PLATFORM_API_KEY=<strong-random-api-key>
GPU_READY=true
WORKER_READY=true
MAX_INFERENCE_CONCURRENCY=4
GPU_MEMORY_RESERVATION_MB=8192
TRAINING_DEFAULT_BATCH_SIZE=32
HOST_MODELS_DIR=/mnt/storage/models
HOST_DATASETS_DIR=/mnt/storage/datasets
HOST_LOGS_DIR=/mnt/storage/logs
```

### High-Availability Environment

```bash
# .env for HA (requires load balancer and shared storage)
POSTGRES_HOST=postgres-cluster
REDIS_URL=redis://redis-cluster:6379/0
GPU_DEVICE=0,1
GPU_COUNT=2
NVIDIA_VISIBLE_DEVICES=0,1
MAX_INFERENCE_CONCURRENCY=8
```

## Configuration Validation

### Verify Environment Variables

```bash
# Check all required variables are set
docker compose config

# Validate specific service
docker compose config api
```

### Test Configuration

```bash
# Test database connection
docker compose exec api python -c "
import psycopg
conn = psycopg.connect('postgresql://ai_platform:password@postgres:5432/ai_platform')
print('Database connected')
conn.close()
"

# Test Redis connection
docker compose exec api python -c "
import redis
r = redis.from_url('redis://redis:6379/0')
r.ping()
print('Redis connected')
"
```

## Configuration Updates

### Updating Environment Variables

1. Edit `infra/.env`
2. Restart affected services:

```bash
docker compose restart api worker inference
```

### Updating Docker Compose

1. Edit `infra/docker-compose.yml`
2. Recreate services:

```bash
docker compose up -d --force-recreate
```

## Next Steps

- [Installation Guide](installation.md)
- [Troubleshooting Guide](troubleshooting.md)
- [Operations Guide](operations.md)
