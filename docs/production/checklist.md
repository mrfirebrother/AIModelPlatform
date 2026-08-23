# Production Readiness Checklist

## 1. Health Probes

| Probe | Endpoint | Purpose | Failure Action |
|-------|----------|---------|----------------|
| Liveness | `GET /health/live` | Process is alive | Kubernetes restarts pod |
| Readiness | `GET /health/ready` | Dependencies reachable | Remove from Service endpoints |
| Startup | `GET /health/startup` | Initialisation complete | Probe retries until ready |

### Verification

```bash
# Liveness
curl -f http://localhost:8000/health/live

# Readiness (needs Postgres + Redis)
curl -f http://localhost:8000/health/ready

# Startup (after migration completes)
curl -f http://localhost:8000/health/startup
```

## 2. Docker Compose

### Resource Limits

Every service must have explicit `deploy.resources.limits` and `deploy.resources.reservations` to prevent runaway memory usage.

| Service | CPU Limit | Memory Limit | GPU Limit |
|---------|-----------|--------------|-----------|
| postgres | 2.0 | 2G | - |
| redis | 1.0 | 1G | - |
| api | 4.0 | 4G | - |
| inference | 4.0 | 8G | 1 GPU |
| worker | 4.0 | 8G | 1 GPU |
| scheduler | 1.0 | 1G | - |
| frontend | 1.0 | 512M | - |

### Restart Policies

| Service | Policy | Max Retry |
|---------|--------|-----------|
| postgres | unless-stopped | 10 |
| redis | unless-stopped | 10 |
| api | unless-stopped | 10 |
| inference | unless-stopped | 10 |
| worker | unless-stopped | 10 |
| scheduler | unless-stopped | 10 |
| frontend | unless-stopped | 10 |

## 3. Environment Variables

### Required (no default)

| Variable | Description |
|----------|-------------|
| `POSTGRES_PASSWORD` | Database password |
| `PLATFORM_API_KEY` | API authentication key |

### Expected Defaults

| Variable | Default | Notes |
|----------|---------|-------|
| `POSTGRES_DB` | ai_platform | |
| `POSTGRES_USER` | ai_platform | |
| `REDIS_URL` | redis://redis:6379/0 | |
| `GPU_READY` | false | Set `true` when GPU is available |
| `WORKER_READY` | false | Set `true` when worker is ready |

## 4. Observability

### Logging

- Structured JSON logging in production (`uvicorn --log-level warning`)
- Celery worker log level: `INFO`
- Database migration logging via Alembic

### Metrics (future)

- `/metrics` endpoint (Prometheus)
- Key signals: request latency, error rate, active jobs, GPU utilisation

## 5. Security

| Item | Status | Notes |
|------|--------|-------|
| Secrets via env vars | Done | `.env` not committed |
| API key authentication | Done | `PLATFORM_API_KEY` header |
| No default passwords | Done | `change-me` placeholder |
| CORS | Check | Frontend origin only |
| Rate limiting | Todo | Per-IP, per-user |

## 6. Data Persistence

| Service | Volume | Backup Strategy |
|---------|--------|----------------|
| postgres | `HOST_POSTGRES_DIR` | pg_dump cron / point-in-time recovery |
| redis | `HOST_REDIS_DIR` | RDB snapshot (periodic) |
| models | `HOST_MODELS_DIR` | S3 / NFS |
| datasets | `HOST_DATASETS_DIR` | S3 / NFS |
| checkpoints | `HOST_CHECKPOINTS_DIR` | S3 / NFS |
| logs | `HOST_LOGS_DIR` | Log aggregation (Loki, ELK) |

## 7. Scaling Considerations

- **Horizontal**: Run multiple `api` instances behind a load balancer.
- **Worker scaling**: Increase `--concurrency` or run additional Celery workers.
- **Database**: Move to managed Postgres (RDS, Cloud SQL) for production.
- **Redis**: Consider Redis Cluster / Sentinel for HA.

## 8. Pre-Deploy Verification

```bash
# 1. Validate docker-compose syntax
docker compose -f infra/docker-compose.yml config --quiet

# 2. Run health probes
docker compose -f infra/docker-compose.yml up -d
docker compose -f infra/docker-compose.yml ps

# 3. Check logs for errors
docker compose -f infra/docker-compose.yml logs --tail=50 api

# 4. Verify probe endpoints
curl -sf http://localhost:8000/health/live  | jq .
curl -sf http://localhost:8000/health/ready | jq .
curl -sf http://localhost:8000/health/startup | jq .
```
