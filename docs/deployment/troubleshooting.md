# Troubleshooting Guide

This document covers common issues and their solutions.

## Common Errors

### Database Errors

#### PostgreSQL Connection Refused

**Symptoms:**
```
connection to server at "postgres" (172.18.0.2), port 5432 failed: Connection refused
```

**Causes:**
- PostgreSQL container not running
- Wrong hostname or port
- Authentication failure

**Solutions:**

```bash
# Check PostgreSQL status
docker compose ps postgres

# Check PostgreSQL logs
docker compose logs postgres

# Verify credentials
docker compose exec postgres psql -U ai_platform -d ai_platform

# Restart PostgreSQL
docker compose restart postgres
```

#### Database Migration Failed

**Symptoms:**
```
alembic.util.exc.CommandError: Can't connect to PostgreSQL
```

**Solutions:**

```bash
# Ensure PostgreSQL is healthy
docker compose ps postgres

# Run migration manually
docker compose run --rm migrate

# Check migration status
docker compose run --rm migrate alembic current
```

### Redis Errors

#### Redis Connection Refused

**Symptoms:**
```
redis.exceptions.ConnectionError: Error connecting to redis:6379
```

**Solutions:**

```bash
# Check Redis status
docker compose ps redis

# Check Redis logs
docker compose logs redis

# Test Redis connectivity
docker compose exec redis redis-cli ping
# Should return: PONG

# Restart Redis
docker compose restart redis
```

### GPU Errors

#### GPU Not Detected

**Symptoms:**
```
RuntimeError: No CUDA GPUs are available
```

**Solutions:**

```bash
# Verify host GPU
nvidia-smi

# Verify container GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Check NVIDIA Container Toolkit
nvidia-ctk --version

# Restart Docker daemon
sudo systemctl restart docker

# Verify environment variables
docker compose exec inference env | grep NVIDIA
```

#### GPU Memory Overflow

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

```bash
# Check GPU memory usage
nvidia-smi

# Reduce batch size
# Edit infra/.env
TRAINING_DEFAULT_BATCH_SIZE=8

# Increase GPU memory reservation
# Edit infra/.env
GPU_MEMORY_RESERVATION_MB=8192

# Restart affected services
docker compose restart worker inference
```

### API Errors

#### API Returns 503 Service Unavailable

**Symptoms:**
```json
{"detail": "Service Unavailable"}
```

**Solutions:**

```bash
# Check API health
curl http://localhost:8000/health

# Check API logs
docker compose logs api

# Verify dependencies
docker compose ps

# Restart API
docker compose restart api
```

#### API Timeout

**Symptoms:**
```
httpx.ReadTimeout: The read operation timed out
```

**Solutions:**

```bash
# Increase timeout in client
# Or increase API timeout
# Edit infra/docker-compose.yml
# Add to api service:
# environment:
#   - TIMEOUT=300

# Check API load
docker compose logs api | grep timeout
```

### Worker Errors

#### Celery Worker Not Starting

**Symptoms:**
```
celery.errors.NotReady: The node is not ready
```

**Solutions:**

```bash
# Check worker status
docker compose ps worker

# Check worker logs
docker compose logs worker

# Restart worker
docker compose restart worker

# Verify Redis connection
docker compose exec worker celery -A app.workers.celery_app inspect ping
```

#### Training Task Stuck

**Symptoms:**
- Task status remains PENDING or STARTED
- No progress updates

**Solutions:**

```bash
# Check worker logs
docker compose logs worker --tail=100

# Check task status via API
curl http://localhost:8000/api/tasks/<task-id>

# Restart worker
docker compose restart worker

# Check GPU utilization
nvidia-smi
```

### Frontend Errors

#### Frontend Cannot Connect to API

**Symptoms:**
- Admin UI shows "Network Error"
- API calls fail

**Solutions:**

```bash
# Check frontend configuration
docker compose exec frontend cat /etc/nginx/conf.d/default.conf

# Verify API is running
curl http://localhost:8000/health

# Check network connectivity
docker compose exec frontend curl http://api:8000/health

# Restart frontend
docker compose restart frontend
```

## Log Viewing

### View Service Logs

```bash
# All logs for a service
docker compose logs api

# Follow logs in real-time
docker compose logs -f api

# Last 100 lines
docker compose logs --tail=100 api

# Logs since a specific time
docker compose logs --since="2024-01-01T00:00:00" api
```

### View All Logs

```bash
# All services
docker compose logs

# Follow all
docker compose logs -f

# Filter by service
docker compose logs api worker inference
```

### Log Locations

| Service | Log Location | Description |
|---------|--------------|-------------|
| API | `/data/logs/api.log` | API server logs |
| Worker | `/data/logs/worker.log` | Celery worker logs |
| Inference | `/data/logs/inference.log` | Inference service logs |
| PostgreSQL | `/var/log/postgresql/` | Database logs |
| Redis | `/data/redis.log` | Redis logs |

### Log Levels

```bash
# Set log level for API
# Edit infra/docker-compose.yml
# environment:
#   - LOG_LEVEL=DEBUG

# Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Performance Issues

### High Latency

**Diagnosis:**

```bash
# Check API response time
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Check GPU utilization
nvidia-smi

# Check system resources
docker stats
```

**Solutions:**

```bash
# Increase API concurrency
# Edit infra/.env
MAX_INFERENCE_CONCURRENCY=4

# Optimize GPU settings
# Edit infra/.env
GPU_MEMORY_RESERVATION_MB=8192

# Scale API replicas
docker compose up -d --scale api=2
```

### High Memory Usage

**Diagnosis:**

```bash
# Check container memory
docker stats

# Check host memory
free -h

# Check for memory leaks
docker compose exec api python -c "
import psutil
print(f'RSS: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB')
"
```

**Solutions:**

```bash
# Set memory limits in docker-compose.yml
# Edit infra/docker-compose.yml
# services:
#   api:
#     deploy:
#       resources:
#         limits:
#           memory: 4G

# Restart services periodically
docker compose restart api worker
```

### Disk Space Issues

**Diagnosis:**

```bash
# Check disk usage
df -h

# Check Docker disk usage
docker system df

# Check volume sizes
docker system df -v
```

**Solutions:**

```bash
# Clean Docker resources
docker system prune -a

# Remove old logs
find ./logs -name "*.log" -mtime +30 -delete

# Remove old checkpoints
find ./checkpoints -name "*.pt" -mtime +30 -delete
```

## Network Issues

### Service Cannot Connect to Another Service

**Diagnosis:**

```bash
# Check network
docker network inspect ai-model-platform_platform-net

# Test connectivity
docker compose exec api ping postgres
docker compose exec api ping redis
```

**Solutions:**

```bash
# Verify service names in configuration
docker compose config

# Ensure services are on same network
docker compose ps

# Restart networking
docker compose down
docker compose up -d
```

### Port Conflicts

**Symptoms:**
```
Bind for 0.0.0.0:5432 failed: port is already allocated
```

**Solutions:**

```bash
# Check what's using the port
netstat -tulpn | grep 5432

# Change port in .env
POSTGRES_EXTERNAL_PORT=5433

# Or stop conflicting service
sudo systemctl stop postgresql
```

## Recovery Procedures

### Service Recovery

```bash
# Restart a specific service
docker compose restart api

# Force recreate a service
docker compose up -d --force-recreate api

# Rebuild and restart
docker compose up -d --build api
```

### Full System Recovery

```bash
# Stop all services
docker compose down

# Clean up
docker system prune -a

# Restart
docker compose up -d
```

### Data Recovery

```bash
# Restore from backup
# See operations.md for backup/restore procedures

# Check backup integrity
ls -la ./backups/
```

## Getting Help

### Collect Debug Information

```bash
# Generate debug report
docker compose ps > debug.txt
docker compose logs --tail=100 >> debug.txt
docker system info >> debug.txt
nvidia-smi >> debug.txt
```

### Contact Support

1. Collect debug information using the script above
2. Check [GitHub Issues](https://github.com/your-org/ai-model-platform/issues)
3. Contact support with debug information

## Next Steps

- [Installation Guide](installation.md)
- [Configuration Guide](configuration.md)
- [Operations Guide](operations.md)
