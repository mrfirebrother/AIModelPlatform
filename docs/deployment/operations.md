# Operations Guide

This document covers daily operations, emergency procedures, backup/restore, and version upgrades.

## Daily Operations

### Health Monitoring

```bash
# Check all service status
docker compose ps

# Check API health
curl http://localhost:8000/health

# Check GPU status
nvidia-smi

# Check disk usage
df -h

# Check memory usage
free -h
```

### Log Monitoring

```bash
# Follow API logs
docker compose logs -f api

# Follow worker logs
docker compose logs -f worker

# Check error logs
docker compose logs api | grep -i error

# Check recent logs
docker compose logs --tail=50 api worker inference
```

### Service Management

```bash
# Restart a service
docker compose restart api

# Stop a service
docker compose stop worker

# Start a service
docker compose start worker

# Scale API (if needed)
docker compose up -d --scale api=2
```

### Database Maintenance

```bash
# Connect to database
docker compose exec postgres psql -U ai_platform -d ai_platform

# Check database size
docker compose exec postgres psql -U ai_platform -d ai_platform -c "
SELECT pg_size_pretty(pg_database_size('ai_platform'));
"

# Check table sizes
docker compose exec postgres psql -U ai_platform -d ai_platform -c "
SELECT 
  relname as table_name,
  pg_size_pretty(pg_total_relation_size(relid)) as total_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
"

# Vacuum database
docker compose exec postgres psql -U ai_platform -d ai_platform -c "VACUUM ANALYZE;"
```

## Emergency Procedures

### Service Down

**Immediate Actions:**

```bash
# 1. Check service status
docker compose ps

# 2. Check logs for errors
docker compose logs --tail=100 api

# 3. Restart the affected service
docker compose restart api

# 4. If still down, restart all services
docker compose restart
```

**Escalation:**

```bash
# If restart doesn't work, recreate the service
docker compose up -d --force-recreate api

# If still failing, check dependencies
docker compose ps postgres redis

# Full system restart (last resort)
docker compose down
docker compose up -d
```

### Database Down

**Immediate Actions:**

```bash
# 1. Check PostgreSQL status
docker compose ps postgres

# 2. Check PostgreSQL logs
docker compose logs postgres

# 3. Check disk space
df -h

# 4. Restart PostgreSQL
docker compose restart postgres

# 5. Verify connection
docker compose exec postgres psql -U ai_platform -d ai_platform -c "SELECT 1;"
```

### GPU Failure

**Immediate Actions:**

```bash
# 1. Check GPU status
nvidia-smi

# 2. Check NVIDIA driver
dmesg | grep -i nvidia

# 3. Restart Docker daemon
sudo systemctl restart docker

# 4. Restart GPU services
docker compose restart inference worker

# 5. If GPU hardware failure, switch to CPU mode
# Edit infra/.env
GPU_READY=false
docker compose restart inference worker
```

### Disk Full

**Immediate Actions:**

```bash
# 1. Check disk usage
df -h

# 2. Find large files
du -sh /* | sort -rh | head -10

# 3. Clean Docker resources
docker system prune -a

# 4. Clean old logs
find ./logs -name "*.log" -mtime +7 -delete

# 5. Clean old checkpoints
find ./checkpoints -name "*.pt" -mtime +14 -delete
```

### Memory Full

**Immediate Actions:**

```bash
# 1. Check memory usage
free -h

# 2. Check container memory
docker stats

# 3. Kill memory-hungry processes
docker compose restart worker

# 4. If persistent, restart all services
docker compose restart
```

## Backup and Restore

### Backup Procedures

#### Full System Backup

```bash
#!/bin/bash
# backup.sh - Full system backup

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 1. Backup database
docker compose exec -T postgres pg_dump -U ai_platform ai_platform > "$BACKUP_DIR/database.sql"

# 2. Backup configuration
cp infra/.env "$BACKUP_DIR/"

# 3. Backup models
tar -czf "$BACKUP_DIR/models.tar.gz" ./models/

# 4. Backup datasets
tar -czf "$BACKUP_DIR/datasets.tar.gz" ./datasets/

# 5. Backup checkpoints
tar -czf "$BACKUP_DIR/checkpoints.tar.gz" ./checkpoints/

# 6. Create manifest
cat > "$BACKUP_DIR/manifest.txt" << EOF
Backup Date: $(date)
Database: ai_platform
Models: $(du -sh ./models/)
Datasets: $(du -sh ./datasets/)
Checkpoints: $(du -sh ./checkpoints/)
EOF

echo "Backup completed: $BACKUP_DIR"
```

#### Database Backup

```bash
# Backup database only
docker compose exec -T postgres pg_dump -U ai_platform ai_platform > ./backups/db_$(date +%Y%m%d).sql

# Compressed backup
docker compose exec -T postgres pg_dump -U ai_platform ai_platform | gzip > ./backups/db_$(date +%Y%m%d).sql.gz
```

#### Model Backup

```bash
# Backup models only
tar -czf ./backups/models_$(date +%Y%m%d).tar.gz ./models/
```

### Restore Procedures

#### Full System Restore

```bash
#!/bin/bash
# restore.sh - Full system restore

BACKUP_DIR="$1"

if [ -z "$BACKUP_DIR" ]; then
  echo "Usage: $0 <backup-directory>"
  exit 1
fi

# 1. Stop services
docker compose stop api worker inference

# 2. Restore database
docker compose exec -T postgres psql -U ai_platform ai_platform < "$BACKUP_DIR/database.sql"

# 3. Restore models
tar -xzf "$BACKUP_DIR/models.tar.gz"

# 4. Restore datasets
tar -xzf "$BACKUP_DIR/datasets.tar.gz"

# 5. Restore checkpoints
tar -xzf "$BACKUP_DIR/checkpoints.tar.gz"

# 6. Restore configuration
cp "$BACKUP_DIR/.env" infra/.env

# 7. Start services
docker compose up -d

echo "Restore completed from: $BACKUP_DIR"
```

#### Database Restore

```bash
# Stop API first
docker compose stop api worker

# Restore database
docker compose exec -T postgres psql -U ai_platform ai_platform < ./backups/db_20240101.sql

# Restart services
docker compose start api worker
```

### Backup Schedule

Recommended backup schedule:

| Backup Type | Frequency | Retention |
|-------------|-----------|-----------|
| Full system | Daily | 30 days |
| Database | Every 6 hours | 7 days |
| Models | On change | All versions |
| Configuration | On change | 30 days |

### Backup Verification

```bash
# Verify backup integrity
./scripts/verify_backup.sh ./backups/20240101_120000

# Test restore on staging
./scripts/test_restore.sh ./backups/20240101_120000
```

## Version Upgrades

### Pre-Upgrade Checklist

1. Review release notes for breaking changes
2. Backup current system
3. Test upgrade in staging environment
4. Schedule maintenance window
5. Notify stakeholders

### Upgrade Procedure

#### Step 1: Backup

```bash
# Create full backup
./scripts/backup.sh
```

#### Step 2: Pull New Version

```bash
# Pull latest code
git fetch origin
git checkout v1.2.0  # or desired version

# Pull latest changes
git pull origin main
```

#### Step 3: Update Dependencies

```bash
# Rebuild images
docker compose build --no-cache

# Or pull new images if using registry
docker compose pull
```

#### Step 4: Run Migrations

```bash
# Run database migrations
docker compose run --rm migrate

# Check migration status
docker compose run --rm migrate alembic current
```

#### Step 5: Update Services

```bash
# Rolling update
docker compose up -d --remove-orphans

# Verify health
docker compose ps
curl http://localhost:8000/health
```

#### Step 6: Verify

```bash
# Run smoke tests
./scripts/smoke_test.sh

# Check logs for errors
docker compose logs --tail=50 api worker inference
```

### Rollback Procedure

If upgrade fails:

```bash
# 1. Stop current services
docker compose down

# 2. Restore from backup
./scripts/restore.sh ./backups/pre_upgrade/

# 3. Checkout previous version
git checkout v1.1.0  # previous version

# 4. Rebuild and start
docker compose build
docker compose up -d

# 5. Verify
docker compose ps
curl http://localhost:8000/health
```

### Version Compatibility

| Version | Database | API | Frontend | Notes |
|---------|----------|-----|----------|-------|
| 0.1.0 | v1 | v1 | v1 | Initial release |
| 0.2.0 | v2 | v1 | v1 | Added training features |

## Maintenance Windows

### Recommended Schedule

| Task | Frequency | Duration | Impact |
|------|-----------|----------|--------|
| Database vacuum | Weekly | 10 min | Low |
| Log rotation | Daily | 5 min | None |
| Backup verification | Weekly | 30 min | None |
| Security patches | As needed | 30 min | Low |
| Major upgrades | Monthly | 2 hours | High |

### Maintenance Procedure

```bash
# 1. Enable maintenance mode (if available)
curl -X POST http://localhost:8000/api/admin/maintenance -H "X-API-Key: your-key"

# 2. Perform maintenance
# ...

# 3. Disable maintenance mode
curl -X DELETE http://localhost:8000/api/admin/maintenance -H "X-API-Key: your-key"

# 4. Verify services
docker compose ps
curl http://localhost:8000/health
```

## Monitoring Setup

### Health Check Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `/health/live` | Liveness probe | `{"status": "alive"}` |
| `/health/ready` | Readiness probe | `{"status": "ready"}` |
| `/health` | Full health check | `{"status": "healthy", "checks": {...}}` |

### Metrics Collection

```bash
# Export Prometheus metrics (if configured)
curl http://localhost:8000/metrics
```

### Alerting Rules

Recommended alerts:

- API down for > 5 minutes
- GPU memory usage > 90%
- Disk usage > 85%
- Database connection errors > 10/min
- Training task failures > 5/hour

## Security Operations

### Access Control

```bash
# Rotate API keys
# Edit infra/.env
PLATFORM_API_KEY=<new-api-key>

# Restart services
docker compose restart api worker
```

### Security Patches

```bash
# Update base images
docker compose build --pull --no-cache

# Rebuild all services
docker compose up -d --build
```

### Audit Logging

```bash
# Check access logs
docker compose logs api | grep "X-API-Key"

# Monitor failed authentication
docker compose logs api | grep "401"
```

## Performance Tuning

### API Performance

```bash
# Increase workers
# Edit infra/docker-compose.yml
# services:
#   api:
#     command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Enable caching
# Edit infra/.env
# REDIS_CACHE_TTL=300
```

### Database Performance

```bash
# Optimize queries
docker compose exec postgres psql -U ai_platform -d ai_platform -c "
EXPLAIN ANALYZE SELECT * FROM models WHERE status = 'active';
"

# Add indexes if needed
docker compose exec postgres psql -U ai_platform -d ai_platform -c "
CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);
"
```

### GPU Performance

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Optimize batch size
# Edit infra/.env
# TRAINING_DEFAULT_BATCH_SIZE=32
```

## Next Steps

- [Installation Guide](installation.md)
- [Configuration Guide](configuration.md)
- [Troubleshooting Guide](troubleshooting.md)
