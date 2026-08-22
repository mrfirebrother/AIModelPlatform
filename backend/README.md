# Backend Runtime Skeleton

This directory contains the Task 1 FastAPI, Celery, database-check, migration,
and container runtime skeleton. Model training, datasets, model registry, and
business API routes are intentionally not implemented here.

## BOM-Aware Commands

The workspace requires UTF-8 BOM on text files. The local TOML/INI parsers may
reject that BOM, so use the following commands:

```text
python -m pytest -c NUL -p no:cacheprovider backend/tests/unit/test_health.py -q
python backend/scripts/verify_bom.py
python -m compileall backend
```

Do not assume ordinary local pytest or Alembic commands can parse the BOM-aware
configuration files. The Dockerfile strips the BOM from `pyproject.toml` and
`alembic.ini` inside the image before package installation or migration use.
