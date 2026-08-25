FROM python:3.11

ARG GPU_ENABLED=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY pyproject.toml alembic.ini ./
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip install --no-cache-dir .

COPY app ./app
COPY alembic ./alembic
RUN ln -s /app /app/backend

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
