from functools import lru_cache

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8-sig",
        case_sensitive=False,
        extra="ignore",
    )

    postgres_url: str | None = Field(default=None, validation_alias="POSTGRES_URL")
    postgres_host: str = Field(default="postgres", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(default="ai_platform", validation_alias="POSTGRES_DB")
    postgres_user: str = Field(default="ai_platform", validation_alias="POSTGRES_USER")
    postgres_password: str = Field(
        default="change-me",
        validation_alias="POSTGRES_PASSWORD",
    )
    redis_url: str = Field(
        default="redis://redis:6379/0",
        validation_alias="REDIS_URL",
    )

    model_dir: str = Field(default="/data/models", validation_alias="MODEL_DIR")
    dataset_dir: str = Field(default="/data/datasets", validation_alias="DATASET_DIR")
    log_dir: str = Field(default="/data/logs", validation_alias="LOG_DIR")
    checkpoint_dir: str = Field(
        default="/data/checkpoints",
        validation_alias="CHECKPOINT_DIR",
    )

    gpu_device: str = Field(default="0", validation_alias="GPU_DEVICE")
    gpu_ready: bool = Field(default=False, validation_alias="GPU_READY")
    platform_api_key: str = Field(
        default="change-me",
        validation_alias="PLATFORM_API_KEY",
    )

    max_inference_input_bytes: int = Field(
        default=10 * 1024 * 1024,
        validation_alias="MAX_INFERENCE_INPUT_BYTES",
    )
    max_inference_concurrency: int = Field(
        default=1,
        validation_alias="MAX_INFERENCE_CONCURRENCY",
    )
    gpu_memory_reservation_mb: int = Field(
        default=4096,
        validation_alias="GPU_MEMORY_RESERVATION_MB",
    )
    training_default_epochs: int = Field(
        default=100,
        validation_alias="TRAINING_DEFAULT_EPOCHS",
    )
    training_default_batch_size: int = Field(
        default=16,
        validation_alias="TRAINING_DEFAULT_BATCH_SIZE",
    )

    training_default_imgsz: int = Field(
        default=640,
        validation_alias="TRAINING_DEFAULT_IMGSZ",
    )

    training_default_min_map50: float = Field(
        default=0.0,
        validation_alias="TRAINING_DEFAULT_MIN_MAP50",
    )

    training_default_cache: bool = Field(
        default=False,
        validation_alias="TRAINING_DEFAULT_CACHE",
    )

    training_default_patience: int = Field(
        default=10,
        validation_alias="TRAINING_DEFAULT_PATIENCE",
    )

    training_default_augmentation: str = Field(
        default="default",
        validation_alias="TRAINING_DEFAULT_AUGMENTATION",
    )

    worker_ready: bool = Field(default=False, validation_alias="WORKER_READY")

    @model_validator(mode="after")
    def build_postgres_url(self) -> "Settings":
        if not self.postgres_url:
            self.postgres_url = URL.create(
                drivername="postgresql+psycopg",
                username=self.postgres_user,
                password=self.postgres_password,
                host=self.postgres_host,
                port=self.postgres_port,
                database=self.postgres_db,
            ).render_as_string(hide_password=False)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
