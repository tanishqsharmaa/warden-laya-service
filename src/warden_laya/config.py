from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class LayaSettings(BaseSettings):
    """
    Typed settings matrix for warden-laya-service.
    Governs model execution, thread allocations, concurrency ceilings, and ports.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    MODEL_NAME_OR_PATH: str = "convaiinnovations/laya"
    INFERENCE_BACKEND: str = "onnx_runtime"
    TORCH_NUM_THREADS: int = 2
    TORCH_DTYPE: str = "bfloat16"
    MAX_SEQUENCE_LENGTH: int = 1024
    MAX_CONCURRENT_REQUESTS: int = 16
    PORT: int = 8000
    GRPC_PORT: int = 50051
    LOG_LEVEL: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> LayaSettings:
    """Return a cached singleton instance of LayaSettings."""
    return LayaSettings()
