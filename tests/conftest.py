import pytest


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Ensure test runs start with a clean configuration environment."""
    monkeypatch.delenv("MODEL_NAME_OR_PATH", raising=False)
    monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
    monkeypatch.delenv("TORCH_NUM_THREADS", raising=False)
    monkeypatch.delenv("TORCH_DTYPE", raising=False)
    monkeypatch.delenv("MAX_SEQUENCE_LENGTH", raising=False)
    monkeypatch.delenv("MAX_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("GRPC_PORT", raising=False)
