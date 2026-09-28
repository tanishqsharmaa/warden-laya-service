import pytest
from warden_laya.config import LayaSettings, get_settings

def test_default_configuration_matrix():
    settings = LayaSettings()
    assert settings.MODEL_NAME_OR_PATH == "convaiinnovations/laya"
    assert settings.INFERENCE_BACKEND == "onnx_runtime"
    assert settings.TORCH_NUM_THREADS == 2
    assert settings.TORCH_DTYPE == "bfloat16"
    assert settings.MAX_SEQUENCE_LENGTH == 1024
    assert settings.MAX_CONCURRENT_REQUESTS == 16
    assert settings.PORT == 8000
    assert settings.GRPC_PORT == 50051

def test_singleton_settings_caching():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2

def test_environment_variable_override(monkeypatch):
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "32")
    settings = LayaSettings()
    assert settings.PORT == 9000
    assert settings.MAX_CONCURRENT_REQUESTS == 32
