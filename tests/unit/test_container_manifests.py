import os

import yaml


def test_dockerfile_exists_and_uses_non_root_debian():
    path = "docker/Dockerfile.laya"
    assert os.path.exists(path), f"Expected {path} to exist"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "python:3.12.2-slim-bookworm" in content
    assert "USER 10001:10001" in content
    assert "8000" in content and "50051" in content
    assert "health" in content.lower()


def test_hpa_manifest_conforms_to_spec():
    path = "k8s/hpa/warden-laya-service-hpa.yaml"
    assert os.path.exists(path), f"Expected {path} to exist"
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    assert doc["apiVersion"] == "autoscaling/v2"
    assert doc["kind"] == "HorizontalPodAutoscaler"
    assert doc["metadata"]["name"] == "warden-laya-service-hpa"
    spec = doc["spec"]
    assert spec["scaleTargetRef"]["name"] == "warden-laya-service"
    assert spec["minReplicas"] == 2
    assert spec["maxReplicas"] == 6
    metrics = {
        m["resource"]["name"]: m["resource"]["target"]["averageUtilization"]
        for m in spec["metrics"]
        if m["type"] == "Resource"
    }
    assert metrics["cpu"] == 65
    assert metrics["memory"] == 80
