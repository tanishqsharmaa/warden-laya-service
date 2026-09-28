import pytest
from httpx import AsyncClient, ASGITransport
from warden_laya.main import create_app


@pytest.fixture
def app():
    return create_app(mock_mode=True)


@pytest.mark.asyncio
async def test_health_probe_returns_healthy(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "HEALTHY"
        assert data["service"] == "warden-laya-service"
        assert data["model_loaded"] == "convaiinnovations/laya"
        assert data["backbone"] == "ModernBERT-large"
        assert data["parameters"] == "421M"
        assert data["max_concurrency"] == 16
        assert "active_inference_slots" in data


@pytest.mark.asyncio
async def test_rest_rerank_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "query": "How many days of bereavement leave am I entitled to?",
            "candidates": [
                {"candidate_id": "cand-001", "text": "Up to 5 days bereavement leave."},
                {"candidate_id": "cand-002", "text": "Parking subsidies available."}
            ],
            "primitive": "noul"
        }
        resp = await client.post("/rerank", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["primitive_used"] == "noul"
        assert len(data["ranked_results"]) == 2
        assert data["ranked_results"][0]["rank"] == 1


@pytest.mark.asyncio
async def test_rest_route_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "query": "Can you draft a legal contract for my external freelance business?",
            "choices": ["HR_POLICY", "OUT_OF_SCOPE"]
        }
        resp = await client.post("/route", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["selected_choice"] in ["HR_POLICY", "OUT_OF_SCOPE"]
        assert len(data["confidence_distribution"]) == 2


@pytest.mark.asyncio
async def test_rest_route_endpoint_invalid_choices(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "query": "Some query",
            "choices": ["ONLY_ONE"]
        }
        resp = await client.post("/route", json=payload)
        assert resp.status_code in [400, 422]
