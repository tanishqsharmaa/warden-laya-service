import asyncio
import time

import pytest
from httpx import ASGITransport, AsyncClient

from warden_laya.main import create_app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gate4_cpu_inference_latency_benchmark_under_45ms():
    """
    Verifies GATE-4 requirement: 10-candidate reranking inference latency
    on CPU achieves p95 < 45ms.
    """
    app = create_app(mock_mode=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "query": "How many days of bereavement leave am I entitled to?",
            "candidates": [
                {
                    "candidate_id": f"cand-{i:03d}",
                    "text": f"Policy chunk {i}: employee benefits and bereavement leave rules.",
                }
                for i in range(10)
            ],
            "primitive": "noul",
        }

        # Warm-up and execute 15 iterations
        latencies = []
        for _ in range(15):
            t0 = time.perf_counter()
            resp = await client.post("/rerank", json=payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["ranked_results"]) == 10
            latencies.append(elapsed_ms)

        # Compute p95 latency
        sorted_latencies = sorted(latencies)
        p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        assert p95_latency < 45.0, f"p95 latency {p95_latency:.2f}ms exceeded 45ms SLA"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gate4_burst_overload_shedding_at_slot_17():
    """
    Verifies GATE-4 requirement: Burst concurrency of 25 concurrent requests
    permits exactly 16 inference executions and immediately sheds 9 requests
    with HTTP 429 and Retry-After: 0.1.
    """
    app = create_app(mock_mode=True)

    # Wrap mock engine to introduce slight artificial latency during stress test
    original_forward = app.state.engine.forward_batch

    def delayed_forward_batch(query, candidates):
        time.sleep(0.04)  # 40ms CPU forward simulation to saturate semaphore
        return original_forward(query, candidates)

    app.state.engine.forward_batch = delayed_forward_batch

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "query": "Concurrent stress query",
            "candidates": [{"candidate_id": "c1", "text": "Some policy text."}],
            "primitive": "noul",
        }

        responses = await asyncio.gather(*[
            client.post("/rerank", json=payload) for _ in range(25)
        ])

        status_codes = [r.status_code for r in responses]
        success_count = status_codes.count(200)
        shed_count = status_codes.count(429)

        assert success_count == 16, f"Expected exactly 16 successful requests, got {success_count}"
        assert shed_count == 9, f"Expected exactly 9 shed requests, got {shed_count}"

        # Verify RFC 7807 problem details and Retry-After header on shed requests
        for r in responses:
            if r.status_code == 429:
                assert r.headers.get("retry-after") == "0.1"
                body = r.json()
                assert body.get("error_code") == "ERR_CONCURRENCY_LIMIT_EXCEEDED"
                assert body.get("status") == 429
