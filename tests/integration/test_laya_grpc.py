import asyncio
import time

import grpc
import pytest
from warden_shared.proto.v1 import laya_pb2, laya_pb2_grpc

from warden_laya.concurrency import ConcurrencyGuard
from warden_laya.grpc_server import create_grpc_server
from warden_laya.model import LayaModelEngine


@pytest.fixture
async def grpc_test_server():
    """Starts an in-process gRPC server bound to an ephemeral localhost port."""
    engine = LayaModelEngine(mock_mode=True)
    guard = ConcurrencyGuard(max_concurrency=16)
    server = create_grpc_server(host="127.0.0.1", port=0, engine=engine, guard=guard)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = laya_pb2_grpc.LayaInferenceServiceStub(channel)

    yield stub, guard, engine

    await channel.close()
    await server.stop(grace=0.5)


@pytest.mark.integration
@pytest.mark.latency_benchmark
@pytest.mark.shedding
@pytest.mark.asyncio
async def test_gate4_grpc_inference_latency_benchmark_under_45ms(grpc_test_server):
    """
    Verifies GATE-4 requirement via live gRPC client/server:
    10-candidate rerank inference latency achieves p95 < 45ms.
    """
    stub, _, _ = grpc_test_server
    candidates = [
        laya_pb2.CandidateItem(
            candidate_id=f"cand-{i:03d}",
            text=f"Policy chunk {i}: employee benefits and bereavement leave rules.",
        )
        for i in range(10)
    ]
    request = laya_pb2.RerankRequest(
        query="How many days of bereavement leave am I entitled to?",
        candidates=candidates,
        primitive="noul",
    )

    # Warm-up and execute 15 iterations
    latencies = []
    for _ in range(15):
        t0 = time.perf_counter()
        response = await stub.Rerank(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert len(response.ranked_results) == 10
        latencies.append(elapsed_ms)

    sorted_latencies = sorted(latencies)
    p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)]
    assert p95_latency < 45.0, f"p95 gRPC latency {p95_latency:.2f}ms exceeded 45ms SLA"


@pytest.mark.integration
@pytest.mark.latency_benchmark
@pytest.mark.shedding
@pytest.mark.asyncio
async def test_gate4_grpc_burst_overload_shedding_at_slot_17(grpc_test_server):
    """
    Verifies GATE-4 requirement via live gRPC client/server:
    Burst of 25 concurrent requests permits 16 executions and sheds 9
    with StatusCode.RESOURCE_EXHAUSTED.
    """
    stub, _, engine = grpc_test_server

    original_forward = engine.forward_batch

    def delayed_forward_batch(query, candidates):
        time.sleep(0.04)  # 40ms simulation
        return original_forward(query, candidates)

    engine.forward_batch = delayed_forward_batch

    request = laya_pb2.RerankRequest(
        query="Concurrent gRPC query",
        candidates=[laya_pb2.CandidateItem(candidate_id="c1", text="Sample policy text.")],
        primitive="noul",
    )

    async def call_rerank():
        try:
            await stub.Rerank(request)
            return "SUCCESS"
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                return "SHED"
            return f"ERROR: {e.code()}"

    results = await asyncio.gather(*[call_rerank() for _ in range(25)])

    success_count = results.count("SUCCESS")
    shed_count = results.count("SHED")

    assert success_count == 16, f"Expected 16 successful gRPC requests, got {success_count}"
    assert shed_count == 9, f"Expected 9 shed gRPC requests, got {shed_count}"
