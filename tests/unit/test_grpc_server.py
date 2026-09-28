import pytest
import grpc
from warden_shared.proto.v1 import laya_pb2, laya_pb2_grpc
from warden_laya.model import LayaModelEngine
from warden_laya.concurrency import ConcurrencyGuard
from warden_laya.grpc_server import LayaGrpcService


class MockGrpcContext:
    def __init__(self):
        self.aborted = False
        self.code = None
        self.details = None

    def abort(self, code, details):
        self.aborted = True
        self.code = code
        self.details = details
        raise grpc.RpcError(f"RPC Aborted: {code} - {details}")


@pytest.fixture
def grpc_service():
    engine = LayaModelEngine(mock_mode=True)
    guard = ConcurrencyGuard(max_concurrency=16)
    return LayaGrpcService(engine=engine, guard=guard)


@pytest.mark.asyncio
async def test_grpc_rerank_rpc_success(grpc_service):
    request = laya_pb2.RerankRequest(
        query="What is the bereavement leave policy?",
        candidates=[
            laya_pb2.CandidateItem(candidate_id="c1", text="Up to 5 days bereavement leave."),
            laya_pb2.CandidateItem(candidate_id="c2", text="Parking subsidies.")
        ],
        primitive="noul"
    )
    context = MockGrpcContext()
    response = await grpc_service.Rerank(request, context)
    assert response.query == request.query
    assert len(response.ranked_results) == 2
    assert response.ranked_results[0].rank == 1
    assert response.ranked_results[1].rank == 2
    assert response.inference_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_grpc_route_rpc_success(grpc_service):
    request = laya_pb2.RouteRequest(
        query="Can you draft a freelance contract?",
        choices=["HR_POLICY", "OUT_OF_SCOPE"]
    )
    context = MockGrpcContext()
    response = await grpc_service.Route(request, context)
    assert response.selected_choice in ["HR_POLICY", "OUT_OF_SCOPE"]
    assert len(response.confidence_distribution) == 2
    assert response.inference_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_grpc_route_rpc_invalid_argument_under_two_choices(grpc_service):
    request = laya_pb2.RouteRequest(
        query="Some query",
        choices=["ONLY_ONE"]
    )
    context = MockGrpcContext()
    with pytest.raises(grpc.RpcError):
        await grpc_service.Route(request, context)
    assert context.aborted is True
    assert context.code == grpc.StatusCode.INVALID_ARGUMENT
