import time
import logging
from typing import Optional
import grpc

from warden_shared.proto.v1 import laya_pb2, laya_pb2_grpc
from warden_laya.model import LayaModelEngine
from warden_laya.concurrency import ConcurrencyGuard, ConcurrencyLimitExceededError
from warden_laya.primitives import evaluate_noul, evaluate_choice, evaluate_score

logger = logging.getLogger(__name__)


class LayaGrpcService(laya_pb2_grpc.LayaInferenceServiceServicer):
    """
    gRPC Servicer implementing LayaInferenceService over HTTP/2.
    Serves high-throughput candidate reranking and intent routing.
    """

    def __init__(self, engine: LayaModelEngine, guard: ConcurrencyGuard) -> None:
        self.engine = engine
        self.guard = guard

    async def Rerank(
        self,
        request: laya_pb2.RerankRequest,
        context: Optional[grpc.aio.ServicerContext],
    ) -> laya_pb2.RerankResponse:
        """
        Executes candidate reranking using calibrated decision primitives.
        """
        try:
            async with self.guard:
                t0 = time.perf_counter()
                candidates = [(c.candidate_id, c.text) for c in request.candidates]

                if request.primitive == "score":
                    ranked = evaluate_score(self.engine, request.query, candidates)
                else:
                    ranked = evaluate_noul(self.engine, request.query, candidates)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                ranked_items = [
                    laya_pb2.RankedCandidate(
                        candidate_id=cand_id,
                        probability_true=prob,
                        rank=r,
                    )
                    for cand_id, prob, r in ranked
                ]

                return laya_pb2.RerankResponse(
                    query=request.query,
                    ranked_results=ranked_items,
                    inference_latency_ms=elapsed_ms,
                )
        except ConcurrencyLimitExceededError as e:
            logger.warning("Rerank rejected due to saturated inference slots")
            if context is not None:
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(e))
            raise

    async def Route(
        self,
        request: laya_pb2.RouteRequest,
        context: Optional[grpc.aio.ServicerContext],
    ) -> laya_pb2.RouteResponse:
        """
        Executes intent routing across discrete string choices.
        """
        try:
            async with self.guard:
                if len(request.choices) < 2:
                    if context is not None:
                        context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "choices must contain at least 2 distinct options",
                        )
                    raise ValueError("choices must contain at least 2 distinct options")

                t0 = time.perf_counter()
                selected_choice, conf_dist = evaluate_choice(
                    self.engine, request.query, list(request.choices)
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                return laya_pb2.RouteResponse(
                    query=request.query,
                    selected_choice=selected_choice,
                    confidence_distribution=conf_dist,
                    inference_latency_ms=elapsed_ms,
                )
        except ConcurrencyLimitExceededError as e:
            logger.warning("Route rejected due to saturated inference slots")
            if context is not None:
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(e))
            raise


def create_grpc_server(
    host: str,
    port: int,
    engine: LayaModelEngine,
    guard: ConcurrencyGuard,
) -> grpc.aio.Server:
    """
    Constructs and configures an asynchronous gRPC server with tuned keepalive options.
    """
    options = [
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 5000),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.max_receive_message_length", 32 * 1024 * 1024),
        ("grpc.max_send_message_length", 32 * 1024 * 1024),
    ]
    server = grpc.aio.server(options=options)
    servicer = LayaGrpcService(engine=engine, guard=guard)
    laya_pb2_grpc.add_LayaInferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    return server
