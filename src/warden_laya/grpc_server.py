import asyncio
import logging
import time
from typing import Optional

import grpc
from warden_shared.proto.v1.laya_pb2 import (  # type: ignore[attr-defined]
    RankedCandidate,
    RerankRequest,
    RerankResponse,
    RouteRequest,
    RouteResponse,
)
from warden_shared.proto.v1.laya_pb2_grpc import (
    LayaInferenceServiceServicer,
    add_LayaInferenceServiceServicer_to_server,
)

from warden_laya.concurrency import ConcurrencyGuard, ConcurrencyLimitExceededError
from warden_laya.model import LayaModelEngine
from warden_laya.primitives import evaluate_choice, evaluate_noul, evaluate_score

logger = logging.getLogger(__name__)


class LayaGrpcService(LayaInferenceServiceServicer):
    """
    gRPC Servicer implementing LayaInferenceService over HTTP/2.
    Serves high-throughput candidate reranking and intent routing.
    """

    def __init__(self, engine: LayaModelEngine, guard: ConcurrencyGuard) -> None:
        self.engine = engine
        self.guard = guard

    async def Rerank(
        self,
        request: RerankRequest,
        context: Optional[grpc.aio.ServicerContext],
    ) -> RerankResponse:
        """
        Executes candidate reranking using calibrated decision primitives.
        """
        try:
            async with self.guard:
                t0 = time.perf_counter()
                candidates = [(c.candidate_id, c.text) for c in request.candidates]

                if request.primitive == "score":
                    ranked = await asyncio.to_thread(
                        evaluate_score, self.engine, request.query, candidates
                    )
                else:
                    ranked = await asyncio.to_thread(
                        evaluate_noul, self.engine, request.query, candidates
                    )

                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                ranked_items = [
                    RankedCandidate(
                        candidate_id=cand_id,
                        probability_true=prob,
                        rank=r,
                    )
                    for cand_id, prob, r in ranked
                ]

                return RerankResponse(
                    query=request.query,
                    ranked_results=ranked_items,
                    inference_latency_ms=elapsed_ms,
                )
        except ConcurrencyLimitExceededError as e:
            logger.warning("Rerank rejected due to saturated inference slots")
            if context is not None:
                abort_res = context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(e))
                if asyncio.iscoroutine(abort_res):
                    await abort_res
                return RerankResponse()
            raise

    async def Route(
        self,
        request: RouteRequest,
        context: Optional[grpc.aio.ServicerContext],
    ) -> RouteResponse:
        """
        Executes intent routing across discrete string choices.
        """
        try:
            async with self.guard:
                if len(request.choices) < 2:
                    if context is not None:
                        abort_res = context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "choices must contain at least 2 distinct options",
                        )
                        if asyncio.iscoroutine(abort_res):
                            await abort_res
                        return RouteResponse()
                    raise ValueError("choices must contain at least 2 distinct options")

                t0 = time.perf_counter()
                selected_choice, conf_dist = await asyncio.to_thread(
                    evaluate_choice, self.engine, request.query, list(request.choices)
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                return RouteResponse(
                    query=request.query,
                    selected_choice=selected_choice,
                    confidence_distribution=conf_dist,
                    inference_latency_ms=elapsed_ms,
                )
        except ConcurrencyLimitExceededError as e:
            logger.warning("Route rejected due to saturated inference slots")
            if context is not None:
                abort_res = context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(e))
                if asyncio.iscoroutine(abort_res):
                    await abort_res
                return RouteResponse()
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
    add_LayaInferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    return server
