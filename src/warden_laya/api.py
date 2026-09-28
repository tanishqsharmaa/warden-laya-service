import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
import psutil
from fastapi import APIRouter, Request, status, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from warden_laya.concurrency import ConcurrencyLimitExceededError
from warden_laya.primitives import evaluate_noul, evaluate_choice, evaluate_score

logger = logging.getLogger(__name__)
router = APIRouter()


class CandidateInput(BaseModel):
    candidate_id: str
    text: str


class RerankRequestPayload(BaseModel):
    query: str
    candidates: List[CandidateInput] = Field(default_factory=list)
    primitive: str = Field(default="noul", description="'noul' or 'score'")


class RankedResultItem(BaseModel):
    candidate_id: str
    probability_true: float
    rank: int


class RerankResponsePayload(BaseModel):
    query: str
    primitive_used: str
    ranked_results: List[RankedResultItem]
    inference_latency_ms: float


class RouteRequestPayload(BaseModel):
    query: str
    choices: List[str]

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, v: List[str]) -> List[str]:
        if len(v) < 2:
            raise ValueError("choices must contain at least 2 distinct options")
        return v


class RouteResponsePayload(BaseModel):
    query: str
    primitive_used: str = "choice"
    selected_choice: str
    confidence_distribution: Dict[str, float]
    inference_latency_ms: float


class HealthResponsePayload(BaseModel):
    status: str
    service: str
    model_loaded: str
    backbone: str
    parameters: str
    runtime_device: str
    inference_backend: str
    precision: str
    active_inference_slots: int
    max_concurrency: int
    memory_rss_mb: float
    timestamp: str


@router.post(
    "/rerank",
    response_model=RerankResponsePayload,
    status_code=status.HTTP_200_OK,
)
async def rerank_endpoint(payload: RerankRequestPayload, request: Request):
    engine = request.app.state.engine
    guard = request.app.state.guard

    try:
        async with guard:
            t0 = time.perf_counter()
            candidates = [(c.candidate_id, c.text) for c in payload.candidates]

            if payload.primitive == "score":
                ranked = evaluate_score(engine, payload.query, candidates)
            else:
                ranked = evaluate_noul(engine, payload.query, candidates)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            results = [
                RankedResultItem(
                    candidate_id=cand_id,
                    probability_true=round(prob, 4),
                    rank=r,
                )
                for cand_id, prob, r in ranked
            ]

            return RerankResponsePayload(
                query=payload.query,
                primitive_used=payload.primitive,
                ranked_results=results,
                inference_latency_ms=round(elapsed_ms, 2),
            )
    except ConcurrencyLimitExceededError as e:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "type": "https://warden.enterprise/errors/concurrency-limit-exceeded",
                "title": "Concurrency Limit Exceeded",
                "status": 429,
                "detail": str(e),
                "error_code": "ERR_CONCURRENCY_LIMIT_EXCEEDED",
                "instance": "/rerank",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            headers={"Retry-After": str(e.retry_after)},
        )


@router.post(
    "/route",
    response_model=RouteResponsePayload,
    status_code=status.HTTP_200_OK,
)
async def route_endpoint(payload: RouteRequestPayload, request: Request):
    engine = request.app.state.engine
    guard = request.app.state.guard

    try:
        async with guard:
            t0 = time.perf_counter()
            selected_choice, conf_dist = evaluate_choice(
                engine, payload.query, payload.choices
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Round distribution floats for clean JSON
            rounded_dist = {k: round(v, 4) for k, v in conf_dist.items()}

            return RouteResponsePayload(
                query=payload.query,
                primitive_used="choice",
                selected_choice=selected_choice,
                confidence_distribution=rounded_dist,
                inference_latency_ms=round(elapsed_ms, 2),
            )
    except ConcurrencyLimitExceededError as e:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "type": "https://warden.enterprise/errors/concurrency-limit-exceeded",
                "title": "Concurrency Limit Exceeded",
                "status": 429,
                "detail": str(e),
                "error_code": "ERR_CONCURRENCY_LIMIT_EXCEEDED",
                "instance": "/route",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            headers={"Retry-After": str(e.retry_after)},
        )


@router.get(
    "/health",
    response_model=HealthResponsePayload,
    status_code=status.HTTP_200_OK,
)
async def health_endpoint(request: Request):
    engine = request.app.state.engine
    guard = request.app.state.guard
    settings = request.app.state.settings

    process = psutil.Process()
    memory_info = process.memory_info()
    rss_mb = memory_info.rss / (1024 * 1024)

    return HealthResponsePayload(
        status="HEALTHY" if engine.is_loaded else "DEGRADED",
        service="warden-laya-service",
        model_loaded="convaiinnovations/laya",
        backbone="ModernBERT-large",
        parameters="421M",
        runtime_device="cpu",
        inference_backend=settings.INFERENCE_BACKEND,
        precision=settings.TORCH_DTYPE,
        active_inference_slots=guard.active_slots,
        max_concurrency=guard.max_concurrency,
        memory_rss_mb=round(rss_mb, 1),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
