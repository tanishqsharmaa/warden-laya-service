import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI

from warden_shared.errors import register_error_handlers
from warden_laya.config import LayaSettings, get_settings
from warden_laya.model import LayaModelEngine
from warden_laya.concurrency import ConcurrencyGuard
from warden_laya.grpc_server import create_grpc_server
from warden_laya.api import router as api_router

logger = logging.getLogger(__name__)


def create_app(
    mock_mode: bool = False,
    settings: Optional[LayaSettings] = None,
) -> FastAPI:
    """
    Application factory for warden-laya-service.
    Configures FastAPI, RFC 7807 problem details handlers, and shared state.
    """
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: Initialize inference engine and concurrency guard
        logger.info("Initializing warden-laya-service application lifespan...")
        engine = LayaModelEngine(
            model_name_or_path=app_settings.MODEL_NAME_OR_PATH,
            num_threads=app_settings.TORCH_NUM_THREADS,
            backend=app_settings.INFERENCE_BACKEND,
            mock_mode=mock_mode,
        )
        guard = ConcurrencyGuard(max_concurrency=app_settings.MAX_CONCURRENT_REQUESTS)

        app.state.engine = engine
        app.state.guard = guard
        app.state.settings = app_settings

        grpc_server = None
        if not mock_mode:
            logger.info(f"Starting gRPC server on port {app_settings.GRPC_PORT}...")
            grpc_server = create_grpc_server(
                host="0.0.0.0",
                port=app_settings.GRPC_PORT,
                engine=engine,
                guard=guard,
            )
            await grpc_server.start()

        yield

        # Shutdown: Terminate background servers
        if grpc_server is not None:
            logger.info("Stopping gRPC server...")
            await grpc_server.stop(grace=5.0)
        logger.info("warden-laya-service shutdown complete.")

    app = FastAPI(
        title="warden-laya-service",
        description="Non-Generative Decision Modeling Subsystem (Tier 4)",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Register RFC 7807 error handling from warden-shared
    register_error_handlers(app)

    # Attach endpoints
    app.include_router(api_router)

    # For fast test access without lifespan trigger
    engine = LayaModelEngine(
        model_name_or_path=app_settings.MODEL_NAME_OR_PATH,
        num_threads=app_settings.TORCH_NUM_THREADS,
        backend=app_settings.INFERENCE_BACKEND,
        mock_mode=mock_mode,
    )
    guard = ConcurrencyGuard(max_concurrency=app_settings.MAX_CONCURRENT_REQUESTS)
    app.state.engine = engine
    app.state.guard = guard
    app.state.settings = app_settings

    return app


app = create_app()
