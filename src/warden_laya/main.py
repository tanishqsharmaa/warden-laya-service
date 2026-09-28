import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from warden_shared.errors import register_error_handlers

from warden_laya.api import router as api_router
from warden_laya.concurrency import ConcurrencyGuard
from warden_laya.config import LayaSettings, get_settings
from warden_laya.grpc_server import create_grpc_server
from warden_laya.model import LayaModelEngine

logger = logging.getLogger(__name__)


def create_app(
    mock_mode: bool = False,
    settings: Optional[LayaSettings] = None,
    engine: Optional[LayaModelEngine] = None,
    guard: Optional[ConcurrencyGuard] = None,
) -> FastAPI:
    """
    Application factory for warden-laya-service.
    Configures FastAPI, RFC 7807 problem details handlers, and shared state.
    """
    app_settings = settings or get_settings()

    # Shared single engine and guard instances (prevents memory doubling)
    resolved_engine = engine or LayaModelEngine(
        model_name_or_path=app_settings.MODEL_NAME_OR_PATH,
        num_threads=app_settings.TORCH_NUM_THREADS,
        backend=app_settings.INFERENCE_BACKEND,
        mock_mode=mock_mode,
    )
    resolved_guard = guard or ConcurrencyGuard(
        max_concurrency=app_settings.MAX_CONCURRENT_REQUESTS
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: Attach state and start gRPC server if live
        logger.info("Initializing warden-laya-service application lifespan...")
        app.state.engine = resolved_engine
        app.state.guard = resolved_guard
        app.state.settings = app_settings

        grpc_server = None
        if not resolved_engine.mock_mode:
            logger.info(f"Starting gRPC server on port {app_settings.GRPC_PORT}...")
            grpc_server = create_grpc_server(
                host="0.0.0.0",
                port=app_settings.GRPC_PORT,
                engine=resolved_engine,
                guard=resolved_guard,
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

    # Attach shared state
    app.state.engine = resolved_engine
    app.state.guard = resolved_guard
    app.state.settings = app_settings

    return app


app = create_app()
