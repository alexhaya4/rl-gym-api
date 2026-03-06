import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from app.api.v1 import router as v1_router
from app.api.v1.websockets import router as ws_router
from app.config import get_settings
from app.core.logging import RequestIDMiddleware, configure_logging
from app.core.rate_limit import limiter, rate_limit_exceeded_handler

VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger = logging.getLogger(__name__)
    settings = get_settings()
    logger.info(f"RL Gym API starting up (environment={settings.ENVIRONMENT})")
    yield


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title="RL Gym API",
        description=(
            "A production-ready RESTful API for reinforcement learning experiments "
            "built with FastAPI, OpenAI Gymnasium, and Stable Baselines3.\n\n"
            "## Key Features\n\n"
            "- **Environments** — Create, reset, step, and manage Gymnasium environments\n"
            "- **Training** — Train RL agents with PPO, A2C, and DQN algorithms\n"
            "- **Experiments** — Track and manage experiment lifecycle and results\n"
            "- **Benchmarking** — Compare algorithms across multiple environments\n"
            "- **WebSockets** — Real-time training metrics streaming\n"
            "- **JWT Authentication** — Secure endpoints with token-based auth\n"
        ),
        version=VERSION,
        lifespan=lifespan,
        terms_of_service="https://rlgymapi.dev/terms",
        contact={
            "name": "RL Gym API Team",
            "email": "support@rlgymapi.dev",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        openapi_tags=[
            {"name": "auth", "description": "User registration and JWT authentication"},
            {"name": "environments", "description": "Gymnasium environment management"},
            {"name": "training", "description": "RL agent training and evaluation"},
            {"name": "experiments", "description": "Experiment tracking and lifecycle management"},
            {"name": "benchmarks", "description": "Algorithm benchmarking across environments"},
            {"name": "websockets", "description": "Real-time training metrics via WebSocket"},
            {"name": "status", "description": "API status and health checks"},
        ],
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)

    app.include_router(v1_router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/api/v1")

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": VERSION}

    return app


app = create_app()
