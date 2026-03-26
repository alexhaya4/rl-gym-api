import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware
from starlette_prometheus import PrometheusMiddleware
from starlette_prometheus.view import metrics as handle_metrics

from app.api.v1 import router as v1_router
from app.api.v1.websockets import router as ws_router
from app.config import get_settings
from app.core.logging import RequestIDMiddleware, configure_logging
from app.core.rate_limit import limiter, rate_limit_exceeded_handler
from app.db.session import get_db
from app.grpc_server.server import start_grpc_server, stop_grpc_server

VERSION = "1.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger = logging.getLogger(__name__)
    settings = get_settings()
    logger.info(f"RL Gym API starting up (environment={settings.ENVIRONMENT})")

    _validate_secret_key(settings, logger)

    # Initialize custom Prometheus metrics (register collectors on import)
    import app.core.prometheus as _  # noqa: F401
    await start_grpc_server(port=settings.GRPC_PORT)
    yield
    await stop_grpc_server()


def _validate_secret_key(settings: Any, logger: logging.Logger) -> None:
    default_key = "change-me-to-a-random-secret-key"
    key = settings.SECRET_KEY
    is_default = key == default_key
    is_short = len(key) < 32
    is_weak = is_default or is_short

    if is_short:
        logger.warning(
            "SECRET_KEY is too short. Generate a secure key with: "
            "python -c 'import secrets; print(secrets.token_hex(32))'"
        )

    if is_default:
        logger.critical(
            "SECRET_KEY is using the default value. "
            "This is a security risk in production."
        )

    if settings.ENVIRONMENT == "production" and is_weak:
        raise RuntimeError(
            "Refusing to start in production with a weak SECRET_KEY. "
            "Set a strong SECRET_KEY (>= 32 characters) in your environment."
        )


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds the configured size limit."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            if int(content_length) > self.max_bytes:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large. Maximum size is 10MB."},
                )
        else:
            body = b""
            async for chunk in request.stream():
                body += chunk
                if len(body) > self.max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Request body too large. Maximum size is 10MB."},
                    )
            # Re-inject the consumed body so downstream handlers can read it
            async def receive() -> dict[str, Any]:
                return {"type": "http.request", "body": body}

            request._receive = receive

        response: Response = await call_next(request)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to every response."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        settings = get_settings()
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fastapi.tiangolo.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fastapi.tiangolo.com; "
            "img-src 'self' data: https://cdn.jsdelivr.net https://fastapi.tiangolo.com"
        )
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        if settings.ENVIRONMENT == "production":
            response.headers["Strict-Transport-Security"] = (
                f"max-age={settings.HSTS_MAX_AGE}; includeSubDomains; preload"
            )
        return response


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

    logger = logging.getLogger(__name__)

    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        if settings.ENVIRONMENT == "production":
            logger.exception("Unhandled exception")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {
                        "loc": list(err.get("loc", [])),
                        "msg": err.get("msg", ""),
                        "type": err.get("type", ""),
                    }
                    for err in exc.errors()
                ]
            },
        )

    app.add_exception_handler(Exception, generic_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]

    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded,
        rate_limit_exceeded_handler,  # type: ignore[arg-type]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(RequestIDMiddleware)
    max_bytes = settings.MAX_REQUEST_SIZE_MB * 1024 * 1024
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=max_bytes)
    app.add_middleware(SecurityHeadersMiddleware)

    app.include_router(v1_router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/api/v1")

    @app.get("/health")
    async def health(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
        from app.core.health import get_health_status

        return await get_health_status(db)

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        if settings.METRICS_TOKEN is not None:
            client_ip = request.client.host if request.client else None
            allowed_ips = {ip.strip() for ip in settings.METRICS_ALLOWED_IPS.split(",")}
            token_ok = request.headers.get("X-Metrics-Token") == settings.METRICS_TOKEN
            ip_ok = client_ip in allowed_ips
            if not token_ok and not ip_ok:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Metrics endpoint requires authentication"},
                )
        return handle_metrics(request)

    return app


app = create_app()
