import time

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.benchmarks import router as benchmarks_router
from app.api.v1.environments import router as environments_router
from app.api.v1.experiments import router as experiments_router
from app.api.v1.training import router as training_router
from app.api.v1.websockets import router as ws_router
from app.config import get_settings
from app.services.environment import _environments

_start_time = time.time()

router = APIRouter()
router.include_router(auth_router)
router.include_router(environments_router)
router.include_router(training_router)
router.include_router(experiments_router)
router.include_router(benchmarks_router)
router.include_router(ws_router)


@router.get("/status", tags=["status"])
async def api_status():
    """Return current API status including version, environment, and uptime."""
    settings = get_settings()
    return {
        "api_version": "0.1.0",
        "environment": settings.ENVIRONMENT,
        "active_environments": len(_environments),
        "uptime_seconds": round(time.time() - _start_time, 2),
    }
