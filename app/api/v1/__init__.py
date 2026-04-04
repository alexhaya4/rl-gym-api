import time
from typing import Any

from fastapi import APIRouter

from app.api.v1.ab_testing import router as ab_testing_router
from app.api.v1.algorithms import router as algorithms_router
from app.api.v1.artifacts import router as artifacts_router
from app.api.v1.audit import router as audit_router
from app.api.v1.auth import router as auth_router
from app.api.v1.benchmarks import router as benchmarks_router
from app.api.v1.billing import router as billing_router
from app.api.v1.comparison import router as comparison_router
from app.api.v1.custom_environments import router as custom_environments_router
from app.api.v1.datasets import router as datasets_router
from app.api.v1.environments import router as environments_router
from app.api.v1.evaluation import router as evaluation_router
from app.api.v1.experiments import router as experiments_router
from app.api.v1.inference import router as inference_router
from app.api.v1.ml import router as ml_router
from app.api.v1.models import router as models_router
from app.api.v1.multi_agent import router as multi_agent_router
from app.api.v1.oauth import router as oauth_router
from app.api.v1.optimization import router as optimization_router
from app.api.v1.organizations import router as organizations_router
from app.api.v1.pbt import router as pbt_router
from app.api.v1.pipelines import router as pipelines_router
from app.api.v1.ray_training import router as ray_training_router
from app.api.v1.rbac import router as rbac_router
from app.api.v1.registry import router as registry_router
from app.api.v1.training import router as training_router
from app.api.v1.vec_environments import router as vec_environments_router
from app.api.v1.video import router as video_router
from app.api.v1.websockets import router as ws_router
from app.config import get_settings
from app.services.environment import _environments

_start_time = time.time()

router = APIRouter()
router.include_router(ab_testing_router)
router.include_router(algorithms_router)
router.include_router(artifacts_router)
router.include_router(audit_router)
router.include_router(auth_router)
router.include_router(comparison_router)
router.include_router(custom_environments_router)
router.include_router(datasets_router)
router.include_router(environments_router)
router.include_router(training_router)
router.include_router(experiments_router)
router.include_router(inference_router)
router.include_router(ml_router)
router.include_router(models_router)
router.include_router(multi_agent_router)
router.include_router(oauth_router)
router.include_router(organizations_router)
router.include_router(billing_router)
router.include_router(ray_training_router)
router.include_router(benchmarks_router)
router.include_router(evaluation_router)
router.include_router(optimization_router)
router.include_router(pbt_router)
router.include_router(pipelines_router)
router.include_router(rbac_router)
router.include_router(registry_router)
router.include_router(vec_environments_router)
router.include_router(video_router)
router.include_router(ws_router)


@router.get("/status", tags=["status"])
async def api_status() -> dict[str, Any]:
    """Return current API status including version, environment, and uptime."""
    settings = get_settings()
    return {
        "api_version": "0.1.0",
        "environment": settings.ENVIRONMENT,
        "active_environments": len(_environments),
        "uptime_seconds": round(time.time() - _start_time, 2),
    }


@router.get("/status/grpc", tags=["status"])
async def grpc_status() -> dict[str, Any]:
    """Return gRPC inference server status."""
    settings = get_settings()
    return {
        "grpc_host": settings.GRPC_HOST,
        "grpc_port": settings.GRPC_PORT,
        "status": "running",
    }
