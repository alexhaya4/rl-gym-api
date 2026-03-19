from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ray_utils import get_ray_address, get_ray_dashboard_url, is_ray_available
from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.ray_training import (
    DistributedTrainingRequest,
    DistributedTrainingResponse,
)
from app.services.ray_training import run_distributed_training

router = APIRouter(prefix="/distributed", tags=["distributed-training"])

# In-memory store for completed job results
_job_results: dict[str, DistributedTrainingResponse] = {}


@router.post("/train", response_model=DistributedTrainingResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_distributed_training(
    request: DistributedTrainingRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> DistributedTrainingResponse:
    """Run distributed hyperparameter search across multiple parallel Ray workers."""
    response = await run_distributed_training(request, db, current_user.id)
    _job_results[response.job_id] = response
    return response


@router.get("/status")
async def ray_cluster_status() -> dict[str, Any]:
    return {
        "ray_available": is_ray_available(),
        "ray_address": get_ray_address(),
        "dashboard_url": get_ray_dashboard_url(),
        "active_trials": 0,
    }


@router.get("/trials/{job_id}", response_model=DistributedTrainingResponse)
async def get_trial_results(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
) -> DistributedTrainingResponse:
    result = _job_results.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return result
