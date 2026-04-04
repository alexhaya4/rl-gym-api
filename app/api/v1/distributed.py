from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.distributed import (
    DistributedStatus,
    DistributedTrainRequest,
    DistributedTrainResponse,
)
from app.services.distributed import (
    cancel_job,
    create_job,
    get_cluster_info,
    get_status,
    list_jobs,
)

router = APIRouter(prefix="/distributed", tags=["distributed"])


@router.post("/train", response_model=DistributedTrainResponse, status_code=202)
async def start_distributed_training(
    body: DistributedTrainRequest,
    current_user: User = Depends(get_current_active_user),
) -> DistributedTrainResponse:
    """Start a distributed training job using Ray."""
    try:
        return await create_job(body, current_user.id)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


@router.get("/{job_id}/status", response_model=DistributedStatus)
async def job_status(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
) -> DistributedStatus:
    """Get the status of a distributed training job."""
    try:
        return await get_status(job_id, current_user.id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None


@router.post("/{job_id}/cancel")
async def cancel_distributed_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
) -> dict[str, str]:
    """Cancel a distributed training job."""
    try:
        await cancel_job(job_id, current_user.id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    return {"message": "Job cancellation requested", "job_id": job_id}


@router.get("/jobs", response_model=list[DistributedStatus])
async def list_distributed_jobs(
    current_user: User = Depends(get_current_active_user),
) -> list[DistributedStatus]:
    """List all distributed training jobs for the current user."""
    return await list_jobs(current_user.id)


@router.get("/cluster")
async def cluster_info(
    current_user: User = Depends(get_current_active_user),
) -> dict[str, Any]:
    """Get Ray cluster information."""
    return get_cluster_info()
