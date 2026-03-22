import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.pipeline import PipelineRunRequest, PipelineRunResponse
from app.services.pipeline_store import (
    get_pipeline_run,
    list_pipeline_runs,
    store_pipeline_run,
    update_pipeline_run,
)

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


async def _run_training_pipeline(
    pipeline_id: str, request_dict: dict, user_id: int
) -> None:
    from app.pipelines.flows import rl_training_pipeline

    result = await rl_training_pipeline(request_dict, user_id, pipeline_id)
    update_pipeline_run(pipeline_id, result)


async def _run_search_pipeline(
    pipeline_id: str, request_dict: dict, user_id: int
) -> None:
    from app.pipelines.flows import hyperparameter_search_pipeline

    result = await hyperparameter_search_pipeline(request_dict, user_id, pipeline_id)
    update_pipeline_run(pipeline_id, result)


@router.get("/health", response_model=dict[str, Any])
async def pipeline_health() -> dict[str, Any]:
    """Check Prefect connectivity."""
    try:
        import prefect

        return {"prefect_available": True, "version": prefect.__version__}
    except ImportError:
        return {"prefect_available": False, "version": ""}


@router.post("/run", response_model=PipelineRunResponse, status_code=202)
async def run_pipeline(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
) -> PipelineRunResponse:
    """Trigger a full RL training pipeline."""
    pipeline_id = str(uuid.uuid4())
    experiment_name = request.experiment_name or (
        f"{request.algorithm}-{request.environment_id}"
    )
    started_at = datetime.now(UTC).isoformat()

    initial = PipelineRunResponse(
        pipeline_id=pipeline_id,
        experiment_name=experiment_name,
        status="pending",
        started_at=started_at,
    )
    store_pipeline_run(pipeline_id, {**initial.model_dump(), "user_id": current_user.id})

    request_dict = request.model_dump()
    background_tasks.add_task(
        _run_training_pipeline, pipeline_id, request_dict, current_user.id
    )

    return initial


@router.post("/search", response_model=PipelineRunResponse, status_code=202)
async def search_pipeline(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
) -> PipelineRunResponse:
    """Trigger a hyperparameter search pipeline."""
    pipeline_id = str(uuid.uuid4())
    experiment_name = request.experiment_name or (
        f"hpsearch-{request.algorithm}-{request.environment_id}"
    )
    started_at = datetime.now(UTC).isoformat()

    initial = PipelineRunResponse(
        pipeline_id=pipeline_id,
        experiment_name=experiment_name,
        status="pending",
        started_at=started_at,
    )
    store_pipeline_run(pipeline_id, {**initial.model_dump(), "user_id": current_user.id})

    request_dict = request.model_dump()
    background_tasks.add_task(
        _run_search_pipeline, pipeline_id, request_dict, current_user.id
    )

    return initial


@router.get("/", response_model=list[PipelineRunResponse])
async def list_pipelines(
    current_user: User = Depends(get_current_active_user),
) -> list[PipelineRunResponse]:
    """List all pipeline runs for the current user."""
    runs = list_pipeline_runs(current_user.id)
    return [PipelineRunResponse(**{k: v for k, v in run.items() if k != "user_id"}) for run in runs]


@router.get("/{pipeline_id}", response_model=PipelineRunResponse)
async def get_pipeline(
    pipeline_id: str,
    current_user: User = Depends(get_current_active_user),
) -> PipelineRunResponse:
    """Get pipeline run status and results."""
    run = get_pipeline_run(pipeline_id)
    if run is None or run.get("user_id") != current_user.id:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return PipelineRunResponse(**{k: v for k, v in run.items() if k != "user_id"})
