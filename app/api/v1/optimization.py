import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.optuna import OptimizationRequest, OptimizationResponse
from app.services.optuna_optimization import (
    DEFAULT_HYPERPARAMETER_SPACES,
    get_optimization_history,
    get_study,
    list_studies,
    run_optimization,
)

router = APIRouter(prefix="/optimization", tags=["optimization"])


async def _run_optimization_bg(
    request: OptimizationRequest, user_id: int
) -> None:
    from app.db.session import AsyncSessionLocal

    db = AsyncSessionLocal()
    try:
        await run_optimization(db, request, user_id)
    finally:
        await db.close()


@router.get("/algorithms/spaces", response_model=dict[str, Any])
async def get_algorithm_spaces() -> dict[str, Any]:
    """Get default hyperparameter search spaces for each algorithm."""
    return DEFAULT_HYPERPARAMETER_SPACES


@router.post("/run", response_model=OptimizationResponse, status_code=202)
async def start_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> OptimizationResponse:
    """Run Bayesian hyperparameter optimization using Optuna TPE sampler."""
    study_id = str(uuid.uuid4())
    experiment_name = request.experiment_name or (
        f"optuna-{request.algorithm}-{request.environment_id}"
    )
    started_at = datetime.now(UTC).isoformat()

    initial = OptimizationResponse(
        study_id=study_id,
        experiment_name=experiment_name,
        status="running",
        n_trials=request.n_trials,
        n_completed=0,
        n_pruned=0,
        best_trial=None,
        best_hyperparameters=None,
        best_mean_reward=None,
        started_at=started_at,
    )

    background_tasks.add_task(_run_optimization_bg, request, current_user.id)

    return initial


@router.get("/", response_model=list[OptimizationResponse])
async def list_optimization_studies(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[OptimizationResponse]:
    """List all optimization studies for the current user."""
    return await list_studies(db, current_user.id)


@router.get("/{study_id}", response_model=OptimizationResponse)
async def get_optimization_study(
    study_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> OptimizationResponse:
    """Get optimization study status and results."""
    result = await get_study(db, study_id, current_user.id)
    if result is None:
        raise HTTPException(status_code=404, detail="Study not found")
    return result


@router.get("/{study_id}/history", response_model=list[dict[str, Any]])
async def get_study_history(
    study_id: str,
    current_user: User = Depends(get_current_active_user),
) -> list[dict[str, Any]]:
    """Get trial history for visualization."""
    return await get_optimization_history(study_id)
