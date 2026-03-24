from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.episode import Episode
from app.models.user import User
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentListResponse,
    ExperimentResponse,
    ExperimentUpdate,
)
from app.services.audit_log import log_event
from app.services.experiment import (
    create_experiment,
    delete_experiment,
    get_experiment,
    get_experiment_episodes,
    list_experiments,
    update_experiment,
)

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post(
    "", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED
)
async def create_experiment_endpoint(
    experiment_in: ExperimentCreate,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """Create a new reinforcement learning experiment.

    Registers an experiment with the specified environment, algorithm,
    and hyperparameters. The experiment starts in 'pending' status.
    """
    experiment = await create_experiment(db, experiment_in, current_user.id)
    await log_event(
        db, "experiment_create", request=request,
        user_id=current_user.id, username=current_user.username,
        resource_type="experiment", resource_id=str(experiment.id), action="create",
    )
    return ExperimentResponse.model_validate(experiment)


@router.get("", response_model=ExperimentListResponse)
async def list_experiments_endpoint(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: str | None = Query(
        None, alias="status", description="Filter by experiment status"
    ),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ExperimentListResponse:
    """List all experiments for the authenticated user.

    Supports pagination and optional filtering by experiment status
    (e.g. 'pending', 'running', 'completed', 'failed').
    """
    items, total = await list_experiments(
        db, current_user.id, page, page_size, status_filter
    )
    return ExperimentListResponse(
        items=[ExperimentResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment_endpoint(
    experiment_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """Retrieve a single experiment by ID.

    Returns the full experiment details including status, timestamps,
    and reward statistics. Returns 404 if the experiment does not exist
    or does not belong to the authenticated user.
    """
    experiment = await get_experiment(db, experiment_id, current_user.id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
        )
    response = ExperimentResponse.model_validate(experiment)
    if experiment.status == "completed" and experiment.mean_reward is not None:
        response.metrics_summary = {
            "min_episode_reward": experiment.mean_reward - (experiment.std_reward or 0),
            "max_episode_reward": experiment.mean_reward + (experiment.std_reward or 0),
            "mean_episode_reward": experiment.mean_reward,
            "total_timesteps": experiment.total_timesteps,
        }
    return response


@router.patch("/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment_endpoint(
    experiment_id: int,
    update: ExperimentUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """Update an existing experiment.

    Supports partial updates — only the provided fields will be modified.
    Returns 404 if the experiment does not exist or does not belong to
    the authenticated user.
    """
    experiment = await update_experiment(
        db, experiment_id, current_user.id, update
    )
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
        )
    return ExperimentResponse.model_validate(experiment)


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment_endpoint(
    experiment_id: int,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete an experiment by ID.

    Permanently removes the experiment and returns 204 on success.
    Returns 404 if the experiment does not exist or does not belong to
    the authenticated user.
    """
    deleted = await delete_experiment(db, experiment_id, current_user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
        )
    await log_event(
        db, "experiment_delete", request=request,
        user_id=current_user.id, username=current_user.username,
        resource_type="experiment", resource_id=str(experiment_id), action="delete",
    )


@router.get("/{experiment_id}/episodes", response_model=None)
async def get_experiment_episodes_endpoint(
    experiment_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[Episode]:
    """List all episodes for a given experiment.

    Returns the episode history including rewards and episode lengths.
    Returns 404 if the experiment does not exist or does not belong to
    the authenticated user.
    """
    experiment = await get_experiment(db, experiment_id, current_user.id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
        )
    return await get_experiment_episodes(db, experiment_id)
