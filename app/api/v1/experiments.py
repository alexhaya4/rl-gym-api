from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentListResponse,
    ExperimentResponse,
    ExperimentUpdate,
)
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
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new reinforcement learning experiment.

    Registers an experiment with the specified environment, algorithm,
    and hyperparameters. The experiment starts in 'pending' status.
    """
    return await create_experiment(db, experiment_in, current_user.id)


@router.get("", response_model=ExperimentListResponse)
async def list_experiments_endpoint(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: str | None = Query(
        None, alias="status", description="Filter by experiment status"
    ),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """List all experiments for the authenticated user.

    Supports pagination and optional filtering by experiment status
    (e.g. 'pending', 'running', 'completed', 'failed').
    """
    items, total = await list_experiments(
        db, current_user.id, page, page_size, status_filter
    )
    return ExperimentListResponse(
        items=items, total=total, page=page, page_size=page_size
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment_endpoint(
    experiment_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
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
    return experiment


@router.patch("/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment_endpoint(
    experiment_id: int,
    update: ExperimentUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
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
    return experiment


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment_endpoint(
    experiment_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
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


@router.get("/{experiment_id}/episodes")
async def get_experiment_episodes_endpoint(
    experiment_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
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
