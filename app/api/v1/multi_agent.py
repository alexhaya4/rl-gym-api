from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.multi_agent import AgentPolicy
from app.models.user import User
from app.schemas.multi_agent import (
    AgentPolicyResponse,
    MultiAgentExperimentListResponse,
    MultiAgentExperimentResponse,
    MultiAgentTrainingRequest,
)
from app.services.multi_agent import (
    create_multi_agent_experiment,
    get_available_multi_agent_environments,
    get_multi_agent_experiment,
    list_multi_agent_experiments,
    run_multi_agent_training,
)

router = APIRouter(prefix="/multi-agent", tags=["multi-agent"])


async def _run_training_bg(experiment_id: int, request: MultiAgentTrainingRequest) -> None:
    from app.db.session import AsyncSessionLocal

    db = AsyncSessionLocal()
    try:
        await run_multi_agent_training(db, experiment_id, request)
    finally:
        await db.close()


@router.get("/environments", response_model=list[dict[str, Any]])
async def list_environments() -> list[dict[str, Any]]:
    """List supported PettingZoo multi-agent environments."""
    return await get_available_multi_agent_environments()


@router.post("/train", response_model=MultiAgentExperimentResponse, status_code=202)
async def start_training(
    request: MultiAgentTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> MultiAgentExperimentResponse:
    """Start multi-agent training."""
    try:
        experiment = await create_multi_agent_experiment(db, request, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    background_tasks.add_task(_run_training_bg, experiment.id, request)

    return MultiAgentExperimentResponse.model_validate(experiment)


@router.get("/experiments", response_model=MultiAgentExperimentListResponse)
async def list_experiments(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> MultiAgentExperimentListResponse:
    """List all multi-agent experiments for the current user."""
    experiments = await list_multi_agent_experiments(db, current_user.id)
    return MultiAgentExperimentListResponse(
        items=[MultiAgentExperimentResponse.model_validate(e) for e in experiments],
        total=len(experiments),
    )


@router.get(
    "/experiments/{experiment_id}",
    response_model=MultiAgentExperimentResponse,
)
async def get_experiment(
    experiment_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> MultiAgentExperimentResponse:
    """Get multi-agent experiment details with per-agent results."""
    experiment = await get_multi_agent_experiment(db, experiment_id, current_user.id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    # Fetch agent policies
    result = await db.execute(
        select(AgentPolicy).where(AgentPolicy.experiment_id == experiment_id)
    )
    policies = list(result.scalars().all())

    response = MultiAgentExperimentResponse.model_validate(experiment)
    response.agent_policies = [
        AgentPolicyResponse.model_validate(p) for p in policies
    ]
    return response


@router.get(
    "/experiments/{experiment_id}/agents",
    response_model=list[AgentPolicyResponse],
)
async def get_agent_policies(
    experiment_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[AgentPolicyResponse]:
    """Get all agent policies for a multi-agent experiment."""
    experiment = await get_multi_agent_experiment(db, experiment_id, current_user.id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    result = await db.execute(
        select(AgentPolicy).where(AgentPolicy.experiment_id == experiment_id)
    )
    policies = list(result.scalars().all())
    return [AgentPolicyResponse.model_validate(p) for p in policies]
