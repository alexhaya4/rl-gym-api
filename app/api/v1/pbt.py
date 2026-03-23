from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.pbt import PBTMember
from app.models.user import User
from app.schemas.pbt import (
    PBTExperimentResponse,
    PBTListResponse,
    PBTMemberResponse,
    PBTRequest,
)
from app.services.pbt import (
    create_pbt_experiment,
    get_pbt_experiment,
    list_pbt_experiments,
    run_pbt,
)

router = APIRouter(prefix="/pbt", tags=["population-based-training"])


async def _run_pbt_bg(pbt_experiment_id: int, request: PBTRequest) -> None:
    from app.db.session import AsyncSessionLocal

    db = AsyncSessionLocal()
    try:
        await run_pbt(db, pbt_experiment_id, request)
    finally:
        await db.close()


@router.post("/", response_model=PBTExperimentResponse, status_code=202)
async def create_and_start(
    request: PBTRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> PBTExperimentResponse:
    """Create and start a Population Based Training experiment."""
    experiment = await create_pbt_experiment(db, request, current_user.id)
    background_tasks.add_task(_run_pbt_bg, experiment.id, request)
    return PBTExperimentResponse.model_validate(experiment)


@router.get("/", response_model=PBTListResponse)
async def list_all(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> PBTListResponse:
    """List all PBT experiments for the current user."""
    experiments = await list_pbt_experiments(db, current_user.id)
    return PBTListResponse(
        items=[PBTExperimentResponse.model_validate(e) for e in experiments],
        total=len(experiments),
    )


@router.get("/{pbt_id}", response_model=PBTExperimentResponse)
async def get_one(
    pbt_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> PBTExperimentResponse:
    """Get PBT experiment with all member details."""
    experiment = await get_pbt_experiment(db, pbt_id, current_user.id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="PBT experiment not found")

    result = await db.execute(
        select(PBTMember)
        .where(PBTMember.pbt_experiment_id == pbt_id)
        .order_by(PBTMember.member_index)
    )
    members = list(result.scalars().all())

    response = PBTExperimentResponse.model_validate(experiment)
    response.members = [PBTMemberResponse.model_validate(m) for m in members]
    return response


@router.get("/{pbt_id}/members", response_model=list[PBTMemberResponse])
async def get_members(
    pbt_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[PBTMemberResponse]:
    """Get all population members for a PBT experiment."""
    experiment = await get_pbt_experiment(db, pbt_id, current_user.id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="PBT experiment not found")

    result = await db.execute(
        select(PBTMember)
        .where(PBTMember.pbt_experiment_id == pbt_id)
        .order_by(PBTMember.member_index)
    )
    members = list(result.scalars().all())
    return [PBTMemberResponse.model_validate(m) for m in members]


@router.get("/{pbt_id}/best", response_model=PBTMemberResponse)
async def get_best(
    pbt_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> PBTMemberResponse:
    """Get the best performing population member."""
    experiment = await get_pbt_experiment(db, pbt_id, current_user.id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="PBT experiment not found")

    result = await db.execute(
        select(PBTMember).where(
            PBTMember.pbt_experiment_id == pbt_id,
            PBTMember.is_best.is_(True),
        )
    )
    best = result.scalar_one_or_none()
    if best is None:
        raise HTTPException(
            status_code=404, detail="No best member found (experiment may still be running)"
        )
    return PBTMemberResponse.model_validate(best)
