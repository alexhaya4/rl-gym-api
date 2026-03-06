from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.training import TrainingConfig, TrainingResult, TrainingStatus
from app.services.training import (
    get_training_status,
    list_training_sessions,
    start_training,
)

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/", response_model=TrainingStatus, status_code=status.HTTP_202_ACCEPTED)
async def create_training(
    config: TrainingConfig,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> TrainingStatus:
    if config.algorithm not in ("PPO", "A2C", "DQN"):
        raise HTTPException(status_code=400, detail="Unsupported algorithm. Use PPO, A2C, or DQN.")
    try:
        result = await start_training(db, config, current_user.id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return TrainingStatus(**result)


@router.get("/", response_model=list[TrainingStatus])
async def list_trainings(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[TrainingStatus]:
    sessions = await list_training_sessions(db, current_user.id)
    return [TrainingStatus(**s) for s in sessions]


@router.get("/{experiment_id}", response_model=TrainingStatus)
async def get_training(
    experiment_id: int,
    _current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> TrainingStatus:
    session = await get_training_status(db, experiment_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return TrainingStatus(**session)


@router.get("/{experiment_id}/result", response_model=TrainingResult)
async def get_training_result(
    experiment_id: int,
    _current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> TrainingResult:
    session = await get_training_status(db, experiment_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if session["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not yet completed")
    return TrainingResult(**session)
