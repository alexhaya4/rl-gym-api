from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.episode import Episode
from app.models.user import User
from app.schemas.evaluation import EvaluationRequest, EvaluationResponse
from app.services.evaluation import evaluate_experiment
from app.services.experiment import get_experiment, get_experiment_episodes

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationResponse)
async def run_evaluation_endpoint(
    request: EvaluationRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> EvaluationResponse:
    """Evaluate a completed experiment across multiple episodes."""
    try:
        return await evaluate_experiment(db, request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@router.get("/experiments/{experiment_id}/episodes", response_model=None)
async def list_evaluation_episodes(
    experiment_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[Episode]:
    """List all episode records for an experiment with reward statistics."""
    experiment = await get_experiment(db, experiment_id, current_user.id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found"
        )
    return await get_experiment_episodes(db, experiment_id)
