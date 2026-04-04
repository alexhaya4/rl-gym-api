from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.ml import (
    MLModelInfo,
    MLPredictRequest,
    MLPredictResponse,
    MLTrainRequest,
    MLTrainResponse,
)
from app.services.ml import (
    SUPPORTED_ALGORITHMS,
    delete_model,
    get_model,
    list_models,
    predict,
    train,
)

router = APIRouter(prefix="/ml", tags=["ml"])


@router.post("/train", response_model=MLTrainResponse, status_code=201)
async def train_model(
    body: MLTrainRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> MLTrainResponse:
    """Train a scikit-learn model on a dataset."""
    try:
        return await train(body, current_user.id, db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


@router.post("/predict", response_model=MLPredictResponse)
async def predict_model(
    body: MLPredictRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> MLPredictResponse:
    """Run predictions using a trained ML model."""
    try:
        return await predict(body, current_user.id, db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


@router.get("/models")
async def list_ml_models(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:  # type: ignore[type-arg]
    """List trained ML models for the current user."""
    models, total = await list_models(db, current_user.id, page, page_size)
    return {
        "items": [MLModelInfo.model_validate(m) for m in models],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/models/{model_id}", response_model=MLModelInfo)
async def get_ml_model(
    model_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> MLModelInfo:
    """Get a specific ML model."""
    ml_model = await get_model(db, model_id, current_user.id)
    if ml_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return MLModelInfo.model_validate(ml_model)


@router.delete("/models/{model_id}", status_code=204)
async def delete_ml_model(
    model_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a trained ML model."""
    if not await delete_model(db, model_id, current_user.id):
        raise HTTPException(status_code=404, detail="Model not found")


@router.get("/algorithms")
async def list_algorithms() -> dict[str, list[str]]:
    """List all supported ML algorithms by task type."""
    return SUPPORTED_ALGORITHMS
