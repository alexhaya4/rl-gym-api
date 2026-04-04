import time

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rate_limit import limiter
from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.inference import InferenceRequest, InferenceResponse, ModelCacheInfo
from app.services.audit_log import log_event
from app.services.inference import model_cache
from app.services.registry import get_production_model

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/{environment_id}/predict", response_model=InferenceResponse)
@limiter.limit("60/minute")
async def predict(
    environment_id: str,
    body: InferenceRequest,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> InferenceResponse:
    """Run inference on the production model for an environment."""
    algorithm = body.algorithm

    # Find production model in registry
    if algorithm is None:
        # Look for any production model for this environment
        from sqlalchemy import select

        from app.models.registry import ModelRegistry

        result = await db.execute(
            select(ModelRegistry).where(
                ModelRegistry.environment_id == environment_id,
                ModelRegistry.stage == "production",
                ModelRegistry.is_current.is_(True),
            )
        )
        registry_entry = result.scalar_one_or_none()
        if registry_entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"No production model found for {environment_id}",
            )
        algorithm = registry_entry.algorithm
    else:
        registry_entry = await get_production_model(db, environment_id, algorithm)
        if registry_entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"No production model found for {environment_id}",
            )

    # Resolve model path from the associated ModelVersion
    from sqlalchemy import select

    from app.models.model_version import ModelVersion

    mv_result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == registry_entry.model_version_id)
    )
    model_version = mv_result.scalar_one_or_none()
    if model_version is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model version {registry_entry.model_version_id} not found",
        )

    model_path = model_version.storage_path

    start = time.perf_counter()

    model = await model_cache.get_or_load_model(
        model_path=model_path,
        algorithm=algorithm,
        environment_id=environment_id,
    )

    action, probability = await model_cache.predict(
        model, body.observation, body.deterministic
    )

    latency_ms = (time.perf_counter() - start) * 1000

    # Audit log
    await log_event(
        db,
        event_type="inference",
        request=request,
        user_id=current_user.id,
        username=current_user.username,
        resource_type="model",
        resource_id=str(registry_entry.model_version_id),
        action="predict",
        details={
            "environment_id": environment_id,
            "algorithm": algorithm,
            "deterministic": body.deterministic,
            "latency_ms": round(latency_ms, 2),
        },
    )

    return InferenceResponse(
        action=action,
        action_probability=probability,
        latency_ms=round(latency_ms, 2),
        model_version_id=registry_entry.model_version_id,
        algorithm=algorithm,
        environment_id=environment_id,
    )


@router.get("/{environment_id}/info")
async def inference_info(
    environment_id: str,
    algorithm: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> dict:  # type: ignore[type-arg]
    """Get production model info for an environment. No auth required."""
    if algorithm is not None:
        entry = await get_production_model(db, environment_id, algorithm)
    else:
        from sqlalchemy import select

        from app.models.registry import ModelRegistry

        result = await db.execute(
            select(ModelRegistry).where(
                ModelRegistry.environment_id == environment_id,
                ModelRegistry.stage == "production",
                ModelRegistry.is_current.is_(True),
            )
        )
        entry = result.scalar_one_or_none()

    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No production model found for {environment_id}",
        )

    return {
        "id": entry.id,
        "name": entry.name,
        "environment_id": entry.environment_id,
        "algorithm": entry.algorithm,
        "stage": entry.stage,
        "model_version_id": entry.model_version_id,
        "mean_reward": entry.mean_reward,
    }


@router.get("/cache", response_model=list[ModelCacheInfo])
async def list_cache(
    current_user: User = Depends(get_current_active_user),
) -> list[ModelCacheInfo]:
    """List all currently cached models."""
    return model_cache.list_cached()


@router.delete("/cache")
async def clear_cache(
    current_user: User = Depends(get_current_active_user),
) -> dict[str, object]:
    """Clear the model cache."""
    count = model_cache.clear()
    return {"message": "Cache cleared", "models_evicted": count}
