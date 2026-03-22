from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.registry import ModelRegistry
from app.models.user import User
from app.schemas.registry import (
    ComparisonResult,
    PromoteRequest,
    RegistryEntry,
    RegistryListResponse,
    RollbackRequest,
)
from app.services.registry import (
    compare_models,
    get_production_model,
    list_registry,
    promote_model,
    register_model,
    rollback_production,
)

router = APIRouter(prefix="/registry", tags=["model-registry"])


class _RegisterRequest:
    """Dependency for register endpoint body parsing."""

    def __init__(
        self,
        model_version_id: int,
        environment_id: str,
        algorithm: str,
    ) -> None:
        self.model_version_id = model_version_id
        self.environment_id = environment_id
        self.algorithm = algorithm


@router.post("/register", response_model=RegistryEntry, status_code=201)
async def register(
    model_version_id: int,
    environment_id: str,
    algorithm: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> RegistryEntry:
    """Register a model version in the registry."""
    try:
        entry = await register_model(
            db, model_version_id, environment_id, algorithm, current_user.id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    return RegistryEntry.model_validate(entry)


@router.get("/", response_model=RegistryListResponse)
async def list_entries(
    stage: str | None = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> RegistryListResponse:
    """List all registry entries for the current user."""
    entries = await list_registry(db, current_user.id, stage=stage)
    return RegistryListResponse(
        items=[RegistryEntry.model_validate(e) for e in entries],
        total=len(entries),
    )


@router.get(
    "/production/{environment_id}/{algorithm}",
    response_model=RegistryEntry,
)
async def get_production(
    environment_id: str,
    algorithm: str,
    db: AsyncSession = Depends(get_db),
) -> RegistryEntry:
    """Get the current production model for an environment/algorithm."""
    entry = await get_production_model(db, environment_id, algorithm)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No production model for {environment_id}/{algorithm}",
        )
    return RegistryEntry.model_validate(entry)


@router.post("/{registry_id}/promote", response_model=RegistryEntry)
async def promote(
    registry_id: int,
    request: PromoteRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> RegistryEntry:
    """Promote a model to the next stage."""
    try:
        entry = await promote_model(db, registry_id, request, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    return RegistryEntry.model_validate(entry)


@router.post(
    "/rollback/{environment_id}/{algorithm}",
    response_model=RegistryEntry,
)
async def rollback(
    environment_id: str,
    algorithm: str,
    request: RollbackRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> RegistryEntry:
    """Rollback to the previous production model."""
    entry = await rollback_production(
        db, environment_id, algorithm, current_user.id
    )
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="No previous production model to rollback to",
        )
    return RegistryEntry.model_validate(entry)


@router.get("/{registry_id}/compare", response_model=ComparisonResult)
async def compare(
    registry_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ComparisonResult:
    """Compare a candidate model against current production."""
    result = await db.execute(
        select(ModelRegistry).where(ModelRegistry.id == registry_id)
    )
    entry = result.scalar_one_or_none()
    if entry is None:
        raise HTTPException(status_code=404, detail="Registry entry not found")

    try:
        return await compare_models(
            db, entry.model_version_id, entry.environment_id, entry.algorithm
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
