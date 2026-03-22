from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.artifact import ArtifactCreate, ArtifactResponse
from app.services.artifact import (
    create_artifact,
    delete_artifact,
    get_artifact,
    list_artifacts,
    set_lineage,
)

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


class LineageRequest(BaseModel):
    parent_experiment_id: int
    child_experiment_id: int
    relationship_type: str
    description: str | None = None


@router.post("/", response_model=ArtifactResponse, status_code=status.HTTP_201_CREATED)
async def create(
    artifact_create: ArtifactCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ArtifactResponse:
    """Create an artifact record."""
    artifact = await create_artifact(db, artifact_create, current_user.id)
    return ArtifactResponse.model_validate(artifact)


@router.get("/", response_model=list[ArtifactResponse])
async def list_all(
    experiment_id: int | None = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[ArtifactResponse]:
    """List artifacts for the current user."""
    artifacts = await list_artifacts(db, current_user.id, experiment_id=experiment_id)
    return [ArtifactResponse.model_validate(a) for a in artifacts]


@router.get("/{artifact_id}", response_model=ArtifactResponse)
async def get_one(
    artifact_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ArtifactResponse:
    """Get artifact details."""
    artifact = await get_artifact(db, artifact_id, current_user.id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return ArtifactResponse.model_validate(artifact)


@router.delete("/{artifact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(
    artifact_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete an artifact."""
    if not await delete_artifact(db, artifact_id, current_user.id):
        raise HTTPException(status_code=404, detail="Artifact not found")


@router.post("/{artifact_id}/lineage", response_model=dict[str, Any])
async def create_lineage(
    artifact_id: int,
    request: LineageRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Set parent-child lineage between experiments."""
    lineage = await set_lineage(
        db,
        parent_id=request.parent_experiment_id,
        child_id=request.child_experiment_id,
        relationship_type=request.relationship_type,
        description=request.description,
    )
    return {
        "id": lineage.id,
        "parent_experiment_id": lineage.parent_experiment_id,
        "child_experiment_id": lineage.child_experiment_id,
        "relationship_type": lineage.relationship_type,
        "description": lineage.description,
    }
