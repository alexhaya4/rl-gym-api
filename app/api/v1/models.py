from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.storage import get_storage
from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.model_version import ModelVersion
from app.models.user import User
from app.schemas.model_version import ModelVersionListResponse, ModelVersionResponse
from app.services.model_storage import get_download_url, list_model_versions

router = APIRouter(prefix="/models", tags=["models"])


@router.get(
    "/experiments/{experiment_id}", response_model=ModelVersionListResponse
)
async def list_experiment_models(
    experiment_id: int,
    _current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ModelVersionListResponse:
    """List all model versions for an experiment."""
    items = await list_model_versions(db, experiment_id)
    return ModelVersionListResponse(
        items=[
            ModelVersionResponse.model_validate(mv) for mv in items
        ],
        total=len(items),
    )


@router.get("/{version_id}", response_model=ModelVersionResponse)
async def get_model_version(
    version_id: int,
    _current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ModelVersionResponse:
    """Get a single model version with download URL."""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == version_id)
    )
    mv = result.scalar_one_or_none()
    if mv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model version not found",
        )
    response = ModelVersionResponse.model_validate(mv)
    response.download_url = get_download_url(mv)
    return response


@router.get("/{version_id}/download", response_model=None)
async def download_model(
    version_id: int,
    _current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> FileResponse | RedirectResponse:
    """Download a model file."""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == version_id)
    )
    mv = result.scalar_one_or_none()
    if mv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model version not found",
        )

    if mv.storage_backend == "s3":
        url = get_download_url(mv)
        return RedirectResponse(url=url)

    settings = get_settings()
    storage = get_storage()
    full_path = Path(settings.STORAGE_LOCAL_PATH) / mv.storage_path
    if not await storage.exists(mv.storage_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model file not found in storage",
        )
    return FileResponse(
        path=str(full_path),
        filename=f"{mv.algorithm}_v{mv.version}.zip",
        media_type="application/zip",
    )


@router.delete("/{version_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    version_id: int,
    _current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a model version from storage and database."""
    from app.services.model_storage import delete_model_version

    deleted = await delete_model_version(db, version_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model version not found",
        )
