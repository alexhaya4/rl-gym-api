from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.custom_environment import (
    CustomEnvironmentCreate,
    CustomEnvironmentListResponse,
    CustomEnvironmentResponse,
)
from app.services.custom_environment import (
    create_custom_environment,
    delete_custom_environment,
    get_custom_environment,
    list_custom_environments,
    validate_environment_code,
)

router = APIRouter(prefix="/custom-environments", tags=["custom-environments"])


@router.post(
    "", response_model=CustomEnvironmentResponse, status_code=status.HTTP_201_CREATED
)
async def register_custom_environment(
    env_in: CustomEnvironmentCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> CustomEnvironmentResponse:
    """Register a custom Gymnasium-compatible environment by uploading Python source code."""
    custom_env = await create_custom_environment(db, env_in, current_user.id)
    return CustomEnvironmentResponse.model_validate(custom_env)


@router.get("", response_model=CustomEnvironmentListResponse)
async def list_custom_environments_endpoint(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> CustomEnvironmentListResponse:
    """List all custom environments for the authenticated user."""
    items = await list_custom_environments(db, current_user.id)
    return CustomEnvironmentListResponse(
        items=[CustomEnvironmentResponse.model_validate(i) for i in items],
        total=len(items),
    )


@router.get("/{env_id}", response_model=CustomEnvironmentResponse)
async def get_custom_environment_endpoint(
    env_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> CustomEnvironmentResponse:
    """Retrieve a single custom environment by ID."""
    custom_env = await get_custom_environment(db, env_id, current_user.id)
    if custom_env is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom environment not found",
        )
    return CustomEnvironmentResponse.model_validate(custom_env)


@router.delete("/{env_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_custom_environment_endpoint(
    env_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a custom environment by ID."""
    deleted = await delete_custom_environment(db, env_id, current_user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom environment not found",
        )


@router.post("/{env_id}/validate", response_model=CustomEnvironmentResponse)
async def revalidate_custom_environment(
    env_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> CustomEnvironmentResponse:
    """Re-run validation on an existing custom environment."""
    custom_env = await get_custom_environment(db, env_id, current_user.id)
    if custom_env is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom environment not found",
        )

    is_valid, error_message = validate_environment_code(
        custom_env.source_code, custom_env.name
    )
    custom_env.is_validated = is_valid
    custom_env.validation_error = error_message

    await db.commit()
    await db.refresh(custom_env)
    return CustomEnvironmentResponse.model_validate(custom_env)
