from enum import StrEnum
from typing import Any

from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.organization import OrganizationMember
from app.models.user import User


class Permission(StrEnum):
    EXPERIMENT_CREATE = "experiment:create"
    EXPERIMENT_READ = "experiment:read"
    EXPERIMENT_UPDATE = "experiment:update"
    EXPERIMENT_DELETE = "experiment:delete"

    MODEL_READ = "model:read"
    MODEL_PROMOTE = "model:promote"
    MODEL_DELETE = "model:delete"

    ENVIRONMENT_CREATE = "environment:create"
    ENVIRONMENT_READ = "environment:read"
    ENVIRONMENT_DELETE = "environment:delete"

    DATASET_CREATE = "dataset:create"
    DATASET_READ = "dataset:read"
    DATASET_DELETE = "dataset:delete"

    ORG_MANAGE = "org:manage"
    ORG_READ = "org:read"

    BILLING_READ = "billing:read"
    BILLING_MANAGE = "billing:manage"

    AUDIT_READ = "audit:read"

    CUSTOM_ENV_CREATE = "custom_env:create"
    CUSTOM_ENV_DELETE = "custom_env:delete"

    AB_TEST_CREATE = "ab_test:create"
    AB_TEST_READ = "ab_test:read"

    PIPELINE_CREATE = "pipeline:create"
    PIPELINE_READ = "pipeline:read"


_ALL_PERMISSIONS = set(Permission)

_READ_PERMISSIONS = {
    Permission.EXPERIMENT_READ,
    Permission.MODEL_READ,
    Permission.ENVIRONMENT_READ,
    Permission.DATASET_READ,
    Permission.ORG_READ,
    Permission.BILLING_READ,
    Permission.AUDIT_READ,
    Permission.AB_TEST_READ,
    Permission.PIPELINE_READ,
}

_MEMBER_PERMISSIONS = _READ_PERMISSIONS | {
    Permission.EXPERIMENT_CREATE,
    Permission.EXPERIMENT_UPDATE,
    Permission.ENVIRONMENT_CREATE,
    Permission.DATASET_CREATE,
    Permission.CUSTOM_ENV_CREATE,
    Permission.AB_TEST_CREATE,
    Permission.PIPELINE_CREATE,
}

_ADMIN_PERMISSIONS = _ALL_PERMISSIONS - {
    Permission.BILLING_MANAGE,
    Permission.ORG_MANAGE,
}

ROLE_PERMISSIONS: dict[str, set[Permission]] = {
    "owner": _ALL_PERMISSIONS,
    "admin": _ADMIN_PERMISSIONS,
    "member": _MEMBER_PERMISSIONS,
    "viewer": _READ_PERMISSIONS,
}


def has_permission(role: str, permission: Permission) -> bool:
    """Check if a role has the specified permission."""
    return permission in ROLE_PERMISSIONS.get(role, set())


def require_permission(permission: Permission) -> Any:
    """Return a FastAPI dependency that enforces the given permission."""

    async def _check_permission(
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db),
    ) -> User:
        result = await db.execute(
            select(OrganizationMember.role)
            .where(OrganizationMember.user_id == current_user.id)
            .limit(1)
        )
        role_row = result.scalar_one_or_none()
        role = role_row if role_row is not None else "member"

        if not has_permission(role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission} required",
            )
        return current_user

    return _check_permission
