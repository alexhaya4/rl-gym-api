import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.permissions import ROLE_PERMISSIONS, Permission, has_permission
from app.models.experiment import Experiment
from app.models.role import UserRole

logger = logging.getLogger(__name__)

VALID_ROLES = {"owner", "admin", "member", "viewer"}

_RESOURCE_OWNER_MAP: dict[str, Any] = {
    "experiment": Experiment,
}


async def get_user_role(
    db: AsyncSession, user_id: int, organization_id: int | None = None
) -> str:
    """Fetch a user's role in an organization, or their global role."""
    query = select(UserRole.role).where(UserRole.user_id == user_id)
    if organization_id is not None:
        query = query.where(UserRole.organization_id == organization_id)
    else:
        query = query.where(UserRole.organization_id.is_(None))

    result = await db.execute(query.limit(1))
    role = result.scalar_one_or_none()
    return role if role is not None else "member"


async def assign_role(
    db: AsyncSession, user_id: int, role: str, organization_id: int | None = None
) -> UserRole:
    """Create or update a UserRole record."""
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role: {role}. Must be one of {VALID_ROLES}")

    query = select(UserRole).where(UserRole.user_id == user_id)
    if organization_id is not None:
        query = query.where(UserRole.organization_id == organization_id)
    else:
        query = query.where(UserRole.organization_id.is_(None))

    result = await db.execute(query)
    user_role = result.scalar_one_or_none()

    if user_role is not None:
        user_role.role = role
    else:
        user_role = UserRole(
            user_id=user_id,
            organization_id=organization_id,
            role=role,
        )
        db.add(user_role)

    await db.commit()
    await db.refresh(user_role)
    return user_role


async def check_resource_access(
    db: AsyncSession,
    user_id: int,
    resource_type: str,
    resource_id: int,
    required_permission: Permission,
) -> bool:
    """Check if a user has permission to access a resource."""
    role = await get_user_role(db, user_id)

    if has_permission(role, required_permission):
        return True

    # Check direct ownership for supported resource types
    model_class = _RESOURCE_OWNER_MAP.get(resource_type)
    if model_class is not None:
        result = await db.execute(
            select(model_class).where(
                model_class.id == resource_id,
                model_class.user_id == user_id,
            )
        )
        if result.scalar_one_or_none() is not None:
            return True

    return False


async def list_user_permissions(
    db: AsyncSession, user_id: int, organization_id: int | None = None
) -> list[str]:
    """Return all permission names for a user's role."""
    role = await get_user_role(db, user_id, organization_id)
    permissions = ROLE_PERMISSIONS.get(role, set())
    return sorted(str(p) for p in permissions)
