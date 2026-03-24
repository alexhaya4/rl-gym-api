from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.permissions import ROLE_PERMISSIONS, Permission, has_permission
from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.user import User
from app.schemas.permissions import (
    PermissionCheckRequest,
    PermissionCheckResponse,
    RoleAssignment,
    UserPermissionsResponse,
)
from app.services.rbac import assign_role, get_user_role, list_user_permissions

router = APIRouter(prefix="/rbac", tags=["rbac"])


@router.get("/my-permissions", response_model=UserPermissionsResponse)
async def my_permissions(
    organization_id: int | None = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> UserPermissionsResponse:
    """Get current user's role and permissions."""
    role = await get_user_role(db, current_user.id, organization_id)
    permissions = await list_user_permissions(db, current_user.id, organization_id)
    return UserPermissionsResponse(
        user_id=current_user.id,
        role=role,
        permissions=permissions,
        organization_id=organization_id,
    )


@router.post("/check", response_model=PermissionCheckResponse)
async def check_permission(
    body: PermissionCheckRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> PermissionCheckResponse:
    """Check if the current user has a specific permission."""
    role = await get_user_role(db, current_user.id)

    try:
        perm = Permission(body.permission)
    except ValueError:
        return PermissionCheckResponse(
            allowed=False,
            role=role,
            permission=body.permission,
            reason=f"Unknown permission: {body.permission}",
        )

    allowed = has_permission(role, perm)
    reason = None if allowed else f"Role '{role}' does not have '{perm}' permission"
    return PermissionCheckResponse(
        allowed=allowed,
        role=role,
        permission=body.permission,
        reason=reason,
    )


@router.post("/assign", status_code=status.HTTP_200_OK)
async def assign_user_role(
    body: RoleAssignment,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str | int | None]:
    """Assign a role to a user. Requires ORG_MANAGE permission."""
    caller_role = await get_user_role(db, current_user.id, body.organization_id)
    if not has_permission(caller_role, Permission.ORG_MANAGE):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: org:manage required",
        )

    try:
        user_role = await assign_role(db, body.user_id, body.role, body.organization_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from None

    return {
        "user_id": user_role.user_id,
        "role": user_role.role,
        "organization_id": user_role.organization_id,
    }


@router.get("/roles")
async def list_roles() -> dict[str, list[str]]:
    """List all available roles and their permissions."""
    return {
        role: sorted(str(p) for p in perms)
        for role, perms in ROLE_PERMISSIONS.items()
    }
