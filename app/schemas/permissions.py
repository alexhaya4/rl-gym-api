from pydantic import BaseModel, field_validator


class RoleAssignment(BaseModel):
    user_id: int
    role: str
    organization_id: int | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        valid_roles = {"owner", "admin", "member", "viewer"}
        if v not in valid_roles:
            raise ValueError(f"Role must be one of: {', '.join(sorted(valid_roles))}")
        return v


class PermissionCheckRequest(BaseModel):
    permission: str
    resource_type: str | None = None
    resource_id: int | None = None


class PermissionCheckResponse(BaseModel):
    allowed: bool
    role: str
    permission: str
    reason: str | None = None


class UserPermissionsResponse(BaseModel):
    user_id: int
    role: str
    permissions: list[str]
    organization_id: int | None = None
