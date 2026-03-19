from datetime import datetime

from pydantic import BaseModel, ConfigDict


class OrganizationCreate(BaseModel):
    name: str
    slug: str | None = None


class OrganizationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    plan: str
    is_active: bool
    created_at: datetime


class OrganizationMemberResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    organization_id: int
    user_id: int
    role: str
    created_at: datetime


class OrganizationListResponse(BaseModel):
    items: list[OrganizationResponse]
    total: int
