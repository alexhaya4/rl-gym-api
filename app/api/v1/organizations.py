import re
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.organization import Organization, OrganizationMember
from app.models.subscription import Subscription
from app.models.user import User
from app.schemas.billing import UsageResponse
from app.schemas.organization import (
    OrganizationCreate,
    OrganizationListResponse,
    OrganizationMemberResponse,
    OrganizationResponse,
)
from app.services.audit_log import log_event
from app.services.quota import get_or_create_usage, get_plan_limits

router = APIRouter(prefix="/organizations", tags=["organizations"])


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"[-\s]+", "-", slug)


class AddMemberRequest(BaseModel):
    user_id: int
    role: str = "member"


@router.post("", response_model=OrganizationResponse, status_code=status.HTTP_201_CREATED)
async def create_organization(
    body: OrganizationCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Organization:
    slug = body.slug or _slugify(body.name)

    existing = await db.execute(
        select(Organization).where(
            (Organization.name == body.name) | (Organization.slug == slug)
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Organization name or slug already exists",
        )

    org = Organization(name=body.name, slug=slug)
    db.add(org)
    await db.flush()

    member = OrganizationMember(
        organization_id=org.id, user_id=current_user.id, role="owner"
    )
    db.add(member)

    subscription = Subscription(
        organization_id=org.id, plan="free", status="active"
    )
    db.add(subscription)

    await db.commit()
    await db.refresh(org)
    await log_event(
        db, "org_create", request=request,
        user_id=current_user.id, username=current_user.username,
        resource_type="organization", resource_id=str(org.id), action="create",
    )
    return org


@router.get("", response_model=OrganizationListResponse)
async def list_organizations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> dict[str, Any]:
    result = await db.execute(
        select(Organization)
        .join(OrganizationMember, Organization.id == OrganizationMember.organization_id)
        .where(OrganizationMember.user_id == current_user.id)
    )
    orgs = list(result.scalars().all())

    return {"items": orgs, "total": len(orgs)}


@router.get("/{org_id}", response_model=OrganizationResponse)
async def get_organization(
    org_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> Organization:
    result = await db.execute(
        select(Organization).where(Organization.id == org_id)
    )
    org = result.scalar_one_or_none()
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")

    member_result = await db.execute(
        select(OrganizationMember).where(
            OrganizationMember.organization_id == org_id,
            OrganizationMember.user_id == current_user.id,
        )
    )
    if member_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=403, detail="Not a member of this organization")

    return org


@router.post(
    "/{org_id}/members",
    response_model=OrganizationMemberResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_member(
    org_id: int,
    body: AddMemberRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> OrganizationMember:
    result = await db.execute(
        select(OrganizationMember).where(
            OrganizationMember.organization_id == org_id,
            OrganizationMember.user_id == current_user.id,
        )
    )
    caller = result.scalar_one_or_none()
    if caller is None or caller.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owners and admins can add members")

    existing = await db.execute(
        select(OrganizationMember).where(
            OrganizationMember.organization_id == org_id,
            OrganizationMember.user_id == body.user_id,
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail="User is already a member")

    member = OrganizationMember(
        organization_id=org_id, user_id=body.user_id, role=body.role
    )
    db.add(member)
    await db.commit()
    await db.refresh(member)
    return member


@router.delete("/{org_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    org_id: int,
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> None:
    result = await db.execute(
        select(OrganizationMember).where(
            OrganizationMember.organization_id == org_id,
            OrganizationMember.user_id == current_user.id,
        )
    )
    caller = result.scalar_one_or_none()
    if caller is None or caller.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owners and admins can remove members")

    result = await db.execute(
        select(OrganizationMember).where(
            OrganizationMember.organization_id == org_id,
            OrganizationMember.user_id == user_id,
        )
    )
    member = result.scalar_one_or_none()
    if member is None:
        raise HTTPException(status_code=404, detail="Member not found")

    await db.delete(member)
    await db.commit()


@router.get("/{org_id}/usage", response_model=UsageResponse)
async def get_usage(
    org_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> dict[str, Any]:
    member_result = await db.execute(
        select(OrganizationMember).where(
            OrganizationMember.organization_id == org_id,
            OrganizationMember.user_id == current_user.id,
        )
    )
    if member_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=403, detail="Not a member of this organization")

    org_result = await db.execute(
        select(Organization).where(Organization.id == org_id)
    )
    org = org_result.scalar_one_or_none()
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")

    month = datetime.now(UTC).strftime("%Y-%m")
    usage = await get_or_create_usage(db, org_id, month)
    await db.commit()

    limits = get_plan_limits(org.plan)

    return {
        "month": usage.month,
        "experiments_count": usage.experiments_count,
        "environments_count": usage.environments_count,
        "total_timesteps": usage.total_timesteps,
        "training_jobs_count": usage.training_jobs_count,
        "api_calls_count": usage.api_calls_count,
        "limits": limits,
    }
