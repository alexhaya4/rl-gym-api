from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.organization import Organization, OrganizationMember
from app.models.subscription import Subscription
from app.models.user import User
from app.schemas.billing import (
    CheckoutSessionResponse,
    PlanInfo,
    SubscriptionResponse,
)
from app.services.billing import (
    PLANS,
    create_checkout_session,
    get_subscription_status,
    handle_webhook_event,
)

router = APIRouter(prefix="/billing", tags=["billing"])


class CheckoutRequest(BaseModel):
    org_id: int
    plan: str
    success_url: str
    cancel_url: str


@router.get("/plans", response_model=list[PlanInfo])
async def list_plans() -> list[PlanInfo]:
    return list(PLANS.values())


@router.get("/subscription/{org_id}", response_model=SubscriptionResponse)
async def get_subscription(
    org_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> SubscriptionResponse:
    member_result = await db.execute(
        select(OrganizationMember).where(
            OrganizationMember.organization_id == org_id,
            OrganizationMember.user_id == current_user.id,
        )
    )
    if member_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=403, detail="Not a member of this organization")

    sub = await get_subscription_status(org_id, db)
    if sub is None:
        raise HTTPException(status_code=404, detail="No subscription found")
    return sub


@router.post("/checkout", response_model=CheckoutSessionResponse)
async def create_checkout(
    body: CheckoutRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> dict[str, str]:
    if body.plan not in ("pro", "enterprise"):
        raise HTTPException(status_code=400, detail="Invalid plan. Choose 'pro' or 'enterprise'")

    member_result = await db.execute(
        select(OrganizationMember).where(
            OrganizationMember.organization_id == body.org_id,
            OrganizationMember.user_id == current_user.id,
        )
    )
    caller = member_result.scalar_one_or_none()
    if caller is None or caller.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owners and admins can manage billing")

    org_result = await db.execute(
        select(Organization).where(Organization.id == body.org_id)
    )
    org = org_result.scalar_one_or_none()
    if org is None:
        raise HTTPException(status_code=404, detail="Organization not found")

    from app.config import get_settings

    settings = get_settings()
    price_id = (
        settings.STRIPE_PRO_PRICE_ID if body.plan == "pro"
        else settings.STRIPE_ENTERPRISE_PRICE_ID
    )
    if price_id is None and settings.STRIPE_SECRET_KEY is not None:
        raise HTTPException(status_code=400, detail="Price ID not configured for this plan")

    customer_id = org.stripe_customer_id or ""
    session = await create_checkout_session(
        customer_id=customer_id,
        price_id=price_id or "",
        success_url=body.success_url,
        cancel_url=body.cancel_url,
    )
    return session


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        result = await handle_webhook_event(payload, sig_header, db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return result


@router.post("/cancel/{org_id}", response_model=SubscriptionResponse)
async def cancel_subscription(
    org_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> SubscriptionResponse:
    member_result = await db.execute(
        select(OrganizationMember).where(
            OrganizationMember.organization_id == org_id,
            OrganizationMember.user_id == current_user.id,
        )
    )
    caller = member_result.scalar_one_or_none()
    if caller is None or caller.role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Only owners and admins can manage billing")

    result = await db.execute(
        select(Subscription).where(Subscription.organization_id == org_id)
    )
    sub = result.scalar_one_or_none()
    if sub is None:
        raise HTTPException(status_code=404, detail="No subscription found")

    sub.status = "canceled"
    sub.plan = "free"
    sub.canceled_at = datetime.now(UTC)

    org_result = await db.execute(
        select(Organization).where(Organization.id == org_id)
    )
    org = org_result.scalar_one_or_none()
    if org is not None:
        org.plan = "free"

    await db.commit()
    await db.refresh(sub)
    return SubscriptionResponse.model_validate(sub)
