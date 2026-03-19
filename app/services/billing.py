import logging
from datetime import UTC, datetime
from typing import Any

import stripe
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.subscription import Subscription
from app.schemas.billing import PlanInfo, SubscriptionResponse

logger = logging.getLogger(__name__)

PLANS: dict[str, PlanInfo] = {
    "free": PlanInfo(
        name="Free",
        price_monthly_usd=0.0,
        max_experiments=5,
        max_environments=3,
        max_timesteps=50000,
        features=[
            "5 experiments per month",
            "3 concurrent environments",
            "50k training timesteps",
            "Community support",
        ],
    ),
    "pro": PlanInfo(
        name="Pro",
        price_monthly_usd=49.0,
        max_experiments=100,
        max_environments=20,
        max_timesteps=5000000,
        features=[
            "100 experiments per month",
            "20 concurrent environments",
            "5M training timesteps",
            "Priority support",
            "Model versioning",
            "gRPC inference API",
        ],
    ),
    "enterprise": PlanInfo(
        name="Enterprise",
        price_monthly_usd=199.0,
        max_experiments=-1,
        max_environments=-1,
        max_timesteps=-1,
        features=[
            "Unlimited experiments",
            "Unlimited environments",
            "Unlimited training timesteps",
            "Dedicated support",
            "Model versioning",
            "gRPC inference API",
            "Custom environments",
            "SLA guarantee",
        ],
    ),
}


async def create_stripe_customer(email: str, org_name: str) -> str:
    settings = get_settings()
    if settings.STRIPE_SECRET_KEY is None:
        return "test_customer"

    stripe.api_key = settings.STRIPE_SECRET_KEY
    customer = stripe.Customer.create(
        email=email,
        name=org_name,
        metadata={"source": "rl-gym-api"},
    )
    return customer.id


async def create_checkout_session(
    customer_id: str, price_id: str, success_url: str, cancel_url: str
) -> dict[str, str]:
    settings = get_settings()
    if settings.STRIPE_SECRET_KEY is None:
        return {
            "checkout_url": f"{success_url}?session_id=test_session",
            "session_id": "test_session",
        }

    stripe.api_key = settings.STRIPE_SECRET_KEY
    session = stripe.checkout.Session.create(
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        mode="subscription",
        success_url=success_url,
        cancel_url=cancel_url,
    )
    return {
        "checkout_url": session.url or "",
        "session_id": session.id,
    }


def _price_id_to_plan(price_id: str) -> str:
    settings = get_settings()
    if price_id == settings.STRIPE_PRO_PRICE_ID:
        return "pro"
    if price_id == settings.STRIPE_ENTERPRISE_PRICE_ID:
        return "enterprise"
    if price_id == settings.STRIPE_FREE_PRICE_ID:
        return "free"
    return "free"


async def handle_webhook_event(
    payload: bytes, sig_header: str, db: AsyncSession
) -> dict[str, Any]:
    settings = get_settings()
    if settings.STRIPE_SECRET_KEY is None or settings.STRIPE_WEBHOOK_SECRET is None:
        return {"status": "skipped", "reason": "Stripe not configured"}

    stripe.api_key = settings.STRIPE_SECRET_KEY
    event = stripe.Webhook.construct_event(  # type: ignore[no-untyped-call]
        payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
    )

    event_type = event["type"]
    data_object = event["data"]["object"]
    logger.info("Processing Stripe webhook: %s", event_type)

    if event_type == "customer.subscription.created":
        customer_id = data_object["customer"]
        subscription_id = data_object["id"]
        plan = _price_id_to_plan(data_object["items"]["data"][0]["price"]["id"])

        result = await db.execute(
            select(Subscription).where(
                Subscription.stripe_customer_id == customer_id
            )
        )
        sub = result.scalar_one_or_none()
        if sub is None:
            sub = Subscription(
                organization_id=0,
                stripe_subscription_id=subscription_id,
                stripe_customer_id=customer_id,
                plan=plan,
                status=data_object["status"],
                current_period_start=datetime.fromtimestamp(
                    data_object["current_period_start"], tz=UTC
                ),
                current_period_end=datetime.fromtimestamp(
                    data_object["current_period_end"], tz=UTC
                ),
            )
            db.add(sub)
        else:
            sub.stripe_subscription_id = subscription_id
            sub.plan = plan
            sub.status = data_object["status"]
            sub.current_period_start = datetime.fromtimestamp(
                data_object["current_period_start"], tz=UTC
            )
            sub.current_period_end = datetime.fromtimestamp(
                data_object["current_period_end"], tz=UTC
            )
        await db.commit()
        return {"status": "processed", "event": event_type, "plan": plan}

    if event_type == "customer.subscription.updated":
        subscription_id = data_object["id"]
        plan = _price_id_to_plan(data_object["items"]["data"][0]["price"]["id"])

        result = await db.execute(
            select(Subscription).where(
                Subscription.stripe_subscription_id == subscription_id
            )
        )
        sub = result.scalar_one_or_none()
        if sub is not None:
            sub.plan = plan
            sub.status = data_object["status"]
            sub.current_period_start = datetime.fromtimestamp(
                data_object["current_period_start"], tz=UTC
            )
            sub.current_period_end = datetime.fromtimestamp(
                data_object["current_period_end"], tz=UTC
            )
            await db.commit()
        return {"status": "processed", "event": event_type, "plan": plan}

    if event_type == "customer.subscription.deleted":
        subscription_id = data_object["id"]

        result = await db.execute(
            select(Subscription).where(
                Subscription.stripe_subscription_id == subscription_id
            )
        )
        sub = result.scalar_one_or_none()
        if sub is not None:
            sub.status = "canceled"
            sub.canceled_at = datetime.now(UTC)
            await db.commit()
        return {"status": "processed", "event": event_type}

    return {"status": "ignored", "event": event_type}


async def get_subscription_status(
    organization_id: int, db: AsyncSession
) -> SubscriptionResponse | None:
    result = await db.execute(
        select(Subscription).where(
            Subscription.organization_id == organization_id
        )
    )
    sub = result.scalar_one_or_none()
    if sub is None:
        return None
    return SubscriptionResponse.model_validate(sub)
