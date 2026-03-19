from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.usage import UsageRecord


def _current_month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def get_plan_limits(plan: str) -> dict[str, Any]:
    settings = get_settings()
    if plan == "enterprise":
        return {
            "max_experiments": -1,
            "max_environments": -1,
            "max_timesteps": -1,
        }
    if plan == "pro":
        return {
            "max_experiments": settings.PRO_TIER_MAX_EXPERIMENTS,
            "max_environments": settings.PRO_TIER_MAX_ENVIRONMENTS,
            "max_timesteps": settings.PRO_TIER_MAX_TIMESTEPS,
        }
    return {
        "max_experiments": settings.FREE_TIER_MAX_EXPERIMENTS,
        "max_environments": settings.FREE_TIER_MAX_ENVIRONMENTS,
        "max_timesteps": settings.FREE_TIER_MAX_TIMESTEPS,
    }


async def get_or_create_usage(
    db: AsyncSession, organization_id: int, month: str | None = None
) -> UsageRecord:
    month = month or _current_month()
    result = await db.execute(
        select(UsageRecord).where(
            UsageRecord.organization_id == organization_id,
            UsageRecord.month == month,
        )
    )
    usage = result.scalar_one_or_none()
    if usage is None:
        usage = UsageRecord(organization_id=organization_id, month=month)
        db.add(usage)
        await db.flush()
    return usage


async def check_experiment_quota(
    db: AsyncSession, organization_id: int, plan: str
) -> tuple[bool, str]:
    limits = get_plan_limits(plan)
    max_experiments = limits["max_experiments"]
    if max_experiments == -1:
        return (True, "")
    usage = await get_or_create_usage(db, organization_id)
    if usage.experiments_count >= max_experiments:
        return (False, "Experiment quota exceeded")
    return (True, "")


async def check_environment_quota(
    db: AsyncSession, organization_id: int, plan: str
) -> tuple[bool, str]:
    limits = get_plan_limits(plan)
    max_environments = limits["max_environments"]
    if max_environments == -1:
        return (True, "")
    usage = await get_or_create_usage(db, organization_id)
    if usage.environments_count >= max_environments:
        return (False, "Environment quota exceeded")
    return (True, "")


async def check_timesteps_quota(
    db: AsyncSession, organization_id: int, plan: str, requested_timesteps: int
) -> tuple[bool, str]:
    limits = get_plan_limits(plan)
    max_timesteps = limits["max_timesteps"]
    if max_timesteps == -1:
        return (True, "")
    usage = await get_or_create_usage(db, organization_id)
    if usage.total_timesteps + requested_timesteps > max_timesteps:
        return (False, "Timesteps quota exceeded")
    return (True, "")


async def increment_usage(
    db: AsyncSession, organization_id: int, field: str, amount: int = 1
) -> None:
    usage = await get_or_create_usage(db, organization_id)
    current = getattr(usage, field)
    setattr(usage, field, current + amount)
    await db.flush()
