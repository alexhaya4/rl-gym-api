from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class PlanInfo(BaseModel):
    name: str
    price_monthly_usd: float
    max_experiments: int
    max_environments: int
    max_timesteps: int
    features: list[str]


class SubscriptionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    plan: str
    status: str
    current_period_start: datetime | None
    current_period_end: datetime | None
    canceled_at: datetime | None
    created_at: datetime


class UsageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    month: str
    experiments_count: int
    environments_count: int
    total_timesteps: int
    training_jobs_count: int
    api_calls_count: int
    limits: dict[str, Any]


class CreateCheckoutSessionRequest(BaseModel):
    plan: str
    success_url: str
    cancel_url: str


class CheckoutSessionResponse(BaseModel):
    checkout_url: str
    session_id: str
