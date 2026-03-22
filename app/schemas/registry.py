from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RegistryEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    environment_id: str
    algorithm: str
    stage: str
    model_version_id: int
    previous_production_id: int | None
    mean_reward: float | None
    promoted_by: int | None
    promotion_comment: str | None
    is_current: bool
    created_at: datetime
    updated_at: datetime


class PromoteRequest(BaseModel):
    model_version_id: int
    target_stage: str
    comment: str | None = None


class RollbackRequest(BaseModel):
    comment: str | None = None


class RegistryListResponse(BaseModel):
    items: list[RegistryEntry]
    total: int


class ComparisonResult(BaseModel):
    current_production: RegistryEntry | None
    candidate: RegistryEntry
    mean_reward_delta: float | None
    recommendation: str
