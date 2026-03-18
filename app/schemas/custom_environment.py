from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CustomEnvironmentCreate(BaseModel):
    name: str = Field(
        ..., pattern=r"^[A-Za-z0-9_-]+-v\d+$", examples=["MyEnv-v1"]
    )
    description: str | None = None
    source_code: str
    observation_space_spec: dict[str, Any] | None = None
    action_space_spec: dict[str, Any] | None = None


class CustomEnvironmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    entry_point: str
    is_validated: bool
    validation_error: str | None
    user_id: int
    created_at: datetime
    updated_at: datetime
    observation_space_spec: dict[str, Any] | None
    action_space_spec: dict[str, Any] | None


class CustomEnvironmentListResponse(BaseModel):
    items: list[CustomEnvironmentResponse]
    total: int
