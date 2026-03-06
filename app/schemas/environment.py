from typing import Any

from pydantic import BaseModel, ConfigDict


class EnvironmentCreate(BaseModel):
    environment_id: str
    render_mode: str | None = None


class EnvironmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    env_key: str
    environment_id: str
    observation_space: dict[str, Any]
    action_space: dict[str, Any]
    status: str = "ready"


class StepRequest(BaseModel):
    action: int | list[float]


class StepResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class ResetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    observation: list[float]
    info: dict[str, Any]
