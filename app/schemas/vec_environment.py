from typing import Any

from pydantic import BaseModel, Field


class VecEnvironmentCreate(BaseModel):
    environment_id: str
    n_envs: int = Field(default=4, ge=1, le=32)
    use_subprocess: bool = False
    normalize_observations: bool = False
    normalize_rewards: bool = False
    frame_stack: int | None = None
    seed: int | None = None


class VecEnvironmentResponse(BaseModel):
    vec_key: str
    environment_id: str
    n_envs: int
    use_subprocess: bool
    normalize_observations: bool
    normalize_rewards: bool
    frame_stack: int | None
    observation_space: dict[str, Any]
    action_space: dict[str, Any]
    status: str = "ready"


class VecStepRequest(BaseModel):
    actions: list[Any]


class VecStepResponse(BaseModel):
    observations: list[list[float]]
    rewards: list[float]
    terminated: list[bool]
    truncated: list[bool]
    infos: list[dict[str, Any]]
    n_envs: int


class VecResetResponse(BaseModel):
    observations: list[list[float]]
    infos: list[dict[str, Any]]
    n_envs: int
