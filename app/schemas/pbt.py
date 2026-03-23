from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PBTRequest(BaseModel):
    environment_id: str
    algorithm: str = "PPO"
    population_size: int = Field(default=8, ge=2, le=20)
    total_timesteps_per_member: int = 10000
    exploit_interval: int = 2000
    mutation_rate: float = Field(default=0.2, ge=0.0, le=1.0)
    experiment_name: str | None = None
    initial_hyperparameter_ranges: dict[str, Any] | None = None


class PBTMemberResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    pbt_experiment_id: int
    member_index: int
    hyperparameters: dict[str, Any]
    mean_reward: float | None
    std_reward: float | None
    n_exploits: int
    n_mutations: int
    is_best: bool
    created_at: datetime
    updated_at: datetime


class PBTExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    environment_id: str
    algorithm: str
    population_size: int
    total_timesteps_per_member: int
    exploit_interval: int
    mutation_rate: float
    status: str
    n_generations: int
    best_mean_reward: float | None
    best_hyperparameters: dict[str, Any] | None
    user_id: int
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    members: list[PBTMemberResponse] = []


class PBTListResponse(BaseModel):
    items: list[PBTExperimentResponse]
    total: int
