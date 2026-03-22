from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class MultiAgentTrainingRequest(BaseModel):
    environment_id: str
    algorithm: str = "PPO"
    total_timesteps: int = 50000
    n_eval_episodes: int = 10
    experiment_name: str | None = None
    hyperparameters: dict[str, Any] = {}
    shared_policy: bool = False


class AgentPolicyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    experiment_id: int
    agent_id: str
    role: str | None
    mean_reward: float | None
    std_reward: float | None
    model_path: str | None
    created_at: datetime


class MultiAgentExperimentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    environment_id: str
    environment_type: str
    n_agents: int
    algorithm: str
    status: str
    total_timesteps: int | None
    n_episodes: int
    mean_team_reward: float | None
    std_team_reward: float | None
    hyperparameters: dict[str, Any] | None
    user_id: int
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    agent_policies: list[AgentPolicyResponse] = []


class MultiAgentExperimentListResponse(BaseModel):
    items: list[MultiAgentExperimentResponse]
    total: int
