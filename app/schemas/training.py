from typing import Any

from pydantic import BaseModel, ConfigDict


class TrainingConfig(BaseModel):
    environment_id: str
    algorithm: str = "PPO"
    total_timesteps: int = 10000
    hyperparameters: dict[str, Any] = {}
    experiment_name: str | None = None


class TrainingStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    experiment_id: int
    status: str
    environment_id: str
    algorithm: str
    total_timesteps: int
    job_id: str | None = None
    elapsed_time: float | None = None
    mean_reward: float | None = None
    std_reward: float | None = None


class TrainingResult(TrainingStatus):
    model_path: str | None = None
    completed_at: str | None = None
