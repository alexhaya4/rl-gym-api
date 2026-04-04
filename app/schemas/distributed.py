from typing import Any

from pydantic import BaseModel, Field


class DistributedTrainRequest(BaseModel):
    environment_id: str
    algorithm: str = "PPO"
    total_timesteps: int = 50000
    num_workers: int = Field(default=2, ge=1, le=4)
    num_envs_per_worker: int = Field(default=4, ge=1, le=8)
    hyperparameters: dict[str, Any] = {}
    experiment_name: str | None = None


class DistributedTrainResponse(BaseModel):
    job_id: str
    status: str
    num_workers: int
    total_envs: int
    estimated_speedup: float


class DistributedStatus(BaseModel):
    job_id: str
    status: str  # queued/initializing/training/completed/failed/cancelled
    progress: float = 0.0
    metrics: dict[str, Any] | None = None
    elapsed_seconds: float = 0.0
    num_workers_active: int = 0
    error: str | None = None
