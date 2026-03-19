from typing import Any

from pydantic import BaseModel


class HyperparameterGrid(BaseModel):
    learning_rate: list[float] = [0.0003, 0.001, 0.003]
    n_steps: list[int] = [512, 1024, 2048]
    batch_size: list[int] = [32, 64, 128]
    gamma: list[float] = [0.95, 0.99]


class DistributedTrainingRequest(BaseModel):
    environment_id: str
    algorithm: str = "PPO"
    total_timesteps: int = 10000
    hyperparameter_grid: HyperparameterGrid = HyperparameterGrid()
    max_concurrent_trials: int = 4
    experiment_name: str | None = None
    optimization_metric: str = "mean_reward"


class TrialResult(BaseModel):
    trial_id: str
    hyperparameters: dict[str, Any]
    mean_reward: float
    std_reward: float
    training_time_seconds: float
    status: str


class DistributedTrainingResponse(BaseModel):
    job_id: str
    experiment_name: str
    total_trials: int
    status: str
    results: list[TrialResult] = []
    best_trial: TrialResult | None = None
    best_hyperparameters: dict[str, Any] | None = None
    started_at: str
    completed_at: str | None = None
