from typing import Any

from pydantic import BaseModel


class OptimizationRequest(BaseModel):
    environment_id: str
    algorithm: str = "PPO"
    total_timesteps: int = 10000
    n_trials: int = 20
    n_eval_episodes: int = 5
    experiment_name: str | None = None
    timeout_seconds: int | None = None
    pruning_enabled: bool = True
    hyperparameter_space: dict[str, Any] | None = None


class TrialInfo(BaseModel):
    trial_number: int
    status: str
    hyperparameters: dict[str, Any]
    mean_reward: float | None
    duration_seconds: float | None


class OptimizationResponse(BaseModel):
    study_id: str
    experiment_name: str
    status: str
    n_trials: int
    n_completed: int
    n_pruned: int
    best_trial: TrialInfo | None
    best_hyperparameters: dict[str, Any] | None
    best_mean_reward: float | None
    trials: list[TrialInfo] = []
    started_at: str
    completed_at: str | None = None
    improvement_over_default: float | None = None
