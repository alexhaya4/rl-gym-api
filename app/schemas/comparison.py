from typing import Any

from pydantic import BaseModel, Field


class ExperimentDiff(BaseModel):
    experiment_id_a: int
    experiment_id_b: int
    name_a: str
    name_b: str
    hyperparameter_diff: dict[str, Any]
    metrics_diff: dict[str, Any]
    status_a: str
    status_b: str
    winner: str | None
    improvement_pct: float | None


class ComparisonRequest(BaseModel):
    experiment_ids: list[int] = Field(min_length=2, max_length=10)


class ComparisonResponse(BaseModel):
    experiments: list[dict[str, Any]]
    diffs: list[ExperimentDiff]
    best_experiment_id: int | None
    comparison_metric: str = "mean_reward"
