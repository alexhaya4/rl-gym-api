from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MLTrainRequest(BaseModel):
    dataset_id: int
    algorithm: str
    target_column: str | None = None
    hyperparameters: dict[str, Any] = {}
    test_split: float = Field(default=0.2, ge=0.1, le=0.5)
    task_type: str


class MLTrainResponse(BaseModel):
    model_id: int
    algorithm: str
    task_type: str
    metrics: dict[str, Any]
    training_time_seconds: float
    feature_importance: list[dict[str, Any]] | None = None
    nan_rows_dropped: int = 0


class MLPredictRequest(BaseModel):
    model_id: int
    features: list[list[float]]


class MLPredictResponse(BaseModel):
    predictions: list[int | float]
    probabilities: list[list[float]] | None = None
    model_id: int
    inference_time_ms: float


class MLModelInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    algorithm: str
    task_type: str
    dataset_id: int | None
    metrics: dict[str, Any] | None
    created_at: datetime
