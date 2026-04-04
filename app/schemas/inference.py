from datetime import datetime

from pydantic import BaseModel


class InferenceRequest(BaseModel):
    observation: list[float] | dict  # type: ignore[type-arg]
    algorithm: str | None = None
    deterministic: bool = True


class InferenceResponse(BaseModel):
    action: int | list[float]
    action_probability: float | None
    latency_ms: float
    model_version_id: int
    algorithm: str
    environment_id: str


class ModelCacheInfo(BaseModel):
    model_path: str
    algorithm: str
    environment_id: str
    loaded_at: datetime
    memory_mb: float
