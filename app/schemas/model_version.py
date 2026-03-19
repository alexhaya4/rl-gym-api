from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    experiment_id: int
    version: int
    storage_path: str
    storage_backend: str
    algorithm: str
    total_timesteps: int | None
    mean_reward: float | None
    file_size_bytes: int | None
    metadata: dict[str, Any] | None
    created_at: datetime
    download_url: str | None = None


class ModelVersionListResponse(BaseModel):
    items: list[ModelVersionResponse]
    total: int
