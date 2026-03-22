from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class ArtifactCreate(BaseModel):
    name: str
    artifact_type: str
    experiment_id: int | None = None
    metadata: dict[str, Any] = {}


class ArtifactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    artifact_type: str
    storage_path: str | None
    storage_backend: str
    checksum: str | None
    size_bytes: int | None
    metadata_: dict[str, Any] | None
    experiment_id: int | None
    user_id: int
    created_at: datetime


class LineageNode(BaseModel):
    experiment_id: int
    experiment_name: str
    algorithm: str
    environment_id: str
    mean_reward: float | None
    parent_id: int | None
    children: list[int] = []
    tags: list[str] = []


class LineageGraph(BaseModel):
    nodes: list[LineageNode]
    edges: list[dict[str, Any]]
    root_experiment_id: int | None
