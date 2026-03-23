from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DatasetCreate(BaseModel):
    name: str
    description: str | None = None
    environment_id: str
    algorithm: str | None = None
    storage_format: str = "json"
    is_public: bool = False
    tags: list[str] = []
    metadata: dict[str, Any] = {}


class DatasetEpisodeCreate(BaseModel):
    episode_number: int
    total_reward: float
    episode_length: int
    observations: list[list[float]]
    actions: list[Any]
    rewards: list[float]
    terminated: bool = False


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    version: int
    description: str | None
    environment_id: str
    algorithm: str | None
    n_episodes: int
    n_transitions: int
    mean_episode_reward: float | None
    std_episode_reward: float | None
    mean_episode_length: float | None
    storage_path: str | None
    storage_format: str
    size_bytes: int | None
    is_public: bool
    tags: list[str] | None
    metadata_: dict[str, Any] | None
    user_id: int
    created_at: datetime
    updated_at: datetime


class DatasetEpisodeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    dataset_id: int
    episode_number: int
    total_reward: float | None
    episode_length: int | None
    observations: list[Any] | None
    actions: list[Any] | None
    rewards: list[float] | None
    terminated: bool
    created_at: datetime


class DatasetListResponse(BaseModel):
    items: list[DatasetResponse]
    total: int


class DatasetStatsResponse(BaseModel):
    dataset_id: int
    n_episodes: int
    n_transitions: int
    mean_episode_reward: float | None
    std_episode_reward: float | None
    min_episode_reward: float | None
    max_episode_reward: float | None
    mean_episode_length: float | None
    reward_distribution: list[float]
