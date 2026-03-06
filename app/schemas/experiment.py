from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ExperimentBase(BaseModel):
    name: str
    environment_id: str
    algorithm: str
    hyperparameters: dict = {}


class ExperimentCreate(ExperimentBase):
    total_timesteps: int = 10000


class ExperimentUpdate(BaseModel):
    name: str | None = None
    status: str | None = None
    hyperparameters: dict | None = None


class ExperimentResponse(ExperimentBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    total_timesteps: int
    user_id: int
    created_at: datetime
    updated_at: datetime | None
    completed_at: datetime | None
    mean_reward: float | None
    std_reward: float | None


class ExperimentListResponse(BaseModel):
    items: list[ExperimentResponse]
    total: int
    page: int
    page_size: int
