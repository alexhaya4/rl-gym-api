from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ABTestCreate(BaseModel):
    name: str
    description: str | None = None
    environment_id: str
    model_version_a_id: int
    model_version_b_id: int
    traffic_split_a: float = Field(default=0.5, ge=0.1, le=0.9)
    n_eval_episodes_per_model: int = 100
    significance_level: float = 0.05
    statistical_test: str = "ttest"


class ABTestResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ab_test_id: int
    model_variant: str
    episode_number: int
    total_reward: float
    episode_length: int | None
    created_at: datetime


class ABTestStatistics(BaseModel):
    model_a_mean_reward: float | None
    model_a_std_reward: float | None
    model_a_n_episodes: int
    model_b_mean_reward: float | None
    model_b_std_reward: float | None
    model_b_n_episodes: int
    p_value: float | None
    test_statistic: float | None
    is_significant: bool
    winner: str | None
    confidence_level: float
    effect_size: float | None


class ABTestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    environment_id: str
    model_version_a_id: int
    model_version_b_id: int
    traffic_split_a: float
    status: str
    n_eval_episodes_per_model: int
    significance_level: float
    winner: str | None
    p_value: float | None
    test_statistic: float | None
    statistical_test: str
    user_id: int
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    statistics: ABTestStatistics | None = None


class ABTestListResponse(BaseModel):
    items: list[ABTestResponse]
    total: int
