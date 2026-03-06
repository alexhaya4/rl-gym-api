from pydantic import BaseModel


class EvaluationRequest(BaseModel):
    experiment_id: int
    n_eval_episodes: int = 10
    deterministic: bool = True
    environment_id: str | None = None


class EpisodeMetrics(BaseModel):
    episode_number: int
    total_reward: float
    episode_length: int


class EvaluationResponse(BaseModel):
    experiment_id: int
    environment_id: str
    algorithm: str
    n_eval_episodes: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    episodes: list[EpisodeMetrics]
    evaluated_at: str
