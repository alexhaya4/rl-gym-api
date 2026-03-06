from pydantic import BaseModel


class BenchmarkRequest(BaseModel):
    environments: list[str]
    algorithms: list[str]
    total_timesteps: int = 5000
    n_eval_episodes: int = 5


class BenchmarkResult(BaseModel):
    environment_id: str
    algorithm: str
    mean_reward: float
    std_reward: float
    training_time_seconds: float
    total_timesteps: int


class BenchmarkResponse(BaseModel):
    benchmark_id: str
    results: list[BenchmarkResult]
    total_combinations: int
    completed_at: str
