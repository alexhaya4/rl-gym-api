from pydantic import BaseModel


class PipelineRunRequest(BaseModel):
    environment_id: str
    algorithm: str = "PPO"
    total_timesteps: int = 10000
    hyperparameters: dict = {}
    experiment_name: str | None = None
    min_reward_threshold: float | None = None
    retrain_if_exists: bool = True
    schedule_cron: str | None = None


class PipelineStepResult(BaseModel):
    step_name: str
    status: str
    output: dict | None = None
    error: str | None = None
    duration_seconds: float


class PipelineRunResponse(BaseModel):
    pipeline_id: str
    experiment_name: str
    status: str
    steps: list[PipelineStepResult] = []
    started_at: str
    completed_at: str | None = None
    experiment_id: int | None = None
    model_version_id: int | None = None
    promoted: bool = False
    message: str | None = None
