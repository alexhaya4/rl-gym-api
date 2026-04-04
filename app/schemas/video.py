from pydantic import BaseModel, Field


class VideoRequest(BaseModel):
    environment_id: str
    algorithm: str = "PPO"
    num_episodes: int = Field(default=1, ge=1, le=5)
    max_steps: int = Field(default=500, ge=1, le=10000)
    fps: int = Field(default=30, ge=1, le=60)


class VideoResponse(BaseModel):
    video_id: str
    status: str
    video_url: str | None
    num_episodes: int
    total_steps: int
    total_reward: float
    duration_seconds: float
    file_size_mb: float


class VideoStatus(BaseModel):
    video_id: str
    status: str  # queued/recording/encoding/complete/failed
    progress: float = 0.0  # 0-100
    error: str | None = None
