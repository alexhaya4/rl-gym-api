from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Security
    SECRET_KEY: str = "change-me-to-a-random-secret-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./rl_gym.db"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_COMMAND_TIMEOUT: int = 30

    # Application
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_PASSWORD: str | None = None

    # Model Storage
    STORAGE_BACKEND: str = "local"
    STORAGE_LOCAL_PATH: str = "./models"
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str | None = None
    S3_ENDPOINT_URL: str | None = None

    # gRPC
    GRPC_HOST: str = "0.0.0.0"  # nosec B104 - configurable via env var
    GRPC_PORT: int = 50051
    GRPC_API_KEY: str | None = None

    # Stripe
    STRIPE_SECRET_KEY: str | None = None
    STRIPE_WEBHOOK_SECRET: str | None = None
    STRIPE_FREE_PRICE_ID: str | None = None
    STRIPE_PRO_PRICE_ID: str | None = None
    STRIPE_ENTERPRISE_PRICE_ID: str | None = None

    # Tier Limits
    FREE_TIER_MAX_EXPERIMENTS: int = 5
    FREE_TIER_MAX_ENVIRONMENTS: int = 3
    FREE_TIER_MAX_TIMESTEPS: int = 50000
    PRO_TIER_MAX_EXPERIMENTS: int = 100
    PRO_TIER_MAX_ENVIRONMENTS: int = 20
    PRO_TIER_MAX_TIMESTEPS: int = 5000000

    # Ray
    RAY_ADDRESS: str | None = None

    # Metrics
    METRICS_TOKEN: str | None = None
    METRICS_ALLOWED_IPS: str = "127.0.0.1,::1"

    # Security Headers
    HSTS_MAX_AGE: int = 31536000

    # Request Size Limits
    MAX_REQUEST_SIZE_MB: int = 10

    # Sandbox
    SANDBOX_ENABLED: bool = True
    SANDBOX_TIMEOUT_SECONDS: int = 30
    SANDBOX_MEMORY_LIMIT: str = "128m"
    SANDBOX_CPU_LIMIT: float = 0.5

    # OAuth
    OAUTH_STATE_SECRET: str = "change-me-oauth-state-secret"
    GOOGLE_CLIENT_ID: str | None = None
    GOOGLE_CLIENT_SECRET: str | None = None
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/oauth/google/callback"
    GITHUB_CLIENT_ID: str | None = None
    GITHUB_CLIENT_SECRET: str | None = None
    GITHUB_REDIRECT_URI: str = "http://localhost:8000/api/v1/oauth/github/callback"

    # Inference
    INFERENCE_CACHE_MAX_MODELS: int = 5

    # Datasets
    DATASET_STORAGE_PATH: str = "/tmp/rl_datasets"  # nosec B108 - configurable via env var
    MAX_DATASET_SIZE_MB: int = 100
    ALLOWED_DATASET_EXTENSIONS: list[str] = [".csv", ".json", ".zip"]

    # Video
    VIDEO_STORAGE_PATH: str = "/tmp/rl_videos"  # nosec B108 - configurable via env var
    MAX_VIDEO_SIZE_MB: int = 100
    MAX_VIDEO_STEPS: int = 10000
    VIDEO_CLEANUP_HOURS: int = 1

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000,https://rlgymapi.com,https://www.rlgymapi.com,https://dashboard.rlgymapi.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()
