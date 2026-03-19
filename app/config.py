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

    # Application
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Model Storage
    STORAGE_BACKEND: str = "local"
    STORAGE_LOCAL_PATH: str = "./models"
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str | None = None
    S3_ENDPOINT_URL: str | None = None

    # gRPC
    GRPC_PORT: int = 50051

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

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
