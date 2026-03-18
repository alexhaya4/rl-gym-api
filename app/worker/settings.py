from typing import ClassVar

from arq.connections import RedisSettings

from app.config import get_settings
from app.worker.tasks import run_training_job

settings = get_settings()

redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)


class WorkerSettings:
    """arq worker configuration."""

    functions: ClassVar[list] = [run_training_job]
    redis_settings = redis_settings
    max_jobs = 10
    job_timeout = 3600
    keep_result = 3600
