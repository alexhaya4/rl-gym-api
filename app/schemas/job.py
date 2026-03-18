from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    experiment_id: int | None
    status: str
    result: dict[str, Any] | None
    error: str | None
    enqueued_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
