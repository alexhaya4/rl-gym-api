from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class AuditLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    event_type: str
    user_id: int | None
    username: str | None
    ip_address: str | None
    user_agent: str | None
    resource_type: str | None
    resource_id: str | None
    action: str | None
    status: str
    details: dict[str, Any] | None
    request_id: str | None
    created_at: datetime


class AuditLogListResponse(BaseModel):
    items: list[AuditLogResponse]
    total: int
    page: int
    page_size: int


class AuditLogFilter(BaseModel):
    user_id: int | None = None
    event_type: str | None = None
    action: str | None = None
    status: str | None = None
    ip_address: str | None = None
    from_date: str | None = None
    to_date: str | None = None
    page: int = 1
    page_size: int = 50
