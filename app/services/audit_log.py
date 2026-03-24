import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog
from app.schemas.audit_log import AuditLogFilter

logger = logging.getLogger(__name__)


async def log_event(
    db: AsyncSession,
    event_type: str,
    request: Request | None = None,
    user_id: int | None = None,
    username: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    action: str | None = None,
    status: str = "success",
    details: dict[str, Any] | None = None,
) -> AuditLog | None:
    """Create an audit log entry. Never raises exceptions."""
    try:
        ip_address = None
        user_agent = None
        request_id = None

        if request is not None:
            if request.client:
                ip_address = request.client.host
            user_agent = request.headers.get("user-agent")
            request_id = getattr(request.state, "request_id", None)

        audit_log = AuditLog(
            event_type=event_type,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            status=status,
            details=details,
            request_id=request_id,
        )
        db.add(audit_log)
        await db.commit()
        await db.refresh(audit_log)
        return audit_log
    except Exception:
        logger.exception("Failed to write audit log event: %s", event_type)
        return None


async def query_audit_logs(
    db: AsyncSession, filters: AuditLogFilter
) -> tuple[list[AuditLog], int]:
    """Query audit logs with filtering and pagination."""
    query = select(AuditLog)

    if filters.user_id is not None:
        query = query.where(AuditLog.user_id == filters.user_id)
    if filters.event_type is not None:
        query = query.where(AuditLog.event_type == filters.event_type)
    if filters.action is not None:
        query = query.where(AuditLog.action == filters.action)
    if filters.status is not None:
        query = query.where(AuditLog.status == filters.status)
    if filters.ip_address is not None:
        query = query.where(AuditLog.ip_address == filters.ip_address)
    if filters.from_date is not None:
        from_dt = datetime.fromisoformat(filters.from_date).replace(tzinfo=UTC)
        query = query.where(AuditLog.created_at >= from_dt)
    if filters.to_date is not None:
        to_dt = datetime.fromisoformat(filters.to_date).replace(tzinfo=UTC)
        query = query.where(AuditLog.created_at <= to_dt)

    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    offset = (filters.page - 1) * filters.page_size
    query = query.order_by(AuditLog.created_at.desc()).offset(offset).limit(filters.page_size)

    result = await db.execute(query)
    items = list(result.scalars().all())

    return items, total


async def get_user_audit_trail(
    db: AsyncSession, user_id: int, limit: int = 100
) -> list[AuditLog]:
    """Return recent audit events for a specific user."""
    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.user_id == user_id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())
