from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.dependencies import get_current_active_user
from app.models.audit_log import AuditLog
from app.models.user import User
from app.schemas.audit_log import AuditLogFilter, AuditLogListResponse, AuditLogResponse
from app.services.audit_log import get_user_audit_trail, query_audit_logs

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/logs", response_model=AuditLogListResponse)
async def list_audit_logs(
    user_id: int | None = Query(None),
    event_type: str | None = Query(None),
    action: str | None = Query(None),
    audit_status: str | None = Query(None, alias="status"),
    ip_address: str | None = Query(None),
    from_date: str | None = Query(None),
    to_date: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> AuditLogListResponse:
    """Query audit logs with filters. Users can only see their own logs."""
    filters = AuditLogFilter(
        user_id=current_user.id,
        event_type=event_type,
        action=action,
        status=audit_status,
        ip_address=ip_address,
        from_date=from_date,
        to_date=to_date,
        page=page,
        page_size=page_size,
    )
    items, total = await query_audit_logs(db, filters)
    return AuditLogListResponse(
        items=[AuditLogResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/logs/me", response_model=list[AuditLogResponse])
async def my_audit_trail(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[AuditLogResponse]:
    """Get the current user's recent audit trail (last 100 events)."""
    items = await get_user_audit_trail(db, current_user.id, limit=100)
    return [AuditLogResponse.model_validate(i) for i in items]


@router.get("/logs/{log_id}", response_model=AuditLogResponse)
async def get_audit_log(
    log_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> AuditLogResponse:
    """Get a single audit log entry by ID."""
    result = await db.execute(
        select(AuditLog).where(
            AuditLog.id == log_id,
            AuditLog.user_id == current_user.id,
        )
    )
    entry = result.scalar_one_or_none()
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audit log entry not found",
        )
    return AuditLogResponse.model_validate(entry)
