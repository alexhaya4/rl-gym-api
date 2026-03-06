import time
from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.main import VERSION

START_TIME = time.time()


async def get_health_status(db: AsyncSession) -> dict:
    settings = get_settings()

    db_status = "connected"
    db_error = None
    try:
        await db.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = "disconnected"
        db_error = str(exc)

    status = "healthy" if db_status == "connected" else "degraded"

    result = {
        "status": status,
        "version": VERSION,
        "environment": settings.ENVIRONMENT,
        "database": db_status,
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "timestamp": datetime.now(UTC).isoformat(),
    }

    if db_error:
        result["database_error"] = db_error

    return result
