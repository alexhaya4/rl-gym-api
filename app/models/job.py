from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Job(Base):  # type: ignore[misc]
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    experiment_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("experiments.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(20), default="queued")
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    enqueued_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<Job(id={self.id!r}, status={self.status!r})>"
