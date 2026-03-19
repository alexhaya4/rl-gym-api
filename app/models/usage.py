from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class UsageRecord(Base):  # type: ignore[misc]
    __tablename__ = "usage_records"
    __table_args__ = (
        UniqueConstraint("organization_id", "month"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    organization_id: Mapped[int] = mapped_column(Integer, ForeignKey("organizations.id"), nullable=False)
    month: Mapped[str] = mapped_column(String(7), nullable=False)
    experiments_count: Mapped[int] = mapped_column(Integer, default=0)
    environments_count: Mapped[int] = mapped_column(Integer, default=0)
    total_timesteps: Mapped[int] = mapped_column(Integer, default=0)
    training_jobs_count: Mapped[int] = mapped_column(Integer, default=0)
    api_calls_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    def __repr__(self) -> str:
        return f"<UsageRecord(id={self.id}, org_id={self.organization_id}, month={self.month!r})>"
