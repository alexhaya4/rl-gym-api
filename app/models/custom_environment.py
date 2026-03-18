from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CustomEnvironment(Base):  # type: ignore[misc]
    __tablename__ = "custom_environments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    entry_point: Mapped[str] = mapped_column(String(255), nullable=False)
    source_code: Mapped[str] = mapped_column(Text, nullable=False)
    observation_space_spec: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
    action_space_spec: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
    is_validated: Mapped[bool] = mapped_column(Boolean, default=False)
    validation_error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    def __repr__(self) -> str:
        return f"<CustomEnvironment(id={self.id}, name={self.name!r}, is_validated={self.is_validated})>"
