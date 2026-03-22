from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ModelRegistry(Base):  # type: ignore[misc]
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    environment_id: Mapped[str] = mapped_column(String(50), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)
    stage: Mapped[str] = mapped_column(String(20), nullable=False, default="development")
    model_version_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("model_versions.id"), nullable=False
    )
    previous_production_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("model_versions.id"), nullable=True
    )
    mean_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    promoted_by: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )
    promotion_comment: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    def __repr__(self) -> str:
        return f"<ModelRegistry(id={self.id}, name={self.name!r}, stage={self.stage!r})>"
