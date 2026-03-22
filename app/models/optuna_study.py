from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class OptunaStudy(Base):  # type: ignore[misc]
    __tablename__ = "optuna_studies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    study_name: Mapped[str] = mapped_column(String(200), nullable=False)
    environment_id: Mapped[str] = mapped_column(String(50), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="running")
    n_trials: Mapped[int] = mapped_column(Integer, nullable=False)
    n_completed: Mapped[int] = mapped_column(Integer, default=0)
    n_pruned: Mapped[int] = mapped_column(Integer, default=0)
    best_mean_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_hyperparameters: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    def __repr__(self) -> str:
        return f"<OptunaStudy(id={self.id}, study_name={self.study_name!r}, status={self.status!r})>"
