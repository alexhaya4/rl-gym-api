from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("experiment_id", "version", name="uq_experiment_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("experiments.id"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    storage_backend: Mapped[str] = mapped_column(String(20), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)
    total_timesteps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mean_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSON, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def __repr__(self) -> str:
        return f"<ModelVersion(id={self.id}, experiment_id={self.experiment_id}, version={self.version})>"
