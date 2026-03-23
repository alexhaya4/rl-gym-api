from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Dataset(Base):  # type: ignore[misc]
    __tablename__ = "datasets"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_dataset_name_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    environment_id: Mapped[str] = mapped_column(String(50), nullable=False)
    algorithm: Mapped[str | None] = mapped_column(String(50), nullable=True)
    n_episodes: Mapped[int] = mapped_column(Integer, default=0)
    n_transitions: Mapped[int] = mapped_column(Integer, default=0)
    mean_episode_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    std_episode_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_episode_length: Mapped[float | None] = mapped_column(Float, nullable=True)
    storage_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    storage_format: Mapped[str] = mapped_column(String(20), default="json")
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSON, nullable=True
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
        return f"<Dataset(id={self.id}, name={self.name!r}, version={self.version})>"


class DatasetEpisode(Base):  # type: ignore[misc]
    __tablename__ = "dataset_episodes"
    __table_args__ = (
        Index("ix_dataset_episodes_dataset_id", "dataset_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("datasets.id"), nullable=False
    )
    episode_number: Mapped[int] = mapped_column(Integer, nullable=False)
    total_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    episode_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    observations: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    actions: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    rewards: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    terminated: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def __repr__(self) -> str:
        return f"<DatasetEpisode(id={self.id}, dataset_id={self.dataset_id}, episode={self.episode_number})>"
