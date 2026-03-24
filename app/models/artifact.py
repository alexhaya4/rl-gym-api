from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(50), nullable=False)
    storage_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    storage_backend: Mapped[str] = mapped_column(String(20), default="local")
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSON, nullable=True
    )
    experiment_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("experiments.id"), nullable=True
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def __repr__(self) -> str:
        return f"<Artifact(id={self.id}, name={self.name!r}, type={self.artifact_type!r})>"


class ArtifactLineage(Base):
    __tablename__ = "artifact_lineage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parent_experiment_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("experiments.id"), nullable=True
    )
    child_experiment_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("experiments.id"), nullable=False
    )
    relationship_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    def __repr__(self) -> str:
        return f"<ArtifactLineage(id={self.id}, parent={self.parent_experiment_id}, child={self.child_experiment_id})>"
