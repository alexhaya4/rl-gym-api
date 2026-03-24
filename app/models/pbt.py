from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
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


class PBTExperiment(Base):
    __tablename__ = "pbt_experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    environment_id: Mapped[str] = mapped_column(String(50), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)
    population_size: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    total_timesteps_per_member: Mapped[int] = mapped_column(
        Integer, nullable=False, default=10000
    )
    exploit_interval: Mapped[int] = mapped_column(Integer, default=2000)
    mutation_rate: Mapped[float] = mapped_column(Float, default=0.2)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    n_generations: Mapped[int] = mapped_column(Integer, default=0)
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
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<PBTExperiment(id={self.id}, name={self.name!r}, status={self.status!r})>"


class PBTMember(Base):
    __tablename__ = "pbt_members"
    __table_args__ = (
        UniqueConstraint(
            "pbt_experiment_id", "member_index", name="uq_pbt_experiment_member"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pbt_experiment_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("pbt_experiments.id"), nullable=False
    )
    member_index: Mapped[int] = mapped_column(Integer, nullable=False)
    hyperparameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    mean_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    std_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_exploits: Mapped[int] = mapped_column(Integer, default=0)
    n_mutations: Mapped[int] = mapped_column(Integer, default=0)
    is_best: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    def __repr__(self) -> str:
        return f"<PBTMember(id={self.id}, pbt_experiment_id={self.pbt_experiment_id}, index={self.member_index})>"
